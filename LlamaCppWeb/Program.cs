using System.Net;
using System.Text;
using System.Text.Json;

using LlamaCppLib;
using LlamaCppWeb;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.ModelBinding;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<LlamaCppManager>();
builder.Services.AddCors();

var app = builder.Build();

app.UseCors(configure => configure.AllowAnyOrigin());

app.MapGet("/", async (HttpContext httpContext) =>
{
    await httpContext.Response.WriteAsync("Welcome to LLaMA C++!");
});

// Model endpoints

app.MapGet("/model/list", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsJsonAsync(new { manager.Models });
});

app.MapGet("/model/load", async (HttpContext httpContext, string modelName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.LoadModel(modelName);
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/model/unload", async (HttpContext httpContext, string modelName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.UnloadModel();
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/model/status", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsJsonAsync(new { Status = Enum.GetName(manager.Status), manager.ModelName });
});

app.MapGet("/model/configure", async (HttpContext httpContext, int? threadCount, int? topK, float? topP, float? temperature, float? repeatPenalty) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.ConfigureModel(options =>
    {
        options.ThreadCount = threadCount ?? options.ThreadCount;
        options.TopK = topK ?? options.TopK;
        options.TopP = topP ?? options.TopP;
        options.Temperature = temperature ?? options.Temperature;
        options.RepeatPenalty = repeatPenalty ?? options.RepeatPenalty;
    });
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

// Session endpoints

app.MapGet("/session/list", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsJsonAsync(new { manager.Sessions });
});

app.MapGet("/session/create", async (HttpContext httpContext, string sessionName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.CreateSession(sessionName);
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/session/destroy", async (HttpContext httpContext, string sessionName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.DestroySession(sessionName);
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/session/configure", async (HttpContext httpContext, string sessionName, string template) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(sessionName);
    session.Configure(options => options.Template = template);
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/session/predict", async (HttpContext httpContext, string sessionName, string prompt) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(sessionName);

    httpContext.Response.ContentType = "text/event-stream";

    await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
    {
        await httpContext.Response.WriteAsync(token);
        await httpContext.Response.Body.FlushAsync();
    }
});

// curl http://localhost:5021/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer $API_KEY" -d "{ \"model\": \"tulu-7b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello?\"}] }"
app.MapPost("/v1/chat/completions", async (HttpContext httpContext, CreateChatCompletion createChatCompletion) =>
{
    //Console.WriteLine(JsonSerializer.Serialize(createChatCompletion, new JsonSerializerOptions { WriteIndented = true }));

    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    if (manager.Status == LlamaCppModelStatus.Unloaded || manager.ModelName != createChatCompletion.Model)
    {
        manager.LoadModel(createChatCompletion.Model ?? String.Empty);
        manager.ConfigureModel(options =>
        {
            options.TopP = createChatCompletion.TopP;
            options.Temperature = createChatCompletion.Temperature;
            options.RepeatPenalty = createChatCompletion.PresencePenalty;
        });
    }

    var sessionName = $"{httpContext.Request.Headers.Authorization}";
    if (manager.Sessions.SingleOrDefault(name => name == sessionName) == null)
    {
        manager.CreateSession(sessionName);
        // TODO: Set prompting template according to loaded model
        manager.GetSession(sessionName).Configure(session => session.Template = "You are a helpful assistant.\n\nUSER:\n{prompt}\n\nASSISTANT:\n");
    }

    var session = manager.GetSession(sessionName);

    if (!createChatCompletion.Stream)
    {
        var output = new StringBuilder();
        await foreach (var token in session.Predict(createChatCompletion.Messages[0].Content ?? String.Empty, cancellationToken: cts.Token))
        {
            Console.Write(token);
            output.Append(token);
        }
        Console.WriteLine();

        await httpContext.Response.WriteAsync(output.ToString());
    }
    else
    {
        httpContext.Response.ContentType = "text/event-stream";

        await foreach (var token in session.Predict(createChatCompletion.Messages[0].Content ?? String.Empty, cancellationToken: cts.Token))
        {
            await httpContext.Response.WriteAsync(token);
            await httpContext.Response.Body.FlushAsync();
        }
    }
});

app.MapGet("/debug/load", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    //var modelName = "vicuna-13b-v1.1-q5_1";
    var modelName = "wizard-vicuna-13b-q5_1";
    var sessionName = "Conversation #1";
    manager.LoadModel(modelName);
    var session = manager.CreateSession(sessionName);
    session.Configure(options => options.Template = "USER:\n{prompt}\n\nASSISTANT:\n");
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/debug/run", async (HttpContext httpContext, string prompt) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var sessionName = "Conversation #1";
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(sessionName);
    session.Reset();

    httpContext.Response.ContentType = "text/event-stream";

    await httpContext.Response.Body.WriteAsync(Encoding.UTF8.GetBytes($"data: <|BOS|>\n\n"));
    await httpContext.Response.Body.FlushAsync();

    try
    {
        await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
        {
            var encodedData = token.Replace("\n", "\\n");
            await httpContext.Response.Body.WriteAsync(Encoding.UTF8.GetBytes($"data: {encodedData}\n\n"), cts.Token);
            await httpContext.Response.Body.FlushAsync();
        }
    }
    catch (OperationCanceledException)
    { }

    await httpContext.Response.Body.WriteAsync(Encoding.UTF8.GetBytes($"data: <|EOS|>\n\n"));
    await httpContext.Response.Body.FlushAsync();
});

app.MapGet("/debug/context", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(manager.Sessions.First());
    await httpContext.Response.WriteAsync($"<|BOS|>{session.Conversation}<|EOS|>");
});

app.MapGet("/debug/template", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(manager.Sessions.First());
    await httpContext.Response.WriteAsync($"[{session.Options.Template}]");
});

await app.RunAsync();
