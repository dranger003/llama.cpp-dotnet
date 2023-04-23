using LlamaCppLib;
using LlamaCppWeb;
using System.Net;

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

app.MapGet("/model/configure", async (HttpContext httpContext, int threadCount = 4, int topK = 50, float topP = 0.95f, float temperature = 0.1f, float repeatPenalty = 1.1f) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.ConfigureModel(options =>
    {
        options.ThreadCount = threadCount;
        options.TopK = topK;
        options.TopP = topP;
        options.Temperature = temperature;
        options.RepeatPenalty = repeatPenalty;
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

app.MapGet("/session/configure", async (HttpContext httpContext, string sessionName) =>
{
    // TODO
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
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

app.MapGet("/debug/run", async (HttpContext httpContext) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.LoadModel("vicuna-13b-v1.1");
    manager.ConfigureModel(options =>
    {
        options.ThreadCount = 16;
        options.TopK = 50;
        options.TopP = 0.95f;
        options.Temperature = 0.1f;
        options.RepeatPenalty = 1.1f;
    });

    var sessionName = "Conversation #1";
    var session = manager.CreateSession(sessionName);

    httpContext.Response.ContentType = "text/event-stream";

    var prompts = new[]
    {
        "Describe quantum physics.",
        //"Hi! How can I be of service today?",
        //"Hello! How are you doing?",
        //"I am doing great! Thanks for asking.",
        //"Can you help me with some questions please?",
        //"Absolutely, what questions can I help you with?",
        //"How many planets are there in the solar system?",
        //"Can you list the planets of our solar system?",
        //"What do you think Vicuna 13B is according to you?",
        //"Vicuna 13B is a large language model (LLM).",
    };

    foreach (var prompt in prompts)
    {
        await httpContext.Response.WriteAsync($"{prompt}\n");
        await httpContext.Response.Body.FlushAsync();

        await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
        {
            await httpContext.Response.WriteAsync(token);
            await httpContext.Response.Body.FlushAsync();
        }
    }
});

app.MapGet("/debug/context", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession("Conversation #1");
    await httpContext.Response.WriteAsync(session.Conversation);
});

await app.RunAsync();
