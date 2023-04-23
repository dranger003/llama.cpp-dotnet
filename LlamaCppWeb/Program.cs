using LlamaCppLib;
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

app.MapGet("/debug/context", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(manager.Sessions.First());
    await httpContext.Response.WriteAsync($"<|BOS|>{session.Conversation}<|EOD|>");
});

await app.RunAsync();
