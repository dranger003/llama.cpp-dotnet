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

app.MapGet("/session/configure", async (HttpContext httpContext, string sessionName, StringList initialContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.ConfigureSession(sessionName, initialContext);
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/session/predict", async (HttpContext httpContext, string sessionName, string prompt) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(sessionName);

    httpContext.Response.ContentType = "text/event-stream";

    await foreach (var token in session.Predict(prompt, cts.Token))
    {
        await httpContext.Response.WriteAsync(token);
        await httpContext.Response.Body.FlushAsync();
    }
});

await app.RunAsync();
