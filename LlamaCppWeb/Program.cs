using LlamaCppLib;
using LlamaCppWeb;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<LlamaCppManager>();
builder.Services.AddCors();

var app = builder.Build();

app.UseCors(configure => configure.AllowAnyOrigin());

app.MapGet("/", async (HttpContext httpContext) =>
{
    await httpContext.Response.WriteAsync("Welcome to LLaMA C++!");
});

// Model Operations

app.MapGet("/model/list", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsJsonAsync(new { manager.Models });
});

app.MapGet("/model/load", (HttpContext httpContext, string modelName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.LoadModel(modelName);
});

app.MapGet("/model/unload", (HttpContext httpContext, string modelName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.UnloadModel();
});

app.MapGet("/model/status", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsJsonAsync(new { Status = Enum.GetName(manager.Status) });
});

// Session Operations

app.MapGet("/session/list", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsJsonAsync(new { manager.Sessions });
});

app.MapGet("/session/create", (HttpContext httpContext, string sessionName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.CreateSession(sessionName);
});

app.MapGet("/session/destroy", (HttpContext httpContext, string sessionName) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.DestroySession(sessionName);
});

app.MapGet("/session/configure", (HttpContext httpContext, string sessionName, StringList initialContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.ConfigureSession(sessionName, initialContext);
});

app.MapGet("/session/predict", async (HttpContext httpContext, string sessionName, string prompt) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var session = manager.GetSession(sessionName);

    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    httpContext.Response.ContentType = "text/event-stream";

    await foreach (var token in session.Predict(prompt, cts.Token))
    {
        await httpContext.Response.WriteAsync(token);
        await httpContext.Response.Body.FlushAsync();
    }
});

await app.RunAsync();
