using LlamaCppLib;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<LlamaCppLoader>();

builder.Services.AddCors();

var app = builder.Build();

app.UseCors(configure =>
{
    configure.AllowAnyOrigin();
});

app.MapGet("/", async (HttpContext httpContext) =>
{
    await httpContext.Response.WriteAsync("Welcome to LLaMA C++!");
});

// Model Operations

app.MapGet("/model/list", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    await httpContext.Response.WriteAsJsonAsync(loader.Models);
});

app.MapGet("/model/load", async (HttpContext httpContext, string modelName) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    loader.Load(modelName, out var initialContext);
    await httpContext.Response.WriteAsJsonAsync(new { ModelName = modelName });
});

app.MapGet("/model/unload", async (HttpContext httpContext, string modelName) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    loader.Unload();
    await httpContext.Response.WriteAsJsonAsync(new { ModelName = modelName });
});

app.MapGet("/model/status", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    await httpContext.Response.WriteAsJsonAsync(new { ModelName = loader.Model != null ? loader.Model.ModelName : String.Empty });
});

// Session Operations

app.MapGet("/session/new", async (HttpContext httpContext, string sessionName) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    loader.NewSession(sessionName);
    await httpContext.Response.WriteAsJsonAsync(new { SessionName = sessionName });
});

app.MapGet("/session/delete", async (HttpContext httpContext, string sessionName) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    loader.DeleteSession(sessionName);
    await httpContext.Response.WriteAsJsonAsync(new { SessionName = sessionName });
});

app.MapGet("/session/list", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    await httpContext.Response.WriteAsJsonAsync(loader.Sessions);
});

app.MapGet("/session/predict", async (HttpContext httpContext, string sessionName, string prompt) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    var session = loader.GetSession(sessionName);

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
