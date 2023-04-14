using System.Text;
using LlamaCppLib;
using LlamaCppWeb;

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

app.MapGet("/models", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    await httpContext.Response.WriteAsJsonAsync(loader.Models.Select((modelName, i) => new { Id = $"{i}", ModelName = modelName }));
});

app.MapGet("/load", async (HttpContext httpContext, string modelName) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    loader.Load(modelName);

    await httpContext.Response.WriteAsJsonAsync(new { ModelName = modelName });
});

app.MapGet("/unload", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    var modelName = loader.Model.ModelName;
    loader.Unload();

    await httpContext.Response.WriteAsJsonAsync(new { ModelName = modelName });
});

app.MapGet("/status", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    await httpContext.Response.WriteAsJsonAsync(new { ModelName = loader.IsModelLoaded ? loader.Model.ModelName : String.Empty });
});

app.MapGet("/configure", async (HttpContext httpContext, QueryConfigure query) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    loader.Model.Configure(options =>
    {
        options.ThreadCount = query.ThreadCount;
        options.InstructionPrompt = query.InstructionPrompt ?? String.Empty;
        options.StopOnInstructionPrompt = query.StopOnInstructionPrompt;
    });

    await httpContext.Response.WriteAsJsonAsync(new { loader.Model.ModelName });
});

app.MapGet("/predict", async (HttpContext httpContext, QueryPredict query, ILogger<Program> logger) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    httpContext.Response.ContentType = "text/event-stream";

    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    var model = loader.Model;

    var chatContext = new StringBuilder(query.Context);

    await foreach (var token in model.Predict(chatContext, query.Prompt ?? String.Empty, true, cts.Token))
    {
        await httpContext.Response.WriteAsync(token);
        await httpContext.Response.Body.FlushAsync();
    }
});

await app.RunAsync();
