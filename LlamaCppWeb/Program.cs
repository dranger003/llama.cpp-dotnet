using LlamaCppLib;
using LlamaCppWeb;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<LlamaCppLoader>();
builder.Services.AddSingleton<LlamaContext>();

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

app.MapGet("/load", async (HttpContext httpContext, string? modelName) =>
{
    if (modelName == null)
    {
        await httpContext.Response.WriteAsJsonAsync(new { ModelName = String.Empty });
        return;
    }

    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    var state = httpContext.RequestServices.GetRequiredService<LlamaContext>();

    loader.Load(modelName, out var initialContext);

    state.InitialContext = initialContext;
    state.Context.Clear();
    state.Context.Append(initialContext ?? String.Empty);

    await httpContext.Response.WriteAsJsonAsync(new { ModelName = modelName });
});

app.MapGet("/unload", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    var modelName = loader.Model?.ModelName;
    loader.Unload();

    await httpContext.Response.WriteAsJsonAsync(new { ModelName = modelName });
});

app.MapGet("/status", async (HttpContext httpContext) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();

    await httpContext.Response.WriteAsJsonAsync(new { ModelName = loader.Model != null ? loader.Model.ModelName : String.Empty });
});

app.MapGet("/context", async (HttpContext httpContext, string? context, bool? reset) =>
{
    var state = httpContext.RequestServices.GetRequiredService<LlamaContext>();

    if (reset != null && reset.Value)
        state.ResetContext();

    if (context != null)
        state.Context.Append(context);

    await httpContext.Response.WriteAsJsonAsync(new { Context = $"{state.Context}" });
});

app.MapGet("/predict", async (HttpContext httpContext, string? prompt) =>
{
    var loader = httpContext.RequestServices.GetRequiredService<LlamaCppLoader>();
    var model = loader.Model;

    if (model == null)
    {
        await httpContext.Response.WriteAsync("No model loaded.");
        return;
    }

    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    httpContext.Response.ContentType = "text/event-stream";

    var state = httpContext.RequestServices.GetRequiredService<LlamaContext>();

    await foreach (var token in model.Predict(state.Context, prompt ?? String.Empty, true, cts.Token))
    {
        await httpContext.Response.WriteAsync(token);
        await httpContext.Response.Body.FlushAsync();
    }
});

await app.RunAsync();
