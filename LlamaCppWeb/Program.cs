using System.Net;
using System.Text;
using System.Text.Json;

using LlamaCppLib;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<LlamaCppManager>();
builder.Services.AddCors();

var app = builder.Build();

app.UseCors(configure => configure.AllowAnyOrigin());

app.MapGet("/", async (HttpContext httpContext) =>
{
    await httpContext.Response.WriteAsync("Welcome to LLaMA C++!");
});

app.MapGet("/model/list", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(new { manager.Models }, new JsonSerializerOptions { WriteIndented = true }));
});

app.MapGet("/model/load", async (HttpContext httpContext, string modelName, LlamaCppModelOptions modelOptions) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    if (manager.Status != LlamaCppModelStatus.Loaded || manager.ModelName != modelName)
        manager.LoadModel(modelName, modelOptions);
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/model/unload", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    manager.UnloadModel();
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/model/status", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(new { Status = Enum.GetName(manager.Status), manager.Model?.Options }, new JsonSerializerOptions { WriteIndented = true }));
});

app.MapGet("/model/tokenize", async (HttpContext httpContext, string prompt) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();
    var model = manager.Model;
    if (model == null)
    {
        await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.BadRequest);
        return;
    }

    var tokens = model.Tokenize(prompt);
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(new { TokenCount = tokens.Count, Tokens = tokens }, new JsonSerializerOptions { WriteIndented = true }));
});

app.MapGet("/model/predict", async (HttpContext httpContext, LlamaCppPredictOptions predictOptions) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppManager>();

    var model = manager.Model;
    if (model == null)
        return;

    httpContext.Response.ContentType = "text/event-stream";

    try
    {
        await foreach (var prediction in model.Predict(predictOptions, cts.Token))
        {
            var token = prediction.Value.Replace("\n", "\\n");
            await httpContext.Response.WriteAsync($"data: {token}\n\n", cts.Token);
        }
    }
    catch (OperationCanceledException)
    { }
});

await app.RunAsync();
