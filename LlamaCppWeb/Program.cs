using System.Net;
using System.Text.Json;
using System.Web;

using LlamaCppLib;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddSingleton<LlamaCppModelManager>();
builder.Services.AddSingleton<LlamaCppSessionManager>();
builder.Services.AddCors();

var app = builder.Build();

app.UseCors(configure => configure.AllowAnyOrigin());

app.MapGet("/", async (HttpContext httpContext) =>
{
    await httpContext.Response.WriteAsync("Welcome to LLaMA C++!");
});

app.MapGet("/model/list", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(new { manager.Models }, new JsonSerializerOptions { WriteIndented = true }));
});

app.MapGet("/model/load", async (HttpContext httpContext, string modelName, LlamaCppModelOptions modelOptions) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    if (manager.Status != LlamaCppModelStatus.Loaded || manager.ModelName != modelName)
    {
        manager.LoadModel(modelName, modelOptions);
    }
    else if (manager.Model != null)
    {
        manager.Model.ResetState();
    }

    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/model/unload", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    manager.UnloadModel();

    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/model/status", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(new { Status = Enum.GetName(manager.Status), manager.Model?.Options }, new JsonSerializerOptions { WriteIndented = true }));
});

app.MapGet("/model/tokenize", async (HttpContext httpContext, string prompt) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    var model = manager.Model;
    if (model == null)
    {
        await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.BadRequest);
        return;
    }

    var tokens = model.Tokenize(prompt);
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(new { TokenCount = tokens.Count, Tokens = tokens }, new JsonSerializerOptions { WriteIndented = true }));
});

app.MapGet("/model/reset", async (HttpContext httpContext) =>
{
    var manager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    var model = manager.Model;
    if (model == null)
    {
        await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.BadRequest);
        return;
    }

    model.ResetState();
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapGet("/session/create", async (HttpContext httpContext) =>
{
    var modelManager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    var model = modelManager.Model;
    if (model == null)
    {
        await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.BadRequest);
        return;
    }

    var sessionManager = httpContext.RequestServices.GetRequiredService<LlamaCppSessionManager>();
    var session = model.CreateSession();
    sessionManager.Sessions.Add(session.Id, session);

    await httpContext.Response.WriteAsync(JsonSerializer.Serialize($"{session.Id}"));
});

app.MapGet("/session/list", async (HttpContext httpContext) =>
{
    var sessionManager = httpContext.RequestServices.GetRequiredService<LlamaCppSessionManager>();
    await httpContext.Response.WriteAsync(JsonSerializer.Serialize(sessionManager.Sessions.Select(session => $"{session.Key}")));
});

app.MapGet("/session/get", async (HttpContext httpContext, Guid sessionId) =>
{
    var sessionManager = httpContext.RequestServices.GetRequiredService<LlamaCppSessionManager>();
    if (!sessionManager.Sessions.TryGetValue(sessionId, out var session))
    {
        await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.BadRequest);
        return;
    }

    await httpContext.Response.WriteAsync(HttpUtility.HtmlEncode(session.GetContextAsText()));
});

app.MapGet("/session/reset", async (HttpContext httpContext, Guid sessionId) =>
{
    var sessionManager = httpContext.RequestServices.GetRequiredService<LlamaCppSessionManager>();
    if (!sessionManager.Sessions.TryGetValue(sessionId, out var session))
    {
        await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.BadRequest);
        return;
    }

    session.Reset();
    await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
});

app.MapPost("/model/generate", async (HttpContext httpContext) =>
{
    var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
    using var cts = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

    var content = await httpContext.Request.ReadFromJsonAsync<RequestBody>(cts.Token) ?? new();

    var modelManager = httpContext.RequestServices.GetRequiredService<LlamaCppModelManager>();
    var model = modelManager.Model;
    if (model == null)
    {
        httpContext.Response.StatusCode = (int)HttpStatusCode.BadRequest;
        await httpContext.Response.WriteAsJsonAsync($"Model not loaded.");
        return;
    }

    var sessionManager = httpContext.RequestServices.GetRequiredService<LlamaCppSessionManager>();
    if (!sessionManager.Sessions.TryGetValue(content.SessionId, out var session))
    {
        httpContext.Response.StatusCode = (int)HttpStatusCode.BadRequest;
        await httpContext.Response.WriteAsJsonAsync($"Session not found.");
        return;
    }

    httpContext.Response.ContentType = "text/event-stream; charset=utf-8";

    try
    {
        await foreach (var tokenString in session.GenerateTokenStringAsync(content.Prompt, content.GenerateOptions, cts.Token))
        {
            var encodedTokenString = tokenString.Replace("\n", "\\n").Replace("\t", "\\t");
            await httpContext.Response.WriteAsync($"data: {encodedTokenString}\n\n", cts.Token);
        }
    }
    catch (OperationCanceledException)
    { }
});

await app.RunAsync();

internal class RequestBody
{
    public Guid SessionId { get; set; }
    public LlamaCppGenerateOptions GenerateOptions { get; set; } = new();
    public string Prompt { get; set; } = string.Empty;
}
