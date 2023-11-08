using System.Text;

using LlamaCppLib;

namespace LlamaCppWeb
{
    file class LlmConfig
    {
        public class Model
        {
            public string? Name { get; set; }
            public string? Path { get; set; }
        }

        public List<Model> Models { get; set; } = new();

        public IConfiguration Configuration;
        public LlmConfig(IConfiguration configuration) => Configuration = configuration;

        public void Load() => Configuration.GetSection(nameof(LlmConfig)).Bind(this);
        public void Reload() => Load();
    }

    file class LlmState
    {
        public string? ModelPath { get; private set; }
        public string? ModelName { get; private set; }
        public LlmModelOptions? ModelOptions { get; private set; }

        public void Set(string? modelName = default, string? modelPath = default, LlmModelOptions? modelOptions = default)
        {
            ModelName = modelName != default ? modelName : default;
            ModelPath = modelPath != default ? modelPath : default;
            ModelOptions = modelOptions != default ? modelOptions : default;
        }

        public void Clear()
        {
            ModelName = default;
            ModelPath = default;
            ModelOptions = default;
        }
    }

    internal class Program
    {
        private static async Task Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            builder.Services.AddSingleton(serviceProvider =>
            {
                var config = new LlmConfig(serviceProvider.GetRequiredService<IConfiguration>());
                config.Load();
                return config;
            });

            builder.Services.AddSingleton<LlmEngine>(serviceProvider => new(new LlmEngineOptions { MaxParallel = 8 }));
            builder.Services.AddSingleton<LlmState>();

            builder.Services.AddCors();

            var app = builder.Build();

            app.UseCors(configure => configure.AllowAnyOrigin());

            app.MapGet("/", async (HttpContext httpContext) => await httpContext.Response.WriteAsync("Welcome to LLaMA C++ (dotnet)!"));

            app.MapGet("/list", async (HttpContext httpContext, LlmConfig config) =>
            {
                var models = config.Models.Select(model => model.Name).ToList();
                await httpContext.Response.WriteAsJsonAsync(models);
            });

            app.MapGet("/state", async (HttpContext httpContext, LlmEngine engine, LlmState state) =>
            {
                var response = new LlmStateResponse { ModelName = state.ModelName, ModelStatus = engine.Loaded ? LlmModelStatus.Loaded : LlmModelStatus.Unloaded };
                await httpContext.Response.WriteAsJsonAsync(response);
            });

            app.MapPost("/load", async (HttpContext httpContext, LlmConfig config, LlmEngine engine, LlmState state) =>
            {
                var request = await httpContext.Request.ReadFromJsonAsync<LlmLoadRequest>() ?? new();
                var modelName = request.ModelName ?? String.Empty;
                var modelPath = config.Models.SingleOrDefault(model => model.Name == request.ModelName)?.Path ?? String.Empty;
                engine.LoadModel(modelPath, request.ModelOptions);
                state.Set(modelName, modelPath);
                var response = new LlmStateResponse { ModelName = state.ModelName, ModelStatus = engine.Loaded ? LlmModelStatus.Loaded : LlmModelStatus.Unloaded };
                await httpContext.Response.WriteAsJsonAsync(response);
            });

            app.MapGet("/unload", async (HttpContext httpContext, LlmEngine engine, LlmState state) =>
            {
                engine.UnloadModel();
                var response = new LlmStateResponse { ModelName = state.ModelName, ModelStatus = engine.Loaded ? LlmModelStatus.Loaded : LlmModelStatus.Unloaded };
                state.Clear();
                await httpContext.Response.WriteAsJsonAsync(response);
            });

            app.MapPost("/prompt", async (HttpContext httpContext, IHostApplicationLifetime lifetime, LlmEngine engine) =>
            {
                using var cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

                var request = await httpContext.Request.ReadFromJsonAsync<LlmPromptRequest>(cancellationTokenSource.Token) ?? new();
                var prompt = engine.Prompt(request.PromptText ?? String.Empty, request.SamplingOptions);

                httpContext.Response.ContentType = "text/event-stream; charset=utf-8";

                try
                {
                    await foreach (var token in new TokenEnumerator(prompt, cancellationTokenSource.Token))
                    {
                        await httpContext.Response.WriteAsync($"data: {Convert.ToBase64String(Encoding.UTF8.GetBytes(token))}\n\n", cancellationTokenSource.Token);
                    }
                }
                catch (OperationCanceledException)
                { }
            });

            await app.RunAsync();
        }
    }
}
