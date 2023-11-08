using System.Net;
using System.Text;

using LlamaCppLib;

namespace LlamaCppWeb
{
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

            builder.Services.AddSingleton<LlmEngine>(serviceProvider => new(new EngineOptions { MaxParallel = 8 }));
            builder.Services.AddCors();

            var app = builder.Build();

            app.UseCors(configure => configure.AllowAnyOrigin());

            app.MapGet("/", async (httpContext) => await httpContext.Response.WriteAsync("Welcome to LLaMA C++ (dotnet)!"));

            app.MapGet("/load", async (HttpContext httpContext, LlmConfig config, LlmEngine engine, string name) =>
            {
                var path = config.Models.Single(model => model.Name == name).Path ?? throw new FileNotFoundException();
                engine.LoadModel(path, new ModelOptions { Seed = 0, GpuLayers = 64 });
                engine.StartAsync();
                await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
            });

            app.MapGet("/unload", async (HttpContext httpContext, LlmEngine engine) =>
            {
                await engine.StopAsync();
                engine.UnloadModel();
                await httpContext.Response.WriteAsJsonAsync(HttpStatusCode.OK);
            });

            app.MapPost("/prompt", async (HttpContext httpContext, LlmEngine engine) =>
            {
                var lifetime = httpContext.RequestServices.GetRequiredService<IHostApplicationLifetime>();
                using var cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(httpContext.RequestAborted, lifetime.ApplicationStopping);

                var request = await httpContext.Request.ReadFromJsonAsync<LlmPromptRequest>(cancellationTokenSource.Token) ?? new();
                var prompt = engine.Prompt(request.PromptText, request.SamplingOptions);

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

    file class LlmPromptRequest
    {
        public string PromptText { get; set; } = String.Empty;
        public SamplingOptions SamplingOptions { get; set; } = new();
    }
}
