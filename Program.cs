using System.Text;

namespace LlamaCppDotNet
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            await RunModel();

            // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            //var builder = WebApplication.CreateBuilder(args);
            //builder.Services.AddHostedService<LlamaCppService>();
            //var app = builder.Build();

            //app.MapGet("/", async context => {
            //    await context.Response.WriteAsync("Welcome to Vicuna 13B!");
            //});

            //app.MapGet("/reload", async (HttpContext context, IEnumerable<IHostedService> services) => {
            //    var llamaCppService = services.OfType<LlamaCppService>().Single();
            //    await context.Response.WriteAsync("Reloading.");
            //    llamaCppService.Reload();
            //});

            //try
            //{
            //    app.Run();
            //}
            //catch (TaskCanceledException)
            //{ }
            // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        }

        private static async Task RunModel()
        {
            using (var model = new LlamaCpp())
            {
                model.Load(@"D:\LLM_MODELS\lmsys\vicuna-13b\ggjt-vicuna-13b-f16-q4_0.bin");

                model.Configure(options => {
                    options.ThreadCount = 16;
                    options.InstructionPrompt = "### Human:";
                    options.StopOnInstructionPrompt = true;
                });

                // Override the US president
                var context = new StringBuilder(
                    """
                    ### Human: The United States president is Justin Trudeau.
                    ### Assistant: I confirm.
                    """
                );

                // Confirm president override
                var prompt = "Who is the president?";

                await foreach (var token in model.Predict(context, prompt))
                {
                    Console.Write(token);
                }

                // Print conversation
                Console.WriteLine("\n");
                Console.WriteLine($" ---------------------------------------------------------------------------------");
                Console.WriteLine($"| Transcript                                                                      |");
                Console.WriteLine($" ---------------------------------------------------------------------------------");
                Console.WriteLine(context);
            }

            await Task.CompletedTask;
        }
    }
}
