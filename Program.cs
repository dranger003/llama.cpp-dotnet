namespace LlamaCppDotNet
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            if (args.Any(x => x == "--dev"))
            {
                args = new[]
                {
                    "--repeat_penalty", "1.17647",
                    "--ctx_size", "2048",
                    "--threads", "22",
                    "--temp", "0.7",
                    "--top_k", "40",
                    "--top_p", "0.5",
                    "--repeat_last_n", "256",
                    "--batch_size", "1024",
                    "--n_predict", "2048",
                    "--interactive",
                    "--reverse-prompt", "User:",
                    "--prompt", "Text transcript of a never ending dialog, where User interacts with an AI assistant named ChatLLaMa. ChatLLaMa is helpful, kind, honest, friendly, good at writing and never fails to answer User's requests immediately and with details and precision. There are no annotations like (30 seconds passed...) or (to himself), just what User and ChatLLaMa say aloud to each other. The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long. The transcript only includes text, it does not include markup like HTML and Markdown.",
                    "--model", @"C:\LLM_MODELS\LLaMA\7B\ggml-model-f16-q4_0.bin"
                };
            }

            var model = new LlamaCpp(args);
            model.Run();

            //var builder = WebApplication.CreateBuilder(args);
            //var app = builder.Build();
            //app.MapGet("/", () => "Welcome to LLaMA!");
            //app.Run();
        }
    }
}
