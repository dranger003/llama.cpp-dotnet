using System.Reflection;
using System.Runtime.InteropServices;

internal class Program
{
    delegate string next_input_t();
    delegate void next_output_t(string token);

    [DllImport("main.dll")]
    static extern void set_callbacks(next_input_t ni, next_output_t no);

    [DllImport("main.dll")]
    static extern int main(int argc, string[] argv);

    private static void Main(string[] args)
    {
        //args = new[]
        //{
        //    "--repeat_penalty", "1.17647",
        //    "--ctx_size", "2048",
        //    "--threads", "22",
        //    "--temp", "0.7",
        //    "--top_k", "40",
        //    "--top_p", "0.5",
        //    "--repeat_last_n", "256",
        //    "--batch_size", "1024",
        //    "--n_predict", "2048",
        //    "--interactive",
        //    "--reverse-prompt", "User:",
        //    "--prompt", "Text transcript of a never ending dialog, where User interacts with an AI assistant named ChatLLaMa. ChatLLaMa is helpful, kind, honest, friendly, good at writing and never fails to answer User's requests immediately and with details and precision. There are no annotations like (30 seconds passed...) or (to himself), just what User and ChatLLaMa say aloud to each other. The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long. The transcript only includes text, it does not include markup like HTML and Markdown.",
        //    "--model", @"C:\LLM_MODELS\LLaMA\7B\ggml-model-f16-q4_0.bin"
        //};

        var argc = 1 + args.Length;
        var argv = new[] { Path.GetFileName(Assembly.GetExecutingAssembly().Location) }.Concat(args).ToArray();

        set_callbacks(
            () => Console.ReadLine() ?? String.Empty,
            Console.Write
        );

        _ = main(argc, argv);

        //var builder = WebApplication.CreateBuilder(args);
        //var app = builder.Build();
        //app.MapGet("/", () => "Welcome to LLaMA!");
        //app.Run();
    }
}
