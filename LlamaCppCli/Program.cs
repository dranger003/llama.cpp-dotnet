using LlamaCppLib;

namespace LlamaCppCli
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            args = new[]
            {
                "D:\\LLM_MODELS\\teknium\\ggml-openhermes-2-mistral-7b-q8_0.gguf",
                //"D:\\LLM_MODELS\\codellama\\ggml-codellama-34b-instruct-q4_k.gguf",
            };

            await RunSampleAsync(args);
        }

        static async Task RunSampleAsync(string[] args)
        {
            var modelOptions = new ModelOptions
            {
                GpuLayers = 64,
            };

            using var model = new LlmModel();
            model.Load(args[0], modelOptions, loadProgress => { Console.Write($"\r{new String(' ', 32)}\r{loadProgress:F2}"); });

            Console.WriteLine("Press <any key> to quit.");
            Console.ReadKey(true);

            model.Unload();

            Console.WriteLine("Press <any key> to quit.");
            Console.ReadKey(true);

            model.Load(args[0], modelOptions, loadProgress => { Console.Write($"\r{new String(' ', 32)}\r{loadProgress:F2}"); });

            Console.WriteLine("Press <any key> to quit.");
            Console.ReadKey(true);

            await Task.CompletedTask;
        }
    }
}
