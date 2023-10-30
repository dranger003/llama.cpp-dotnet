using System.Text;
using System.Text.RegularExpressions;

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
            using var llm = new LlmEngine();

            llm.LoadModel(
                args[0],
                new ModelOptions { GpuLayers = 64 },
                loadProgress => Console.Write($"\r{new String(' ', 64)}\rLoading... {loadProgress:F2}%{(loadProgress == 100.0f ? "\n" : "")}")
            );

            var task = Task.Run(
                async () =>
                {
                    await llm.WaitForRunningAsync();
                    Console.WriteLine("\nEngine ready and model loaded.");

                    while (true)
                    {
                        Console.Write("> ");
                        var prompt = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                        if (String.IsNullOrWhiteSpace(prompt))
                            break;

                        var match = Regex.Match(prompt, @"^/load (.+)$");
                        if (match.Success)
                        {
                            var fileName = match.Groups[1].Value;
                            if (File.Exists(fileName))
                            {
                                Console.WriteLine($"File \"{fileName}\" not found.");
                                continue;
                            }

                            prompt = File.ReadAllText(fileName);
                        }

                        var request = llm.NewRequest(prompt, true, true);
                        await foreach (var token in request.Tokens.Reader.ReadAllAsync())
                            Console.Write(Encoding.ASCII.GetString(token));

                        Console.WriteLine();
                    }

                    Console.WriteLine($"Stopping...");
                    await llm.StopAsync();
                }
            );

            await llm.RunAsync();

            await task;
            Console.WriteLine("Done.");
        }
    }
}
