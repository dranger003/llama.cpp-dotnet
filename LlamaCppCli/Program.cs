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
            using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });

            llm.LoadModel(
                args[0],
                new ModelOptions { Seed = 0, GpuLayers = 64 },
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

                        var match = Regex.Match(prompt, @"\/load\s+("".*?""(?:\s+|$))+");
                        if (match.Success)
                        {
                            var fileNames = Regex.Matches(prompt, "\"(.*?)\"").Select(x => x.Groups[1].Value).ToList();

                            fileNames
                                .Where(fileName => !File.Exists(fileName))
                                .ToList()
                                .ForEach(fileName => Console.WriteLine($"File \"{fileName}\" not found."));

                            var requests = fileNames
                                .Where(fileName => File.Exists(fileName))
                                .Select(fileName => llm.NewRequest(File.ReadAllText(fileName), new SamplingOptions { Temperature = 0.0f }, true, true))
                                .Select(async request =>
                                    {
                                        var text = new StringBuilder();
                                        await foreach (var token in request.Tokens.Reader.ReadAllAsync())
                                        {
                                            text.Append(Encoding.ASCII.GetString(token));
                                        }
                                        return (Request: request, Response: text.ToString());
                                    }
                                )
                                .ToList();

                            var results = await Task.WhenAll(requests);

                            Console.WriteLine(new String('=', 64));
                            foreach (var result in results)
                            {
                                Console.WriteLine(result.Request.Prompt);
                                Console.WriteLine(new String('-', 64));
                                Console.WriteLine(result.Response);
                                Console.WriteLine(new String('=', 64));
                            }

                            continue;
                        }

                        var request = llm.NewRequest(prompt, new SamplingOptions { Temperature = 0.0f }, true, true);

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
