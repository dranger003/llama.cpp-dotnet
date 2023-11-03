using System.Text.RegularExpressions;
using System.Text;

using LlamaCppLib;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunSampleAsync(string[] args)
        {
            using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });
            llm.LoadModel(args[0], new ModelOptions { Seed = 0, GpuLayers = 32 });

            var extraStopTokens = new[] { llm.Tokenize("<|end_of_turn|>", false, true)[0] };

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { cancellationTokenSource.Cancel(); e.Cancel = true; };

            var inputTask = Task.Run(
                async () =>
                {
                    await llm.WaitForRunningAsync();
                    Console.WriteLine("\nEngine ready and model loaded.");

                    while (true)
                    {
                        if (cancellationTokenSource.IsCancellationRequested)
                            cancellationTokenSource = new();

                        Console.Write("> ");
                        var prompt = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                        if (String.IsNullOrWhiteSpace(prompt))
                            break;

                        // Bulk parallel requests without streaming
                        // i.e. `/load "prompt_file1.txt" "prompt_file2.txt" ...`
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
                                .Select(fileName => llm.Prompt(File.ReadAllText(fileName), new SamplingOptions { Temperature = 0.0f }, true, true, extraStopTokens))
                                .Select(
                                    async request =>
                                    {
                                        var response = new StringBuilder();
                                        var assembler = new MultibyteCharAssembler();

                                        await foreach (var token in request.NextToken(cancellationTokenSource.Token))
                                            response.Append(assembler.Consume(token));
                                        response.Append(assembler.Consume());

                                        return (Request: request, Response: response.ToString(), request.Cancelled);
                                    }
                                )
                                .ToList();

                            var results = await Task.WhenAll(requests);

                            Console.WriteLine(new String('=', 196));
                            foreach (var result in results)
                            {
                                Console.WriteLine($"Request {result.Request.GetHashCode()} | Prompting {result.Request.PromptingTime.TotalSeconds} | Sampling {result.Request.SamplingTime.TotalSeconds}");
                                Console.WriteLine(new String('-', 196));
                                Console.WriteLine(result.Request.Prompt);
                                Console.WriteLine(new String('-', 196));
                                Console.WriteLine($"{result.Response}{(result.Cancelled ? " [Cancelled]" : "")}");
                                Console.WriteLine(new String('=', 196));
                            }

                            continue;
                        }

                        // Single request with streaming
                        // i.e. `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello there! How are you today?<|im_end|>\n<|im_start|>assistant\n`
                        var request = llm.Prompt(prompt, new SamplingOptions { Temperature = 0.0f }, true, true, extraStopTokens);
                        var assembler = new MultibyteCharAssembler();

                        await foreach (var token in request.NextToken(cancellationTokenSource.Token))
                            Console.Write(assembler.Consume(token));
                        Console.WriteLine(assembler.Consume());

                        if (request.Cancelled)
                            Console.WriteLine($" [Cancelled]");

                        Console.WriteLine($"Prompting {request.PromptingTime.TotalSeconds} | Sampling {request.SamplingTime.TotalSeconds}");
                    }

                    Console.WriteLine($"Shutting down...");
                }
            );

            llm.StartAsync();
            await inputTask;
            await llm.StopAsync();

            Console.WriteLine("Bye.");
        }
    }
}
