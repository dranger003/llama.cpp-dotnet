using System.Text.RegularExpressions;
using System.Text;

using LlamaCppLib;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunSampleLibraryAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunSampleLibraryAsync <ModelPath>");
                return;
            }

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { cancellationTokenSource.Cancel(); e.Cancel = true; };

            using var llm = new LlmEngine(new LlmEngineOptions { MaxParallel = 8 });
            llm.LoadModel(args[0], new LlmModelOptions { Seed = 0, GpuLayers = 64 });

            Console.WriteLine("Press <Ctrl+C> to cancel/quit.");

            while (true)
            {
                if (cancellationTokenSource.IsCancellationRequested)
                    cancellationTokenSource = new();

                Console.Write("> ");
                var promptText = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                if (String.IsNullOrWhiteSpace(promptText))
                    break;

                {
                    // Parallel prompts w/o streaming - e.g.
                    // `/load "prompt_file-1.txt" "prompt_file-2.txt" ...`
                    var match = Regex.Match(promptText, @"\/load\s+("".*?""(?:\s+|$))+");
                    if (match.Success)
                    {
                        var fileNames = Regex.Matches(promptText, "\"(.*?)\"").Select(x => x.Groups[1].Value).ToList();

                        fileNames
                            .Where(fileName => !File.Exists(fileName))
                            .ToList()
                            .ForEach(fileName => Console.WriteLine($"File \"{fileName}\" not found."));

                        var promptTasks = fileNames
                            .Where(fileName => File.Exists(fileName))
                            .Select(fileName => llm.Prompt(File.ReadAllText(fileName), new SamplingOptions { Temperature = 0.0f }))
                            .Select(
                                async prompt =>
                                {
                                    var response = new List<byte>();

                                    // In non-streaming mode, we can collect tokens as raw byte arrays and assemble the response at the end
                                    await foreach (var token in prompt.NextToken(cancellationTokenSource.Token))
                                        response.AddRange(token);

                                    return (Request: prompt, Response: Encoding.UTF8.GetString(response.ToArray()));
                                }
                            )
                            .ToList();

                        while (promptTasks.Any())
                        {
                            var task = await Task.WhenAny(promptTasks);

                            Console.WriteLine(new String('=', 196));
                            Console.WriteLine($"Request {task.Result.Request.GetHashCode()} | Prompting {task.Result.Request.PromptingSpeed:F2} t/s | Sampling {task.Result.Request.SamplingSpeed:F2} t/s");
                            //Console.WriteLine(new String('-', 196));
                            //Console.WriteLine(result.Request.PromptText);
                            Console.WriteLine(new String('-', 196));
                            Console.WriteLine($"{task.Result.Response}{(task.Result.Request.Cancelled ? " [Cancelled]" : "")}");
                            Console.WriteLine(new String('=', 196));

                            promptTasks.Remove(task);
                        }

                        continue;
                    }
                }
                {
                    // Single prompt w/streaming - e.g.
                    // `<|im_start|>system\nYou are an astrophysicist.<|im_end|>\n<|im_start|>user\nDescribe the solar system.<|im_end|>\n<|im_start|>assistant\n`
                    // `[INST] <<SYS>>\nYou are an astrophysicist.\n<</SYS>>\n\nDescribe the solar system. [/INST]\n`
                    var prompt = llm.Prompt(promptText, new SamplingOptions { Temperature = 0.0f });

                    // In streaming mode, we must re-assemble multibyte characters using a TokenEnumerator
                    await foreach (var token in new TokenEnumerator(prompt, cancellationTokenSource.Token))
                        Console.Write(token);

                    Console.WriteLine($"{(prompt.Cancelled ? " [Cancelled]" : "")}");
                    Console.WriteLine($"Prompting {prompt.PromptingSpeed:F2} t/s | Sampling {prompt.SamplingSpeed:F2} t/s");
                }
            }

            Console.WriteLine("Bye.");
        }
    }
}
