using LlamaCppLib;
using System.Text.RegularExpressions;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunSampleClientAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunSampleClientAsync <Endpoint>");
                return;
            }

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { cancellationTokenSource.Cancel(); e.Cancel = true; };

            using var client = new LlmClient(args[0]);

            Console.WriteLine($"Available model(s):");
            var modelNames = await client.ListAsync();
            var state = await client.StateAsync();
            modelNames
                .Select((x, i) => (Name: x, Index: i))
                .ToList()
                .ForEach(model => Console.WriteLine($"    {model.Index}) {model.Name} {(state.ModelName == model.Name && state.ModelStatus == LlmModelStatus.Loaded ? "(loaded)" : "(unloaded)")}"));
            Console.WriteLine();

            var GetSelection = () =>
            {
                while (true)
                {
                    Console.Write($"Select model # to load: ");
                    var key = Console.ReadKey();
                    Console.WriteLine();

                    if (Int32.TryParse($"{key.KeyChar}", out var index) && index >= 0 && index < modelNames.Count)
                        return index;

                    Console.WriteLine();
                }
            };

            var index = GetSelection();

            if (state.ModelStatus == LlmModelStatus.Loaded && state.ModelName != modelNames[index])
            {
                Console.Write("Unloading model...");
                await client.UnloadAsync();
                Console.WriteLine();
            }

            if (state.ModelStatus == LlmModelStatus.Unloaded)
            {
                Console.Write("Loading model...");
                state = await client.LoadAsync(modelNames[index], new LlmModelOptions { Seed = -1, GpuLayers = 128, ContextLength = 0 });
                Console.WriteLine();
            }

            Console.WriteLine($"Model name: {state.ModelName}");
            Console.WriteLine($"Model status: {state.ModelStatus}");

            Console.WriteLine($"\nInput prompt below, including the template (or to load a prompt from a file, type '/load \"your_prompt_file.txt\"' without the single quotes).");
            Console.WriteLine($"To quit, leave a blank input and press <Enter>.");

            while (true)
            {
                Console.Write("\n> ");
                var prompt = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                if (String.IsNullOrWhiteSpace(prompt))
                    break;

                var match = Regex.Match(prompt, @"/load\s+""?([^""\s]+)""?");
                if (match.Success)
                {
                    if (File.Exists(match.Groups[1].Value))
                    {
                        Console.WriteLine($"Loading prompt from file \"{Path.GetFullPath(match.Groups[1].Value)}\"...");
                        prompt = File.ReadAllText(match.Groups[1].Value);
                    }
                    else
                    {
                        Console.WriteLine($"File not found \"{match.Groups[1].Value}\".");
                        continue;
                    }
                }

                //var samplingOptions = new SamplingOptions { Temperature = 0.3f, ExtraStopTokens = ["<|EOT|>", "<|end_of_turn|>", "<|endoftext|>", "<|im_end|>"] };
                var samplingOptions = new SamplingOptions { TopK = 50, TopP = 0.95f, Temperature = 0.7f };
                await foreach (var token in client.PromptAsync(prompt, samplingOptions, cancellationTokenSource.Token))
                    Console.Write(token);

                if (cancellationTokenSource.IsCancellationRequested)
                {
                    Console.WriteLine(" [Cancelled]");
                    cancellationTokenSource = new();
                }
            }
        }
    }
}
