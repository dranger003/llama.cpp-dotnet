using System.Text;
using System.Text.RegularExpressions;

using LlamaCppLib;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunSampleClientAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunSampleClientAsync <Endpoint> [<GpuLayers>] [<CtxLength>] [<RopeFreq>]");
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
                .ForEach(model => Console.WriteLine($"    {model.Index}) {model.Name} {(state.ModelName == model.Name && state.ModelStatus == LlmModelStatus.Loaded ? "(loaded)" : String.Empty)}"));
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
            var unload = state.ModelStatus == LlmModelStatus.Loaded && state.ModelName != modelNames[index];

            if (state.ModelStatus == LlmModelStatus.Loaded && state.ModelName == modelNames[index])
            {
                Console.Write("Model already loaded, reload [y/N]? ");
                var key = Console.ReadKey();
                Console.WriteLine();

                if (key.Key == ConsoleKey.Y)
                    unload = true;
            }

            if (unload)
            {
                Console.Write("Unloading model...");
                await client.UnloadAsync();
                Console.WriteLine();
                state = await client.StateAsync();
            }

            var gpuLayers = args.Length > 1 ? Int32.Parse(args[1]) : 0;
            var ctxLength = args.Length > 2 ? Int32.Parse(args[2]) : 0;
            var ropeFreq = args.Length > 3 ? Int32.Parse(args[3]) : 0.0f;

            if (state.ModelStatus == LlmModelStatus.Unloaded)
            {
                Console.Write("Loading model...");
                state = await client.LoadAsync(modelNames[index], new LlmModelOptions { GpuLayers = gpuLayers, ContextLength = ctxLength, RopeFrequeceBase = ropeFreq, UseFlashAttention = true });
                Console.WriteLine();
            }

            Console.WriteLine($"Model name: {state.ModelName}");
            Console.WriteLine($"Model status: {state.ModelStatus}");

            Console.WriteLine();
            Console.WriteLine($"Input prompt below (or to load a prompt from file, i.e. '/load \"prompt.txt\"').");
            Console.WriteLine($"You can also type '/clear' to erase chat history.");
            Console.WriteLine($"To quit, leave input blank and press <Enter>.");

            var messages = new List<LlmMessage> { new() { Role = "system", Content = "You are a helpful assistant." } };

            while (true)
            {
                Console.Write("\n> ");
                var prompt = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                if (String.IsNullOrWhiteSpace(prompt))
                    break;

                if (prompt == "/clear")
                {
                    messages = new(messages.Take(1));
                    continue;
                }

                if (prompt == "/messages")
                {
                    foreach (var message in messages)
                    {
                        Console.WriteLine($"[{message.Role}][{message.Content}]");
                    }

                    continue;
                }

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

                messages.Add(new() { Role = "user", Content = prompt });
                var response = new StringBuilder();

                //var samplingOptions = new SamplingOptions { Temperature = 0.3f, ExtraStopTokens = ["<|EOT|>", "<|end_of_turn|>", "<|endoftext|>", "<|im_end|>"] };
                var samplingOptions = new SamplingOptions() /*{ TopK = 50, TopP = 0.95f, Temperature = 0.7f }*/;

                await foreach (var token in client.PromptAsync(messages, samplingOptions, cancellationTokenSource.Token))
                {
                    Console.Write(token);
                    response.Append(token);
                }

                if (cancellationTokenSource.IsCancellationRequested)
                {
                    messages.Remove(messages.Last());

                    Console.WriteLine(" [Cancelled]");
                    cancellationTokenSource = new();
                }
                else
                {
                    messages.Add(new() { Role = "assistant", Content = response.ToString() });
                }
            }
        }
    }
}
