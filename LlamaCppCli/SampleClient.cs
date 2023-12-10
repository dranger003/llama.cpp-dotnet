using LlamaCppLib;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunSampleClientAsync(string[] args)
        {
            if (args.Length < 2)
            {
                Console.WriteLine($"Usage: RunSampleClientAsync <Endpoint> <PromptFilePath> [ModelName]");
                return;
            }

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { cancellationTokenSource.Cancel(); e.Cancel = true; };

            // Endpoint (match your hosting server)
            using var client = new LlmClient(args[0]);

            // Model list
            Console.WriteLine($"Model(s):");
            var modelNames = await client.ListAsync();
            modelNames.ForEach(modelName => Console.WriteLine($"    {modelName}"));

            // Model select
            var modelName = args.Length > 2 ? args[2] : modelNames.First();

            // Model state
            Console.WriteLine($"State:");
            var state = await client.StateAsync();
            Console.WriteLine($"    Name: {state.ModelName ?? "<None>"}");
            Console.WriteLine($"    Status: {state.ModelStatus}");

            // Model unload
            if (state.ModelStatus == LlmModelStatus.Loaded && modelName != state.ModelName)
            {
                Console.WriteLine($"Unloading model:");
                state = await client.UnloadAsync();
                Console.WriteLine($"    Name: {state.ModelName}");
                Console.WriteLine($"    Status: {state.ModelStatus}");
            }

            // Model load
            if (state.ModelStatus == LlmModelStatus.Unloaded)
            {
                Console.WriteLine("Loading model:");
                state = await client.LoadAsync(modelName, new LlmModelOptions { Seed = 0, GpuLayers = 48 });
                Console.WriteLine($"    Name: {state.ModelName}");
                Console.WriteLine($"    Status: {state.ModelStatus}");
            }

            // Model prompt
            Console.WriteLine($"Prompting:");
            var promptText = File.ReadAllText(args[1]).Replace("\r\n", "\n");
            var samplingOptions = new SamplingOptions { Temperature = 0.0f, ExtraStopTokens = ["<|EOT|>", "<|end_of_turn|>", "<|endoftext|>", "<|im_end|>"] };

            await foreach (var token in client.PromptAsync(promptText, samplingOptions, cancellationTokenSource.Token))
                Console.Write(token);

            Console.WriteLine($"{(cancellationTokenSource.IsCancellationRequested ? " [Cancelled]" : String.Empty)}");
        }
    }
}
