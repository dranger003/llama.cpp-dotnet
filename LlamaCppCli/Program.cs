using System.Text;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task Main(string[] args)
        {
            // Multibyte encoding handling (e.g. emojis, etc.)
            Console.OutputEncoding = Encoding.UTF8;

            if (args.Length < 1 || !Int32.TryParse(args[0], out var i))
            {
                Console.WriteLine($"Usage: LlamaCppCli <SampleNo> [SampleOpt1] [SampleOpt2] [...]");
                Console.WriteLine($"SampleNo:");
                Console.WriteLine($"    1. {nameof(RunSampleRawAsync)} <ModelPath> <PromptFilePath> [CtxLength] [GpuLayers]");
                Console.WriteLine($"    2. {nameof(RunSampleLibraryAsync)} <ModelPath>");
                Console.WriteLine($"    3. {nameof(RunSampleClientAsync)} <Endpoint> <PromptFilePath> [ModelName]");
                return;
            }

            args = args.Skip(1).ToArray();

            await (i switch
            {
                // Native API using raw function calls (standalone)
                1 => RunSampleRawAsync(args),
                // Library API using wrapped native calls (standalone)
                2 => RunSampleLibraryAsync(args),
                // Remote API using wrapped client calls (first run `LlamaCppWeb.exe` for the API hosting)
                3 => RunSampleClientAsync(args),

                _ => Console.Out.WriteLineAsync("Invalid sample no.")
            });
        }
    }
}
