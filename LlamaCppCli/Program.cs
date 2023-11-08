using System.Text;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task Main(string[] args)
        {
            // Multibyte encoding handling (e.g. emojis, etc.)
            Console.OutputEncoding = Encoding.UTF8;

            // Native API using raw function calls (standalone)
            await RunSampleRawAsync(args);

            // Library API using wrapped native calls (standalone)
            await RunSampleLibraryAsync(args);

            // Remote API using wrapped client calls (first run `LlamaCppWeb.exe` for the API hosting)
            await RunSampleClientAsync(args);
        }
    }
}
