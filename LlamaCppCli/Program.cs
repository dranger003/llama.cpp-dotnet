using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task Main(string[] args)
        {
            // Multibyte encoding handling (e.g. emojis, etc.)
            Console.OutputEncoding = Encoding.UTF8;

            // If you need to support runtime native library loading,
            // uncomment this line and implement `ResolveLibrary()` below.
            //NativeLibrary.SetDllImportResolver(typeof(LlamaCppLib.Native).Assembly, ResolveLibrary);

            if (args.Length < 1 || !Int32.TryParse(args[0], out var i))
            {
                Console.WriteLine($"Usage: LlamaCppCli <SampleNo> [SampleOpt1] [SampleOpt2] [...]");
                Console.WriteLine($"SampleNo:");
                Console.WriteLine($"    1. {nameof(RunSampleRawAsync)}");
                Console.WriteLine($"    2. {nameof(RunSampleLibraryAsync)}");
                Console.WriteLine($"    3. {nameof(RunSampleClientAsync)}");
                //Console.WriteLine($"    4. {nameof(RunSampleEmbeddingAsync)}"); // TODO: Sync with updated upstream API
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
                //// Embeddings API using raw function calls (intfloat/e5-mistral-7b-instruct)
                //4 => RunSampleEmbeddingAsync(args),

                _ => Console.Out.WriteLineAsync("Invalid sample no.")
            });
        }

        static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            // TODO: Determine which DLL to load here, i.e.:
            //if (cpuOnly) libraryName = "CPU-Only.dll";
            //else if (nvidiaGpu) libraryName = "nVIDIA-CUDA.dll";
            //else if (amdGpu) libraryName = "AMD-ROCm.dll";

            if (NativeLibrary.TryLoad(libraryName, out var handle))
            {
                return handle;
            }

            throw new DllNotFoundException($"Unable to load library: {libraryName}");
        }
    }

    internal static class Extensions
    {
        public static string TruncateWithEllipsis(this String text, float percentWidth = 0.75f)
        {
            var maxWidth = (int)(Console.WindowWidth * percentWidth);
            return text.Length > maxWidth ? String.Concat(text.AsSpan(0, maxWidth - 3), "...") : text;
        }
    }
}
