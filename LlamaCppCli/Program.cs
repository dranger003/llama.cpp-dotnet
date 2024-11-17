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

            var samples = new Dictionary<string, Func<string[], Task>>
            {
                // Native API using raw function calls (standalone)
                [nameof(RunSampleRawAsync)] = RunSampleRawAsync,
                // Library API using wrapped native calls (standalone)
                [nameof(RunSampleLibraryAsync)] = RunSampleLibraryAsync,
                // Remote API using wrapped client calls (first run `LlamaCppWeb.exe` for the API hosting)
                [nameof(RunSampleClientAsync)] = RunSampleClientAsync,
                // Dump GGUF meta data
                [nameof(RunDumpMetaAsync)] = RunDumpMetaAsync,
                // State load/save using raw function calls
                [nameof(RunSampleStateRawAsync)] = RunSampleStateRawAsync,
                // Embeddings API using raw function calls (intfloat/e5-mistral-7b-instruct)
                [nameof(RunSampleEmbeddingAsync)] = RunSampleEmbeddingAsync,
            }
                .Select((x, i) => (Index: i + 1, Sample: (Name: x.Key, Func: x.Value)))
                .ToList();

            if (args.Length < 1 || !Int32.TryParse(args[0], out var sampleIndex))
            {
                Console.WriteLine($"Usage: LlamaCppCli <SampleNo> [SampleOpt1] [SampleOpt2] [...]");
                Console.WriteLine($"SampleNo:");
                samples.ForEach(x => Console.WriteLine($"    {x.Index}. {x.Sample.Name}"));
                return;
            }

            if (sampleIndex > 0 && sampleIndex < samples.Count)
            {
                await samples[sampleIndex - 1].Sample.Func(args.Skip(1).ToArray());
            }
            else
            {
                Console.WriteLine($"Invalid sample no. {sampleIndex}.");
            }
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
