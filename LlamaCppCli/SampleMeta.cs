using System.Text;

using static LlamaCppLib.Native;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunDumpMetaAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunDumpMetaAsync <ModelPath> [Key]");
                return;
            }

            RunDumpMeta(args);
            await Task.CompletedTask;
        }

        static void RunDumpMeta(string[] args)
        {
            llama_backend_init();
            llama_numa_init(ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED);

            var mparams = llama_model_default_params();
            var model = llama_load_model_from_file(args[0], mparams);
            var buffer = new byte[0x100000]; // 1 MiB

            try
            {
                if (args.Length < 2)
                {
                    for (var i = 0; i < llama_model_meta_count(model); i++)
                    {
                        var length = llama_model_meta_key_by_index(model, i, buffer, (nuint)buffer.Length);
                        var key = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(buffer, 0, length));

                        length = llama_model_meta_val_str(model, Encoding.UTF8.GetBytes(key), buffer, (nuint)buffer.Length);
                        var value = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(buffer, 0, length));

                        Console.WriteLine($"[{key}]=[{value}]");
                    }
                }
                else
                {
                    var key = args[1];

                    var length = llama_model_meta_val_str(model, Encoding.UTF8.GetBytes(key), buffer, (nuint)buffer.Length);
                    var value = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(buffer, 0, length));

                    Console.WriteLine($"[{key}]=[{value}]");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            llama_free_model(model);
            llama_backend_free();
        }
    }
}
