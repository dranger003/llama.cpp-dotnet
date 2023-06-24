using System.Reflection;
using System.Text;

using LlamaCppLib;

namespace LlamaCppCli
{
    using static LlamaCppLib.LlamaCppInterop;

    using LlamaModel = System.IntPtr;
    using LlamaContext = System.IntPtr;
    using LlamaToken = System.Int32;

    internal class Program
    {
        // TODO: Change these to suit your needs (especially GpuLayers)
        // As a reference for GpuLayers, a value of 20-30 seems to work with 8GB VRAM using Vicuna-13B q4_0
        public static LlamaCppOptions Options = new()
        {
            Seed = 0,
            PredictCount = -1,
            ContextSize = 2048,
            LastTokenCountPenalty = 64,
            UseHalf = true,
            NewLinePenalty = false,
            UseMemoryMapping = true,
            UseMemoryLocking = false,
            GpuLayers = null,

            ThreadCount = 8, //Environment.ProcessorCount / 2,
            TopK = 40,
            TopP = 0.95f,
            Temperature = 0.8f,
            RepeatPenalty = 1.1f,

            // Mirostat sampling options
            TfsZ = 1.0f,
            TypicalP = 0.95f,
            FrequencyPenalty = 0.0f,
            PresencePenalty = 0.0f,
            Mirostat = Mirostat.Mirostat2,
            MirostatTAU = 5.0f,     // Target entropy
            MirostatETA = 0.10f,    // Learning rate
            PenalizeNewLine = true,
        };

        static async Task Main(string[] args)
        {
#if DEBUG
            args = new[] { "0", @"C:\LLM_MODELS\allenai\ggml-tulu-7b-q4_K_M.bin", "Hello?" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(SimpleSample), SimpleSample),
            }
                .Select((sample, index) => (sample, index))
                .ToDictionary(k => k.sample.Name, v => (Index: v.index, v.sample.Func));

            var PrintAvailableSamples = () =>
            {
                Console.WriteLine($"SAMPLES:");
                foreach (var sample in samples)
                    Console.WriteLine($"    [{sample.Value.Index}] = {sample.Key}");
            };

            if (args.Length < 2)
            {
                Console.WriteLine($"USAGE:");
                Console.WriteLine($"    {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} <SampleIndex> <ModelPath>");
                PrintAvailableSamples();
                return;
            }

            if (!Path.Exists(args[1]))
            {
                Console.WriteLine($"ERROR: Model not found ({Path.GetFullPath(args[1])}).");
                return;
            }

            var sampleIndex = Int32.Parse(args[0]);
            var sampleName = samples.SingleOrDefault(sample => sample.Value.Index == sampleIndex).Key;

            if (sampleName == default)
            {
                Console.WriteLine($"ERROR: Sample not found ({sampleIndex}).");
                PrintAvailableSamples();
                return;
            }

            await samples[sampleName].Func(args.Skip(1).ToArray());
        }

        static async Task SimpleSample(string[] args)
        {
            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} model_path [gpu_layers] [template_string]");
                return;
            }

            var aparams = new GptParams();
            aparams.model = args[0];
            aparams.n_gpu_layers = args.Length > 1 ? Int32.Parse(args[1]) : 0;
            aparams.n_ctx = Options.ContextSize ?? 2048;

            var template = args.Length > 2 ? args[2].Replace("\\n", "\n") : "USER:\n{0}\n\nASSISTANT:\n";

            LlamaCppInterop.llama_init_backend();

            var init = Program.llama_init_from_gpt_params(aparams);

            if (init.Context == IntPtr.Zero)
            {
                await Console.Error.WriteLineAsync($"{nameof(SimpleSample)}: error: unable to load model");
                return;
            }

            PrintParams(aparams);

            var max_context_size = LlamaCppInterop.llama_n_ctx(init.Context);
            var max_tokens_list_size = max_context_size - 4;

            var cancel = false;
            Console.CancelKeyPress += (s, e) => e.Cancel = cancel = true;

            await Console.Out.WriteLineAsync("\nEntering interactive mode.\nPress <enter> on an empty input to quit.\nPress <Ctrl+C> to cancel token predictions.");

            while (true)
            {
                await Console.Out.WriteAsync($"\nPrompt: ");
                var prompt = await Console.In.ReadLineAsync();

                if (String.IsNullOrWhiteSpace(prompt))
                    break;

                await Console.Out.WriteAsync($"Response: ");

                var tokens_list = Program.llama_tokenize(init.Context, String.Format(template, prompt), true) ?? new();

                while (LlamaCppInterop.llama_get_kv_cache_token_count(init.Context) < max_context_size && !cancel)
                {
                    if (LlamaCppInterop.llama_eval(init.Context, tokens_list.ToArray(), tokens_list.Count, LlamaCppInterop.llama_get_kv_cache_token_count(init.Context), aparams.n_threads) > 0)
                    {
                        await Console.Error.WriteLineAsync($"{nameof(SimpleSample)} : failed to eval");
                        return;
                    }

                    tokens_list.Clear();

                    var logits = LlamaCppInterop.llama_get_logits(init.Context);
                    var n_vocab = LlamaCppInterop.llama_n_vocab(init.Context);

                    var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);

                    for (LlamaToken tokenId = 0; tokenId < n_vocab; tokenId++)
                    {
                        candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });
                    }

                    var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates.ToArray(), size = (ulong)candidates.Count, sorted = false };

                    var new_token_id = LlamaCppInterop.llama_sample_token_greedy(init.Context, candidates_p);

                    if (new_token_id == LlamaCppInterop.llama_token_eos())
                    {
                        await Console.Out.WriteLineAsync($" [end of text]");
                        break;
                    }

                    await Console.Out.WriteAsync($"{LlamaCppInterop.llama_token_to_str(init.Context, new_token_id)}");
                    await Console.Out.FlushAsync();

                    tokens_list.Add(new_token_id);
                }
            }

            if (cancel)
            {
                await Console.Out.WriteLineAsync($" [cancelled]");
            }

            LlamaCppInterop.llama_free(init.Context);
            LlamaCppInterop.llama_free_model(init.Model);
        }

        static void PrintParams(GptParams aparams)
        {
            Console.WriteLine(
                $"params: n_threads = {aparams.n_threads}, memory_f16 = {aparams.memory_f16}\n" +
                $"sampling: " +
                $"repeat_last_n = {aparams.repeat_last_n}, repeat_penalty = {aparams.repeat_penalty}, presence_penalty = {aparams.presence_penalty}, frequency_penalty = {aparams.frequency_penalty}, " +
                $"top_k = {aparams.top_k}, tfs_z = {aparams.tfs_z}, top_p = {aparams.top_p}, typical_p = {aparams.typical_p}, temp = {aparams.temp}, mirostat = {aparams.mirostat}, " +
                $"mirostat_lr = {aparams.mirostat_eta}, mirostat_ent = {aparams.mirostat_tau}"
            );
        }

        static (LlamaModel Model, LlamaContext Context) llama_init_from_gpt_params(GptParams aparams)
        {
            var lparams = LlamaCppInterop.llama_context_default_params();

            lparams.n_ctx = aparams.n_ctx;
            lparams.n_batch = aparams.n_batch;
            lparams.n_gpu_layers = aparams.n_gpu_layers;
            lparams.main_gpu = aparams.main_gpu;
            Array.Copy(aparams.tensor_split, lparams.tensor_split, lparams.tensor_split.Length);
            lparams.low_vram = aparams.low_vram;
            lparams.seed = aparams.seed;
            lparams.f16_kv = aparams.memory_f16;
            lparams.use_mmap = aparams.use_mmap;
            lparams.use_mlock = aparams.use_mlock;
            lparams.logits_all = aparams.perplexity;
            lparams.embedding = aparams.embedding;

            var model = LlamaCppInterop.llama_load_model_from_file(aparams.model, lparams);
            var lctx = LlamaCppInterop.llama_new_context_with_model(model, lparams);

            if (!String.IsNullOrWhiteSpace(aparams.lora_adapter))
                LlamaCppInterop.llama_model_apply_lora_from_file(model, aparams.lora_adapter, aparams.lora_base, aparams.n_threads);

            return (model, lctx);
        }

        static List<LlamaToken> llama_tokenize(LlamaContext ctx, string text, bool add_bos)
        {
            var res = new LlamaToken[text.Length + (add_bos ? 1 : 0)];
            var n = LlamaCppInterop.llama_tokenize(ctx, text, res, res.Length, add_bos);
            return new(res.Take(n));
        }
    }
}
