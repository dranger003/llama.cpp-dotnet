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

            ThreadCount = Environment.ProcessorCount / 2, // Assuming hyperthreading
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

        //static string ModelName = "Model X";
        //static string ConversationName = "Conversation X";

        //static string[] Prompts = new[]
        //{
        //    "Describe quantum physics in a very concise and brief sentence.",
        //};

        static async Task Main(string[] args)
        {
#if DEBUG
            args = new[] { "0", @"C:\LLM_MODELS\allenai\ggml-tulu-7b-q4_K_M.bin", "Hello?" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(SimpleSample), SimpleSample),
                //(nameof(RawInterfaceSample), RawInterfaceSample),
                //(nameof(WrappedInterfaceSampleWithoutSession), WrappedInterfaceSampleWithoutSession),
                //(nameof(WrappedInterfaceSampleWithSession), WrappedInterfaceSampleWithSession),
                //(nameof(WrappedInterfaceSampleWithSessionInteractive), WrappedInterfaceSampleWithSessionInteractive),
                //(nameof(GetEmbeddings), GetEmbeddings),
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

        //static string LoadTemplate(string path) => File.ReadAllText(path).Replace("{prompt}", "{0}").Replace("{context}", "{1}");

        static async Task SimpleSample(string[] args)
        {
            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} model_path [gpu_layers] [prompt]");
                return;
            }

            var aparams = new GptParams();
            aparams.model = args[0];
            aparams.prompt = args.Length > 2 ? args[2].Replace("\\n", "\n") : "Hello my name is";
            aparams.n_gpu_layers = args.Length > 1 ? Int32.Parse(args[1]) : 0;

            LlamaCppInterop.llama_init_backend();

            var init = Program.llama_init_from_gpt_params(aparams);

            if (init.Context == IntPtr.Zero)
            {
                await Console.Error.WriteLineAsync($"{nameof(SimpleSample)}: error: unable to load model");
                return;
            }

            var tokens_list = Program.llama_tokenize(init.Context, aparams.prompt, true) ?? new();

            var max_context_size = LlamaCppInterop.llama_n_ctx(init.Context);
            var max_tokens_list_size = max_context_size - 4;

            if (tokens_list.Count > max_tokens_list_size)
            {
                await Console.Error.WriteLineAsync($"{nameof(SimpleSample)}: error: prompt too long ({tokens_list.Count} tokens, max {max_tokens_list_size})");
                return;
            }

            await Console.Error.WriteLineAsync("\n");

            foreach (var id in tokens_list)
            {
                await Console.Out.WriteAsync(llama_token_to_str(init.Context, id));
            }

            await Console.Out.FlushAsync();

            var cancel = false;
            Console.CancelKeyPress += (s, e) => e.Cancel = cancel = true;

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

            if (cancel)
            {
                await Console.Out.WriteLineAsync($" [cancelled]");
            }

            LlamaCppInterop.llama_free(init.Context);
            LlamaCppInterop.llama_free_model(init.Model);
        }

        //static async Task RawInterfaceSample(string[] args)
        //{
        //    Console.WriteLine($"Running sample ({nameof(RawInterfaceSample)})...");

        //    var aparams = new GptParams();
        //    aparams.Parse(args);
        //    {
        //        aparams.seed = Options.Seed ?? 0;
        //        aparams.n_threads = Options.ThreadCount ?? 1;
        //        aparams.n_predict = Options.PredictCount ?? -1;
        //        aparams.n_ctx = Options.ContextSize ?? 512;
        //        aparams.n_batch = 512;
        //        aparams.n_keep = 0;
        //        aparams.n_gpu_layers = Options.GpuLayers ?? (args.Length > 2 ? Int32.Parse(args[2]) : 0);
        //        aparams.main_gpu = 0;
        //        aparams.top_k = Options.TopK ?? 40;
        //        aparams.top_p = Options.TopP ?? 0.95f;
        //        aparams.tfs_z = Options.TfsZ ?? 1.0f;
        //        aparams.typical_p = Options.TypicalP ?? 1.0f;
        //        aparams.temp = Options.Temperature ?? 0.8f;
        //        aparams.repeat_penalty = Options.RepeatPenalty ?? 1.1f;
        //        aparams.repeat_last_n = Options.LastTokenCountPenalty ?? 64;
        //        aparams.frequency_penalty = Options.FrequencyPenalty ?? 0.0f;
        //        aparams.presence_penalty = Options.PresencePenalty ?? 0.0f;
        //        aparams.mirostat = (int)(Options.Mirostat ?? 0);
        //        aparams.mirostat_tau = Options.MirostatTAU ?? 5.0f;
        //        aparams.mirostat_eta = Options.MirostatETA ?? 0.1f;
        //        aparams.memory_f16 = Options.UseHalf ?? true;
        //        aparams.penalize_nl = Options.NewLinePenalty ?? true;
        //        aparams.use_mmap = Options.UseMemoryMapping ?? false;
        //        aparams.use_mlock = Options.UseMemoryLocking ?? false;
        //    }

        //    if (!File.Exists(aparams.model))
        //        throw new FileNotFoundException(aparams.model);

        //    LlamaCppInterop.llama_init_backend();

        //    var ctx = llama_init_from_gpt_params(aparams);

        //    Console.WriteLine(LlamaCppInterop.llama_print_system_info());
        //    PrintParams(aparams);

        //    var prompt = Prompts.First();

        //    if (args.Length > 1)
        //    {
        //        var template = LoadTemplate(args[1]);
        //        template = template.Remove(template.Length - 2);

        //        //prompt = String.Format(template, prompt);
        //        prompt = template;
        //    }

        //    var n_ctx = LlamaCppInterop.llama_n_ctx(ctx);

        //    var tokens = new LlamaToken[n_ctx];
        //    var count = LlamaCppInterop.llama_tokenize(ctx, prompt, tokens, tokens.Length, true);

        //    var embd_inp = new List<LlamaToken>(tokens.Take(count));
        //    var last_n_tokens = new List<LlamaToken>(Enumerable.Repeat(0, n_ctx));
        //    var embd = new List<LlamaToken>();

        //    Console.WriteLine($"embd_input: {embd_inp.Count} token(s).");

        //    if (embd_inp.Count > n_ctx)
        //        throw new InsufficientMemoryException($"Token count exceeds context size ({embd_inp.Count} > {n_ctx}).");

        //    var conversation = new StringBuilder();
        //    var done = false;

        //    Console.CancelKeyPress += (s, e) =>
        //    {
        //        e.Cancel = true;
        //        done = true;
        //    };

        //    var mirostat_mu = 2.0f * aparams.mirostat_tau;
        //    var mirostat_m = 100;

        //    var n_past = 0;
        //    var n_consumed = 0;

        //    while (!done)
        //    {
        //        if (embd.Any())
        //        {
        //            if (n_past + embd.Count > n_ctx)
        //            {
        //                var n_left = n_past - aparams.n_keep;
        //                n_past = Math.Max(1, aparams.n_keep);

        //                embd.InsertRange(0, last_n_tokens.GetRange(n_ctx - n_left / 2 - embd.Count, last_n_tokens.Count - embd.Count));
        //            }

        //            for (var i = 0; i < embd.Count; i += aparams.n_batch)
        //            {
        //                var n_eval = embd.Count - i;
        //                if (n_eval > aparams.n_batch)
        //                {
        //                    n_eval = aparams.n_batch;
        //                }

        //                LlamaCppInterop.llama_eval(ctx, embd.Skip(i).ToArray(), n_eval, n_past, aparams.n_threads);
        //                n_past += n_eval;
        //            }
        //        }

        //        embd.Clear();

        //        if (embd_inp.Count <= n_consumed)
        //        {
        //            var id = default(LlamaToken);
        //            {
        //                var logits = LlamaCppInterop.llama_get_logits(ctx);
        //                var n_vocab = LlamaCppInterop.llama_n_vocab(ctx);

        //                // Apply logit biases
        //                foreach (var logit in aparams.logit_bias)
        //                    logits[logit.Key] += logit.Value;

        //                var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);
        //                for (LlamaToken tokenId = 0; tokenId < n_vocab; tokenId++)
        //                    candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });

        //                var candidates_p = new LlamaCppInterop.llama_token_data_array
        //                {
        //                    data = candidates.ToArray(),
        //                    size = (ulong)candidates.Count,
        //                    sorted = false
        //                };

        //                // Apply penalties
        //                var last_repeat = last_n_tokens
        //                    .TakeLast(Math.Min(Math.Min(last_n_tokens.Count, aparams.repeat_last_n), aparams.n_ctx))
        //                    .ToList();

        //                LlamaCppInterop.llama_sample_repetition_penalty(ctx, candidates_p, last_repeat, aparams.repeat_penalty);
        //                LlamaCppInterop.llama_sample_frequency_and_presence_penalties(ctx, candidates_p, last_repeat, aparams.frequency_penalty, aparams.presence_penalty);

        //                if (!aparams.penalize_nl)
        //                    logits[LlamaCppInterop.llama_token_nl()] = logits[LlamaCppInterop.llama_token_nl()];

        //                if (aparams.temp <= 0)
        //                {
        //                    // Greedy sampling
        //                    id = LlamaCppInterop.llama_sample_token_greedy(ctx, candidates_p);
        //                }
        //                else
        //                {
        //                    if (aparams.mirostat == 1)
        //                    {
        //                        LlamaCppInterop.llama_sample_temperature(ctx, candidates_p, aparams.temp);
        //                        id = LlamaCppInterop.llama_sample_token_mirostat(ctx, candidates_p, aparams.mirostat_tau, aparams.mirostat_eta, mirostat_m, ref mirostat_mu);
        //                    }
        //                    else if (aparams.mirostat == 2)
        //                    {
        //                        LlamaCppInterop.llama_sample_temperature(ctx, candidates_p, aparams.temp);
        //                        id = LlamaCppInterop.llama_sample_token_mirostat_v2(ctx, candidates_p, aparams.mirostat_tau, aparams.mirostat_eta, ref mirostat_mu);
        //                    }
        //                    else
        //                    {
        //                        // Temperature sampling
        //                        LlamaCppInterop.llama_sample_top_k(ctx, candidates_p, aparams.top_k);
        //                        LlamaCppInterop.llama_sample_tail_free(ctx, candidates_p, aparams.tfs_z);
        //                        LlamaCppInterop.llama_sample_typical(ctx, candidates_p, aparams.typical_p);
        //                        LlamaCppInterop.llama_sample_top_p(ctx, candidates_p, aparams.top_p);
        //                        LlamaCppInterop.llama_sample_temperature(ctx, candidates_p, aparams.temp);
        //                        id = LlamaCppInterop.llama_sample_token(ctx, candidates_p);
        //                    }
        //                }

        //                last_n_tokens.RemoveAt(0);
        //                last_n_tokens.Add(id);
        //            }

        //            embd.Add(id);
        //        }
        //        else
        //        {
        //            while (embd_inp.Count > n_consumed)
        //            {
        //                embd.Add(embd_inp[n_consumed]);
        //                last_n_tokens.RemoveAt(0);
        //                last_n_tokens.Add(embd_inp[n_consumed]);
        //                ++n_consumed;

        //                if (embd.Count > aparams.n_batch)
        //                    break;
        //            }
        //        }

        //        foreach (var id in embd)
        //        {
        //            var str = LlamaCppInterop.llama_token_to_str(ctx, id);
        //            conversation.Append(str);

        //            Console.Write(str);

        //            if (id == LlamaCppInterop.llama_token_eos())
        //                done = true;
        //        }
        //    }

        //    LlamaCppInterop.llama_print_timings(ctx);
        //    LlamaCppInterop.llama_free(ctx);

        //    //PrintTranscript(conversation.ToString());

        //    await Task.CompletedTask;
        //}

        //static async Task WrappedInterfaceSampleWithoutSession(string[] args)
        //{
        //    Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithoutSession)})...");

        //    using var cts = new CancellationTokenSource();

        //    Console.CancelKeyPress += (s, e) =>
        //    {
        //        e.Cancel = true;
        //        cts.Cancel();
        //    };

        //    Options.GpuLayers = args.Length > 2 ? Int32.Parse(args[2]) : 0;

        //    using (var model = new LlamaCpp(ModelName, Options))
        //    {
        //        model.Load(args[0]);

        //        var prompt = Prompts.First();
        //        Console.WriteLine(prompt);

        //        if (args.Length > 1)
        //            prompt = String.Format(LoadTemplate(args[1]), prompt);

        //        var promptTokens = model.Tokenize(prompt, true);
        //        var conversation = new StringBuilder(prompt);
        //        var predictOptions = new PredictOptions() { PromptVocabIds = promptTokens };

        //        await foreach (var token in model.Predict(predictOptions, cancellationToken: cts.Token))
        //        {
        //            Console.Write(token.Value);
        //            conversation.Append(token.Value);
        //        }

        //        Console.WriteLine();
        //        PrintTranscript(conversation.ToString());
        //    }
        //}

        //static async Task WrappedInterfaceSampleWithSession(string[] args)
        //{
        //    Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithSession)})...");

        //    using var cts = new CancellationTokenSource();

        //    Console.CancelKeyPress += (s, e) =>
        //    {
        //        e.Cancel = true;
        //        cts.Cancel();
        //    };

        //    Options.GpuLayers = args.Length > 2 ? Int32.Parse(args[2]) : 0;

        //    using (var model = new LlamaCpp(ModelName, Options))
        //    {
        //        model.Load(args[0]);

        //        var session = model.CreateSession(ConversationName);

        //        if (args.Length > 1)
        //            session.Configure(options => options.Template = File.ReadAllText(args[1]));

        //        foreach (var prompt in Prompts)
        //        {
        //            Console.WriteLine(prompt);

        //            await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
        //                Console.Write(token);
        //        }

        //        Console.WriteLine();
        //        PrintTranscript(session.Conversation);
        //    }
        //}

        //static async Task WrappedInterfaceSampleWithSessionInteractive(string[] args)
        //{
        //    Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithSessionInteractive)})...");

        //    var cts = new CancellationTokenSource();

        //    Console.CancelKeyPress += (s, e) =>
        //    {
        //        e.Cancel = true;
        //        cts.Cancel();
        //    };

        //    Options.GpuLayers = args.Length > 2 ? Int32.Parse(args[2]) : 0;

        //    using (var model = new LlamaCpp(ModelName, Options))
        //    {
        //        model.Load(args[0]);

        //        var session = model.CreateSession(ConversationName);

        //        if (args.Length > 1)
        //            session.Configure(options => options.Template = File.ReadAllText(args[1]));

        //        Console.WriteLine($"\nEntering interactive mode.");
        //        Console.WriteLine($"Press <Ctrl+C> to interrupt a response.");
        //        Console.WriteLine($"Press <Enter> on an emptpy prompt to quit.");
        //        Console.WriteLine();

        //        while (true)
        //        {
        //            Console.Write("> ");
        //            var prompt = Console.ReadLine();

        //            if (String.IsNullOrWhiteSpace(prompt))
        //                break;

        //            await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
        //                Console.Write(token);

        //            cts.Dispose();
        //            cts = new();
        //        }

        //        Console.WriteLine();
        //        //PrintTranscript(session.Conversation);
        //    }
        //}

        //// Unused, besides making sure the github sample compiles fine
        //static async Task GithubReadmeSample(string[] args)
        //{
        //    // Configure some model options
        //    var options = new LlamaCppOptions
        //    {
        //        ThreadCount = 4,
        //        TopK = 40,
        //        TopP = 0.95f,
        //        Temperature = 0.8f,
        //        RepeatPenalty = 1.1f,
        //        Mirostat = Mirostat.Mirostat2,
        //        GpuLayers = 20,
        //    };

        //    // Create new named model with options
        //    using var model = new LlamaCpp("WizardVicunaLM", options);

        //    // Load model file
        //    model.Load(@"C:\LLM_MODELS\junelee\ggml-wizard-vicuna-13b-q8_0.bin");

        //    // Create new conversation session and configure prompt template
        //    var session = model.CreateSession(ConversationName);
        //    session.Configure(options => options.Template = File.ReadAllText(@"..\..\template_wizardvicunalm.txt"));

        //    while (true)
        //    {
        //        // Get a prompt
        //        Console.Write("> ");
        //        var prompt = Console.ReadLine();

        //        // Quit on blank prompt
        //        if (String.IsNullOrWhiteSpace(prompt))
        //            break;

        //        // Run the predictions
        //        await foreach (var token in session.Predict(prompt))
        //            Console.Write(token);
        //    }
        //}

        //static async Task GetEmbeddings(string[] args)
        //{
        //    Console.WriteLine($"Running sample ({nameof(GetEmbeddings)})...");

        //    var n_threads = Options.ThreadCount ?? 1;
        //    var n_past = 0;

        //    var cparams = LlamaCppInterop.llama_context_default_params();
        //    cparams.n_ctx = Options.ContextSize ?? 512;
        //    cparams.embedding = true;

        //    var handle = LlamaCppInterop.llama_load_model_from_file(args[0], cparams);
        //    Console.WriteLine(LlamaCppInterop.llama_print_system_info());

        //    var tokens = new LlamaToken[cparams.n_ctx];
        //    LlamaCppInterop.llama_tokenize(handle, Prompts.First(), tokens, tokens.Length, true);
        //    var embd_inp = tokens.ToList();

        //    if (embd_inp.Count > 0)
        //    {
        //        LlamaCppInterop.llama_eval(handle, embd_inp.ToArray(), embd_inp.Count, n_past, n_threads);
        //        var embeddings = LlamaCppInterop.llama_get_embeddings(handle);

        //        Console.WriteLine(
        //            embeddings
        //                .Select(embedding => $"{embedding:F6}")
        //                .Aggregate((a, b) => $"{a}, {b}")
        //        );
        //    }

        //    LlamaCppInterop.llama_print_timings(handle);
        //    LlamaCppInterop.llama_free(handle);

        //    await Task.CompletedTask;
        //}

        //static void PrintTranscript(string conversation)
        //{
        //    Console.WriteLine($" --------------------------------------------------------------------------------------------------");
        //    Console.WriteLine($"| Transcript                                                                                       |");
        //    Console.WriteLine($" --------------------------------------------------------------------------------------------------");
        //    Console.Write(conversation.Trim());
        //}

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
