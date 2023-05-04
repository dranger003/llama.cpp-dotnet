using System.Reflection;
using System.Text;

using LlamaCppLib;

namespace LlamaCppCli
{
    using LlamaToken = System.Int32;

    internal class Program
    {
        // TODO: Change these to suit your needs
        public static int ContextSize = 2048;
        public static int Seed = 0;

        public static LlamaCppOptions Options = new()
        {
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
            Mirostat = 2,           // 0 = Disabled, 1 = Mirostat, 2 = Mirostat 2.0
            MirostatTAU = 5.0f,     // Target entropy
            MirostatETA = 0.10f,    // Learning rate
            PenalizeNewLine = true,
        };

        static string ModelName = "Model X";
        static string ConversationName = "Conversation X";

        static string[] Prompts = new[]
        {
            "Describe quantum physics in a very short and brief sentence.",
        };

        static async Task Main(string[] args)
        {
#if DEBUG
            args = new[] { "0", @"D:\LLM_MODELS\lmsys\ggml-vicuna-13b-v1.1-q5_1.bin", @"..\..\template_vicuna-v1.1.txt" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(RawInterfaceSample), RawInterfaceSample),
                (nameof(WrappedInterfaceSampleWithoutSession), WrappedInterfaceSampleWithoutSession),
                (nameof(WrappedInterfaceSampleWithSession), WrappedInterfaceSampleWithSession),
                (nameof(WrappedInterfaceSampleWithSessionInteractive), WrappedInterfaceSampleWithSessionInteractive),
                (nameof(GetEmbeddings), GetEmbeddings),
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
                Console.WriteLine($"    {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} <SampleIndex> <ModelPath> [TemplatePath]");
                PrintAvailableSamples();
                return;
            }

            if (!Path.Exists(args[1]))
            {
                Console.WriteLine($"ERROR: Model not found ({Path.GetFullPath(args[1])}).");
                return;
            }

            if (args.Length > 2 && !Path.Exists(args[2]))
            {
                Console.WriteLine($"ERROR: Template not found ({Path.GetFullPath(args[2])}).");
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

        static string LoadTemplate(string path) => File.ReadAllText(path).Replace("{prompt}", "{0}").Replace("{context}", "{1}");

        static async Task RawInterfaceSample(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(RawInterfaceSample)})...");

            var aparams = new GptParams();
            aparams.Parse(args);
            {
                aparams.use_mmap = false;
                aparams.use_mlock = false;
                aparams.mirostat = 2;
            }

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = aparams.n_ctx;
            cparams.n_parts = aparams.n_parts;
            cparams.seed = aparams.seed;
            cparams.f16_kv = aparams.memory_f16;
            cparams.logits_all = false;
            cparams.vocab_only = false;
            cparams.use_mmap = aparams.use_mmap;
            cparams.use_mlock = aparams.use_mlock;
            cparams.embedding = aparams.embedding;

            var ctx = LlamaCppInterop.llama_init_from_file(aparams.model, cparams);

            if (aparams.lora_adapter != null)
                LlamaCppInterop.llama_apply_lora_from_file(ctx, aparams.lora_adapter, aparams.lora_base, aparams.n_threads);

            Console.WriteLine(LlamaCppInterop.llama_print_system_info());

            Console.WriteLine(
                $"sampling: " +
                $"repeat_last_n = {aparams.repeat_last_n}, repeat_penalty = {aparams.repeat_penalty}, presence_penalty = {aparams.presence_penalty}, frequency_penalty = {aparams.frequency_penalty}, " +
                $"top_k = {aparams.top_k}, tfs_z = {aparams.tfs_z}, top_p = {aparams.top_p}, typical_p = {aparams.typical_p}, temp = {aparams.temp}, mirostat = {aparams.mirostat}, " +
                $"mirostat_lr = {aparams.mirostat_eta}, mirostat_ent = {aparams.mirostat_tau}"
            );

            var prompt = Prompts.First();
            Console.WriteLine(prompt);

            if (args.Length > 1)
                prompt = String.Format(LoadTemplate(args[1]), prompt);

            var embd_inp = new List<LlamaToken>(LlamaCppInterop.llama_tokenize(ctx, $"{prompt}", true));
            var last_n_tokens = new List<LlamaToken>(Enumerable.Repeat(0, aparams.n_ctx));
            var embd = new List<LlamaToken>();

            var conversation = new StringBuilder(prompt);
            var done = false;

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                done = true;
            };

            var mirostat_mu = 2.0f * aparams.mirostat_tau;
            var mirostat_m = 100;

            var n_past = 0;
            var n_consumed = 0;

            while (!done)
            {
                if (embd.Any())
                {
                    LlamaCppInterop.llama_eval(ctx, embd, n_past, Options.ThreadCount ?? 1);
                    n_past += embd.Count;
                }

                embd.Clear();

                if (embd_inp.Count <= n_consumed)
                {
                    var id = default(LlamaToken);
                    {
                        var logits = LlamaCppInterop.llama_get_logits(ctx);
                        var n_vocab = LlamaCppInterop.llama_n_vocab(ctx);

                        // Apply logit biases
                        foreach (var logit in aparams.logit_bias)
                            logits[logit.Key] += logit.Value;

                        var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);
                        for (LlamaToken tokenId = 0; tokenId < n_vocab; tokenId++)
                            candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });

                        var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates, sorted = false };

                        // Apply penalties
                        var last_repeat = last_n_tokens
                            .TakeLast(Math.Min(Math.Min(last_n_tokens.Count, aparams.repeat_last_n), aparams.n_ctx))
                            .ToList();

                        LlamaCppInterop.llama_sample_repetition_penalty(ctx, ref candidates_p, last_repeat, aparams.repeat_penalty);
                        LlamaCppInterop.llama_sample_frequency_and_presence_penalties(ctx, ref candidates_p, last_repeat, aparams.frequency_penalty, aparams.presence_penalty);

                        if (!aparams.penalize_nl)
                            logits[LlamaCppInterop.llama_token_nl()] = logits[LlamaCppInterop.llama_token_nl()];

                        if (aparams.temp <= 0)
                        {
                            // Greedy sampling
                            id = LlamaCppInterop.llama_sample_token_greedy(ctx, ref candidates_p);
                        }
                        else
                        {
                            if (aparams.mirostat == 1)
                            {
                                LlamaCppInterop.llama_sample_temperature(ctx, ref candidates_p, aparams.temp);
                                id = LlamaCppInterop.llama_sample_token_mirostat(ctx, ref candidates_p, aparams.mirostat_tau, aparams.mirostat_eta, mirostat_m, ref mirostat_mu);
                            }
                            else if (aparams.mirostat == 2)
                            {
                                LlamaCppInterop.llama_sample_temperature(ctx, ref candidates_p, aparams.temp);
                                id = LlamaCppInterop.llama_sample_token_mirostat_v2(ctx, ref candidates_p, aparams.mirostat_tau, aparams.mirostat_eta, ref mirostat_mu);
                            }
                            else
                            {
                                // Temperature sampling
                                LlamaCppInterop.llama_sample_top_k(ctx, ref candidates_p, aparams.top_k);
                                LlamaCppInterop.llama_sample_tail_free(ctx, ref candidates_p, aparams.tfs_z);
                                LlamaCppInterop.llama_sample_typical(ctx, ref candidates_p, aparams.typical_p);
                                LlamaCppInterop.llama_sample_top_p(ctx, ref candidates_p, aparams.top_p);
                                LlamaCppInterop.llama_sample_temperature(ctx, ref candidates_p, aparams.temp);
                                id = LlamaCppInterop.llama_sample_token(ctx, ref candidates_p);
                            }
                        }

                        last_n_tokens.RemoveAt(0);
                        last_n_tokens.Add(id);
                    }

                    embd.Add(id);
                }
                else
                {
                    while (embd_inp.Count > n_consumed)
                    {
                        embd.Add(embd_inp[n_consumed]);
                        last_n_tokens.RemoveAt(0);
                        last_n_tokens.Add(embd_inp[n_consumed]);
                        ++n_consumed;
                    }
                }

                foreach (var id in embd)
                {
                    var str = LlamaCppInterop.llama_token_to_str(ctx, id);
                    conversation.Append(str);

                    Console.Write(str);

                    if (id == LlamaCppInterop.llama_token_eos())
                        done = true;
                }
            }

            LlamaCppInterop.llama_print_timings(ctx);
            LlamaCppInterop.llama_free(ctx);

            PrintTranscript(conversation.ToString());

            await Task.CompletedTask;
        }

        static async Task WrappedInterfaceSampleWithoutSession(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithoutSession)})...");

            using var cts = new CancellationTokenSource();

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(args[0], ContextSize, Seed);
                model.Configure(Options);

                var prompt = Prompts.First();
                Console.WriteLine(prompt);

                if (args.Length > 1)
                    prompt = String.Format(LoadTemplate(args[1]), prompt);

                var promptTokens = model.Tokenize(prompt, true);
                var conversation = new StringBuilder(prompt);
                var predictOptions = new PredictOptions() { PromptVocabIds = promptTokens };

                await foreach (var token in model.Predict(predictOptions, cancellationToken: cts.Token))
                {
                    Console.Write(token.Value);
                    conversation.Append(token.Value);
                }

                Console.WriteLine();
                PrintTranscript(conversation.ToString());
            }
        }

        static async Task WrappedInterfaceSampleWithSession(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithSession)})...");

            using var cts = new CancellationTokenSource();

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(args[0]);
                model.Configure(Options);

                var session = model.CreateSession(ConversationName);

                if (args.Length > 1)
                    session.Configure(options => options.Template = File.ReadAllText(args[1]));

                foreach (var prompt in Prompts)
                {
                    Console.WriteLine(prompt);

                    await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
                        Console.Write(token);
                }

                Console.WriteLine();
                PrintTranscript(session.Conversation);
            }
        }

        static async Task WrappedInterfaceSampleWithSessionInteractive(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithSessionInteractive)})...");

            var cts = new CancellationTokenSource();

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cts.Cancel();
            };

            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(args[0]);
                model.Configure(Options);

                var session = model.CreateSession(ConversationName);

                if (args.Length > 1)
                    session.Configure(options => options.Template = File.ReadAllText(args[1]));

                Console.WriteLine($"Entering interactive mode.");
                Console.WriteLine($"Press <Ctrl+C> to interrupt a response.");
                Console.WriteLine($"Press <Enter> on an emptpy prompt to quit.");
                Console.WriteLine();

                while (true)
                {
                    Console.Write("> ");
                    var prompt = Console.ReadLine();

                    if (String.IsNullOrWhiteSpace(prompt))
                        break;

                    await foreach (var token in session.Predict(prompt, cancellationToken: cts.Token))
                        Console.Write(token);

                    cts.Dispose();
                    cts = new();
                }

                Console.WriteLine();
                PrintTranscript(session.Conversation);
            }
        }

        static async Task GetEmbeddings(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(GetEmbeddings)})...");

            var n_threads = Options.ThreadCount ?? 0;
            var n_past = 0;

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = ContextSize;
            cparams.embedding = true;

            var handle = LlamaCppInterop.llama_init_from_file(args[0], cparams);
            Console.WriteLine(LlamaCppInterop.llama_print_system_info());

            var embd_inp = LlamaCppInterop.llama_tokenize(handle, Prompts.First(), true);
            if (embd_inp.Count > 0)
            {
                LlamaCppInterop.llama_eval(handle, embd_inp, n_past, n_threads);
                var embeddings = LlamaCppInterop.llama_get_embeddings(handle);

                Console.WriteLine(
                    embeddings
                        .Select(embedding => $"{embedding:F6}")
                        .Aggregate((a, b) => $"{a}, {b}")
                );
            }

            LlamaCppInterop.llama_print_timings(handle);
            LlamaCppInterop.llama_free(handle);

            await Task.CompletedTask;
        }

        static void PrintTranscript(string conversation)
        {
            Console.WriteLine($" --------------------------------------------------------------------------------------------------");
            Console.WriteLine($"| Transcript                                                                                       |");
            Console.WriteLine($" --------------------------------------------------------------------------------------------------");
            Console.Write(conversation.Trim());
        }
    }
}
