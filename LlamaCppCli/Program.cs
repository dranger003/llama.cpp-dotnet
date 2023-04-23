using LlamaCppLib;
using System.Reflection;
using System.Text;

namespace LlamaCppCli
{
    using LlamaToken = System.Int32;

    internal class Program
    {
        // TODO: Change these to suit your needs
        static int ContextSize = 2048;
        static int Seed = 0;

        static LlamaCppOptions Options = new()
        {
            ThreadCount = Environment.ProcessorCount / 2, // Assuming hyperthreading
            TopK = 50,
            TopP = 0.95f,
            Temperature = 0.1f,
            RepeatPenalty = 1.1f,
        };

        static string ModelName = "Model X";
        static string ConversationName = "Conversation X";

        static string[] Prompts = new[]
        {
            "In a very short sentence, list the main political schools of thought.",
            "Describe quantum physics in a very short sentence.",
        };

        // TODO: Adjust templates according to your model, this is crucial for reliable predictions (refer to your model description for template structure)
        static Dictionary<string, string> Templates = new Dictionary<string, string>
        {
            ["Vicuna v1.1"] = """
                USER:
                {0}

                ASSISTANT:

                """,
            ["Alpaca (no input)"] = """
                ### Instruction:
                {0}

                ### Response:
                
                """,
            ["Alpaca (input)"] = """
                ### Instruction:
                {0}

                ### Input:
                {1}

                ### Response:
                
                """,
        };

        static async Task Main(string[] args)
        {
            var samples = new Dictionary<string, Func<string[], Task>>
            {
                [nameof(RawInterfaceSample)] = RawInterfaceSample,
                [nameof(WrappedInterfaceSampleWithoutSession)] = WrappedInterfaceSampleWithoutSession,
                [nameof(WrappedInterfaceSampleWithSession)] = WrappedInterfaceSampleWithSession,
                [nameof(WrappedInterfaceSampleWithSessionInteractive)] = WrappedInterfaceSampleWithSessionInteractive,
                [nameof(GetEmbeddings)] = GetEmbeddings,
            };

            var PrintAvailableSamples = () =>
            {
                Console.WriteLine($"SAMPLES:");
                foreach (var sample in samples)
                    Console.WriteLine($"    {sample.Key}");
            };

            if (args.Length != 2)
            {
                Console.WriteLine($"USAGE:");
                Console.WriteLine($"    {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} <Sample> <ModelPath>");
                PrintAvailableSamples();
                return;
            }

            var sampleName = args[0];
            var modelPath = args[1];

            if (!Path.Exists(modelPath))
            {
                Console.WriteLine($"ERROR: Model not found ({modelPath}).");
                return;
            }

            if (!samples.Any(sample => sample.Key == sampleName))
            {
                Console.WriteLine($"ERROR: Sample not found ({sampleName}).");
                PrintAvailableSamples();
            }

            await samples[sampleName](args);
        }

        static async Task RawInterfaceSample(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(RawInterfaceSample)})...");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = ContextSize;
            cparams.seed = Seed;

            var model = LlamaCppInterop.llama_init_from_file(args[1], cparams);
            Console.WriteLine(LlamaCppInterop.llama_print_system_info());

            var prompt = Prompts.First();
            Console.WriteLine(prompt);

            prompt = String.Format(Templates["Vicuna v1.1"], prompt);

            var tokens = LlamaCppInterop.llama_tokenize(model, prompt, true);

            var sampled = new List<LlamaToken>(tokens);
            var context = new List<LlamaToken>();

            var conversation = new StringBuilder(prompt);

            while (true)
            {
                // TODO: Context management (i.e. rotating context buffer, maybe keeping some parts, etc.)

                LlamaCppInterop.llama_eval(model, sampled, context.Count, 16);
                context.AddRange(sampled);

                var id = LlamaCppInterop.llama_sample_top_p_top_k(model, context, Options.TopK ?? 0, Options.TopP ?? 0, Options.Temperature ?? 0, Options.RepeatPenalty ?? 0);
                sampled.ClearAdd(id);

                var str = LlamaCppInterop.llama_token_to_str(model, id);
                conversation.Append(str);

                Console.Write(str);

                if (id == LlamaCppInterop.llama_token_eos())
                    break;
            }

            LlamaCppInterop.llama_print_timings(model);
            LlamaCppInterop.llama_free(model);

            PrintTranscript(conversation.ToString());

            await Task.CompletedTask;
        }

        static async Task WrappedInterfaceSampleWithoutSession(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithoutSession)})...");

            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(args[1], ContextSize, Seed);
                model.Configure(Options);

                var prompt = Prompts.First();
                Console.WriteLine(prompt);

                prompt = String.Format(Templates["Vicuna v1.1"], prompt);

                var promptTokens = model.Tokenize(prompt, true);

                var conversation = new StringBuilder(prompt);

                var contextTokens = new List<LlamaToken>();
                await foreach (var token in model.Predict(contextTokens, promptTokens))
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

            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(args[1]);
                model.Configure(Options);

                var session = model.CreateSession(ConversationName);

                foreach (var prompt in Prompts)
                {
                    Console.WriteLine(prompt);

                    var templatizedPrompt = String.Format(Templates["Vicuna v1.1"], prompt);

                    await foreach (var token in session.Predict(templatizedPrompt))
                        Console.Write(token);
                }

                PrintTranscript(session.Conversation);
            }
        }

        static async Task WrappedInterfaceSampleWithSessionInteractive(string[] args)
        {
            Console.WriteLine($"Running sample ({nameof(WrappedInterfaceSampleWithSessionInteractive)})...");

            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(args[1]);
                model.Configure(Options);

                var session = model.CreateSession(ConversationName);

                while (true)
                {
                    Console.Write("> ");
                    var prompt = Console.ReadLine();

                    if (String.IsNullOrWhiteSpace(prompt))
                        break;

                    prompt = String.Format(Templates["Vicuna v1.1"], prompt);

                    await foreach (var token in session.Predict(prompt))
                        Console.Write(token);
                }
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

            var handle = LlamaCppInterop.llama_init_from_file(args[1], cparams);
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
