using LlamaCppLib;
using System.Text;

namespace LlamaCppCli
{
    using LlamaToken = System.Int32;

    internal class Program
    {
        static string ModelPath = @"D:\LLM_MODELS\lmsys\vicuna-13b-v1.1\ggml-vicuna-13b-v1.1-q4_1.bin";
        static string ModelName = "vicuna-13b-v1.1";

        static string[] Context
        {
            get => new[]
            {
                $"Hi! How can I be of service today?",
                $"Hello! How are you doing?",
                $"I am doing great! Thanks for asking.",
                $"Can you help me with some questions please?",
                $"Absolutely, what questions can I help you with?",
                $"How many planets are there in the solar system?",
            };
        }

        static string[] Prompts
        {
            get => new[]
            {
                $"Can you list the planets of our solar system?",
                $"What do you think Vicuna 13B is according to you?",
                $"Vicuna 13B is a large language model (LLM).",
            };
        }

        static int Seed { get => 0; }
        static int ContextSize { get => 2048; }
        static int ThreadCount { get => 16; }
        static int TopK { get => 50; }
        static float TopP { get => 0.95f; }
        static float Temperature { get => 0.05f; }
        static float RepeatPenalty { get => 1.1f; }

        static async Task Main()
        {
            //await RawInterfaceSample();
            await WrappedInterfaceSampleWithoutSession();
            //await WrappedInterfaceSampleWithSession();
            //await GetEmbeddingsAsync();
        }

        static async Task RawInterfaceSample()
        {
            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = ContextSize;
            cparams.seed = Seed;

            var model = LlamaCppInterop.llama_init_from_file(ModelPath, cparams);
            Console.WriteLine(LlamaCppInterop.llama_print_system_info());

            var prompt = Context.Select((x, i) => $"{(i % 2 == 0 ? "ASSISTANT" : "USER")}:\n{x}\n").Aggregate((a, b) => $"{a}\n{b}");
            prompt = $"{prompt}\nASSISTANT:\n";

            var tokens = LlamaCppInterop.llama_tokenize(model, prompt, true);

            var sampled = new List<LlamaToken>(tokens);
            var context = new List<LlamaToken>();

            var conversation = new StringBuilder(prompt);

            Console.WriteLine(Context.Aggregate((a, b) => $"{a}\n{b}"));

            while (true)
            {
                LlamaCppInterop.llama_eval(model, sampled, context.Count, 16);
                context.AddRange(sampled);

                var id = LlamaCppInterop.llama_sample_top_p_top_k(model, context, TopK, TopP, Temperature, RepeatPenalty);
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

        static async Task WrappedInterfaceSampleWithoutSession()
        {
            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(ModelPath, ContextSize, Seed);

                model.Configure(options =>
                {
                    options.ThreadCount = ThreadCount;
                    options.TopK = TopK;
                    options.TopP = TopP;
                    options.Temperature = Temperature;
                    options.RepeatPenalty = RepeatPenalty;
                });

                var prompt = Context.Select((x, i) => $"{(i % 2 == 0 ? "ASSISTANT" : "USER")}:\n{x}\n").Aggregate((a, b) => $"{a}\n{b}");
                prompt = $"{prompt}\nASSISTANT:\n";

                var promptTokens = model.Tokenize(prompt, true);

                var conversation = new StringBuilder(prompt);

                Console.WriteLine(Context.Aggregate((a, b) => $"{a}\n{b}"));

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

        static async Task WrappedInterfaceSampleWithSession()
        {
            using (var model = new LlamaCpp(ModelName))
            {
                model.Load(ModelPath);

                model.Configure(options =>
                {
                    options.ThreadCount = 16;
                    options.TopK = 40;
                    options.TopP = 0.95f;
                    options.Temperature = 0.0f;
                    options.RepeatPenalty = 1.1f;
                });

                var session = model.CreateSession("Conversation #1");

                // Set the initial context, skipping last line since we're going to provide the prompt next
                session.Configure(options => options.InitialContext.AddRange(Context.SkipLast(1)));

                Console.WriteLine(session.InitialContext.Aggregate((a, b) => $"{a}\n{b}"));

                foreach (var prompt in Prompts)
                {
                    Console.WriteLine(prompt);

                    await foreach (var token in session.Predict(prompt))
                        Console.Write(token);
                }

                PrintTranscript(session.Conversation);
            }
        }

        static async Task GetEmbeddingsAsync()
        {
            var n_threads = ThreadCount;
            var n_past = 0;

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = ContextSize;
            cparams.embedding = true;

            var handle = LlamaCppInterop.llama_init_from_file(ModelPath, cparams);
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
            Console.Write(conversation);
        }
    }
}
