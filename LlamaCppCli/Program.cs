using System.Text;

using LlamaCppLib;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task Main(string[] args)
        {
            args = new[]
            {
                @"D:\LLM_MODELS\teknium\ggml-openhermes-2-mistral-7b-q8_0.gguf",
                @"D:\LLM_MODELS\teknium\ggml-openhermes-2.5-mistral-7b-q8_0.gguf",
                @"D:\LLM_MODELS\openchat\ggml-openchat_3.5-q8_0.gguf",
                @"D:\LLM_MODELS\NousResearch\ggml-yarn-mistral-7b-128k-q8_0.gguf",
                @"D:\LLM_MODELS\codellama\ggml-codellama-34b-instruct-q4_k.gguf",
            };

            // Multi-byte character encoding support (e.g. emojis)
            Console.OutputEncoding = Encoding.UTF8;

            // This barebone sample serves for testing the native API using raw function calls
            //await RunSampleRawAsync(args);

            // This sample serves for testing the library wrapped native core functionality
            //await RunSampleAsync(args);

            // Basic sample to demonstrate the library API
            {
                // Initialize
                using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });
                llm.LoadModel(args[0], new ModelOptions { Seed = 0, GpuLayers = 32 });

                // Start
                llm.StartAsync();

                // Prompting
                var promptTemplate = "<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant\n";

                var promptTask = async (string systemPrompt, string userPrompt) =>
                {
                    var prompt = llm.Prompt(
                        String.Format(promptTemplate, systemPrompt, userPrompt),
                        new SamplingOptions { Temperature = 0.0f },
                        prependBosToken: true,
                        processSpecialTokens: true
                    );

                    var response = new StringBuilder();
                    await foreach (var token in new TokenEnumerator(prompt))
                        response.Append(token);

                    Console.WriteLine($"{new String('=', 196)}");
                    Console.WriteLine($"{prompt.PromptingTime} / {prompt.SamplingTime}");
                    Console.WriteLine($"{new String('-', 196)}");
                    Console.Write(response.ToString());
                    Console.WriteLine($"\n{new String('=', 196)}");
                };

                // Requests
                var requests = new[]
                {
                    (
                        System: "You are an emoji expert.",
                        User: "What are the top five emojis on the net?"
                    ),
                    (
                        System: "You are an astrophysicist.",
                        User: "Write two tables listing the planets of the solar system, one in order from the Sun and the other in reverse order."
                    ),
                };

                // Inference
                await Task.WhenAll(requests.Select(request => promptTask(request.System, request.User)));

                // Stop
                await llm.StopAsync();
            }
        }
    }
}
