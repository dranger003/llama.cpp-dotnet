using System.Net.Mime;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Web;

using LlamaCppLib;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task Main(string[] args)
        {
            var models = new[]
            {
                @"D:\LLM_MODELS\01-ai\ggml-yi-34b-q4_k.gguf",                           // none

                @"D:\LLM_MODELS\CalderaAI\ggml-hexoteric-7b-q8_0.gguf",                 // {context}<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
                @"D:\LLM_MODELS\CalderaAI\ggml-naberius-7b-q8_0.gguf",

                @"D:\LLM_MODELS\CausalLM\ggml-causallm-7b-q8_0.gguf",                   // {context}<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
                @"D:\LLM_MODELS\CausalLM\ggml-causallm-14b-q8_0.gguf",

                @"D:\LLM_MODELS\NousResearch\ggml-yarn-mistral-7b-64k-q8_0.gguf",       // none
                @"D:\LLM_MODELS\NousResearch\ggml-yarn-mistral-7b-128k-q8_0.gguf",
                @"D:\LLM_MODELS\NousResearch\ggml-yarn-llama-2-13b-64k-q4_k.gguf",

                @"D:\LLM_MODELS\codellama\ggml-codellama-7b-instruct-q8_0.gguf",        // {context}[INST] {user} [/INST]\n
                @"D:\LLM_MODELS\codellama\ggml-codellama-13b-instruct-q4_k.gguf",
                @"D:\LLM_MODELS\codellama\ggml-codellama-34b-instruct-q4_k.gguf",

                @"D:\LLM_MODELS\teknium\ggml-hermes-trismegistus-mistral-7b-q8_0.gguf", // {context}<|end_of_turn|>GPT4 Correct User: {user}<|end_of_turn|>GPT4 Correct Assistant:\n
                @"D:\LLM_MODELS\teknium\ggml-openhermes-2.5-mistral-7b-q8_0.gguf",
                @"D:\LLM_MODELS\teknium\ggml-openhermes-2-mistral-7b-q8_0.gguf",

                @"D:\LLM_MODELS\openchat\ggml-openchat_3.5-q8_0.gguf",                  // {context}<|end_of_turn|>GPT4 Correct User: {user}<|end_of_turn|>GPT4 Correct Assistant:\n

                @"D:\LLM_MODELS\ehartford\ggml-dolphin-2.2.1-mistral-7b-q8_0.gguf",     // {context}<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n

                @"D:\LLM_MODELS\sequelbox\ggml-daringfortitude-13b-q8_0.gguf",          // {context}[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
            };

            args = new[] { models[16] };

            // Multi-byte character encoding support (e.g. emojis)
            Console.OutputEncoding = Encoding.UTF8;

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { cancellationTokenSource.Cancel(!(e.Cancel = true)); };

            // This sample serves for testing the native API using raw function calls
            //await RunSampleRawAsync(args);

            //// This sample serves for testing the library wrapped native core functionality
            //await RunSampleAsync(args);

            //// Basic sample to demonstrate the library API
            //{
            //    int separatorWidth = 128;
            //    string singleSeparator = new String('-', separatorWidth);
            //    string doubleSeparator = new String('=', separatorWidth);

            //    // Initialize
            //    using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });
            //    llm.LoadModel(args[0], new ModelOptions { Seed = 0, GpuLayers = 32 });

            //    // Start
            //    llm.StartAsync();

            //    // Prompting
            //    var promptTemplate = "<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant\n";

            //    var promptTask = async (string systemPrompt, string userPrompt, bool streamTokens = false) =>
            //    {
            //        var prompt = llm.Prompt(
            //            String.Format(promptTemplate, systemPrompt, userPrompt),
            //            new SamplingOptions { Temperature = 0.0f },
            //            prependBosToken: true,
            //            processSpecialTokens: true
            //        );

            //        if (streamTokens)
            //        {
            //            Console.WriteLine(doubleSeparator);
            //            Console.WriteLine(userPrompt);
            //            Console.WriteLine(singleSeparator);

            //            await foreach (var token in new TokenEnumerator(prompt))
            //                Console.Write(token);
            //        }
            //        else
            //        {
            //            var response = new StringBuilder();
            //            await foreach (var token in new TokenEnumerator(prompt))
            //                response.Append(token);

            //            Console.WriteLine(doubleSeparator);
            //            Console.WriteLine(userPrompt);
            //            Console.WriteLine(singleSeparator);
            //            Console.Write(response);
            //        }

            //        // Statistics
            //        Console.WriteLine();
            //        Console.WriteLine(singleSeparator);
            //        Console.WriteLine($"Prompting: {prompt.PromptingSpeed:F2} t/s / Sampling: {prompt.SamplingSpeed:F2} t/s");
            //        Console.WriteLine(doubleSeparator);
            //    };

            //    var prompts = new[]
            //    {
            //        (
            //            System: "You are an emoji expert.",
            //            User: "What are the top five emojis on the net?"
            //        ),
            //        (
            //            System: "You are an astrophysicist.",
            //            User: "Write two tables listing the planets of the solar system, one in order from the Sun and the other in reverse order."
            //        ),
            //        (
            //            System: "You are an AI scientist and researcher.",
            //            User: "What do you think is the best way (briefly) to run a large language model using cutting edge technology?"
            //        ),
            //        (
            //            System: "Your task is to summarize the text provided in a bullet list of main topics.",
            //            User: File.ReadAllText(@"D:\LLM_MODELS\ESSAY.txt")
            //        ),
            //    };

            //    Console.WriteLine(doubleSeparator);

            //    // Streaming (single)
            //    Console.WriteLine($"Streaming inference...");
            //    await Task.WhenAll(prompts.Select(request => promptTask(request.System, request.User, true)).Skip(1).Take(1));

            //    // Buffering (multiple)
            //    Console.WriteLine($"Buffering inference...");
            //    await Task.WhenAll(prompts.Select(request => promptTask(request.System, request.User, false)));

            //    // Stop
            //    await llm.StopAsync();
            //}

            //// Barebone sample
            //{
            //    var promptTemplate = "<|im_start|>system\n{0}<|im_end|>\n<|im_start|>user\n{1}<|im_end|>\n<|im_start|>assistant\n";
            //    var systemPrompt = "You are a helpful assistant.";

            //    // Initialize
            //    using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });
            //    llm.LoadModel(args[0], new ModelOptions { Seed = 0, GpuLayers = 64 });

            //    // Start
            //    llm.StartAsync();

            //    Console.Write($"{new String('=', 196)}");

            //    while (true)
            //    {
            //        // Input
            //        Console.Write($"\n> ");
            //        var userPrompt = Console.ReadLine() ?? String.Empty;
            //        if (String.IsNullOrWhiteSpace(userPrompt)) break;

            //        var promptText = String.Format(promptTemplate, systemPrompt, userPrompt);

            //        // Load (Optional)
            //        var match = Regex.Match(userPrompt, @"/load\s+""?([^""\s]+)""?");
            //        if (match.Success) promptText = File.ReadAllText(match.Groups[1].Value);

            //        // Prompting
            //        var prompt = llm.Prompt(promptText, new SamplingOptions { Temperature = 0.0f });

            //        // Inference
            //        Console.WriteLine($"{new String('=', 196)}");
            //        {
            //            await foreach (var token in new TokenEnumerator(prompt, cancellationTokenSource.Token))
            //                Console.Write(token);

            //            // Cancellation
            //            if (cancellationTokenSource.IsCancellationRequested)
            //            {
            //                Console.Write($" [Cancelled]");
            //                cancellationTokenSource = new();
            //            }
            //        }
            //        Console.Write($"\n{new String('=', 196)}");
            //    }

            //    // Stop
            //    await llm.StopAsync();
            //}

            // API Client
            {
                var baseUrl = "http://localhost:5021";
                using var httpClient = new HttpClient();

                {
                    var query = HttpUtility.ParseQueryString(String.Empty);
                    query["name"] = "openhermes-mistral-7b-2.5";
                    using var response = await httpClient.GetAsync($"{baseUrl}/load?{query}");
                    Console.WriteLine(await response.Content.ReadAsStringAsync());
                }
                //{
                //    using var response = await httpClient.GetAsync($"{baseUrl}/unload?");
                //    Console.WriteLine(await response.Content.ReadAsStringAsync());
                //}
                {
                    var promptText = """
                        <|im_start|>system
                        You are an emoji expert.<|im_end|>
                        <|im_start|>user
                        Write a table listing to most common emojis.<|im_end|>
                        <|im_start|>assistant

                        """;

                    using var response = await httpClient.PostAsync(
                        $"{baseUrl}/prompt",
                        new StringContent(
                            JsonSerializer.Serialize(new { PromptText = promptText, SamplingOptions = new SamplingOptions { Temperature = 0.0f } }),
                            Encoding.UTF8,
                            MediaTypeNames.Application.Json
                        ),
                        HttpCompletionOption.ResponseHeadersRead,
                        cancellationTokenSource.Token
                    );

                    using var reader = new StreamReader(await response.Content.ReadAsStreamAsync(cancellationTokenSource.Token));
                    while (!reader.EndOfStream)
                    {
                        var data = await reader.ReadLineAsync(cancellationTokenSource.Token);
                        if (data == null)
                            break;

                        var token = Encoding.UTF8.GetString(Convert.FromBase64String(Regex.Replace(data, "^data: |\n\n$", String.Empty)));
                        Console.Write(token);
                    }

                    Console.WriteLine();
                }
            }
        }
    }
}
