using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.Mime;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Web;

using LlamaCppLib;
using BertCppLib;

namespace LlamaCppCli
{
    using llama_token = System.Int32;

    internal class Program
    {
        static async Task Main(string[] args)
        {
#if DEBUG
            //args = new[] { "1", "http://localhost:5021", "meta-llama2-chat-13b-v1.0-q8_0", "60", "4096" };
            //args = new[] { "1", "http://localhost:5021", "openassistant-llama2-13b-orca-8k-3319-q8_0", "60", "8192" };
            args = new[] { "3" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(RunLocalSampleAsync), RunLocalSampleAsync),     // Run locally
                (nameof(RunRemoteSampleAsync), RunRemoteSampleAsync),   // Run via API
                (nameof(RunBertSampleAsync), RunBertSampleAsync),       // BERT
                (nameof(RunDebugSampleAsync), RunDebugSampleAsync),     // Simple (used for debugging)
            }
                .Select((sample, index) => (sample, index))
                .ToDictionary(k => k.sample.Name, v => (Index: v.index, v.sample.Func));

            var PrintAvailableSamples = () =>
            {
                Console.WriteLine($"Available sample(s):");
                foreach (var sample in samples)
                    Console.WriteLine($"    [{sample.Value.Index}] = {sample.Key}");
            };

            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} <SampleIndex> <SampleArgs>");
                PrintAvailableSamples();
                return;
            }

            var sampleIndex = Int32.Parse(args[0]);
            var sampleName = samples.SingleOrDefault(sample => sample.Value.Index == sampleIndex).Key;

            if (sampleName == default)
            {
                Console.WriteLine($"Sample not found ({sampleIndex}).");
                PrintAvailableSamples();
                return;
            }

            // Required for multi-byte character encoding (e.g. emojis)
            Console.OutputEncoding = Encoding.UTF8;

            await samples[sampleName].Func(args.Skip(1).ToArray());
        }

        static async Task RunLocalSampleAsync(string[] args)
        {
            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 0 model_path [gpu_layers] [ctx_length] [template]");
                return;
            }

            var modelPath = args[0];
            var gpuLayers = args.Length > 1 ? Int32.Parse(args[1]) : 0;
            var contextLength = args.Length > 2 ? Int32.Parse(args[2]) : 2048;
            var template = args.Length > 3 ? args[3] : "{0}";

            var modelOptions = new LlamaCppModelOptions
            {
                Seed = 0,
                ContextSize = contextLength,
                GpuLayers = gpuLayers,
                //UseMemoryLocking = false,
                //UseMemoryMapping = false,
                //RopeFrequencyBase = 10000.0f,
                //RopeFrequencyScale = 0.5f,
            };

            using var model = new LlamaCppModel();
            model.Load(modelPath, modelOptions);

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            var generateOptions = new LlamaCppGenerateOptions { Mirostat = Mirostat.Mirostat2, ThreadCount = 8 };
            var session = model.CreateSession();

            await Console.Out.WriteLineAsync(
                """

                Entering interactive mode:
                    * Press <Ctrl+C> to cancel running predictions
                    * Press <Enter> on an empty input prompt to quit
                """
            );

            // ------------------------------------------------------------------------------------------------------------------------------
            // Llama-2
            // ------------------------------------------------------------------------------------------------------------------------------
            // <s>[INST] <<SYS>>
            // {{ system_prompt }}
            // <</SYS>>
            //
            // {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>\
            // <s>[INST] {{ user_msg_2 }} [/INST] {{ model_answer_2 }} </s>\
            // <s>[INST] {{ user_msg_3 }} [/INST]
            //
            // https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L250
            //
            // self.tokenizer.encode(
            //     f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            //     bos=True,
            //     eos=True,
            // )
            // ------------------------------------------------------------------------------------------------------------------------------
            const string B_INST = "[INST]";
            const string E_INST = "[/INST]";
            const string B_SYS = "<<SYS>>\n";
            const string E_SYS = "\n<</SYS>>\n\n";
            const string SYS_PROMPT = "You are a helpful assistant.";
            // ------------------------------------------------------------------------------------------------------------------------------

            var first = true;

            while (true)
            {
                await Console.Out.WriteLineAsync("\nInput:");

                var userPrompt = await Console.In.ReadLineAsync() ?? String.Empty;
                if (String.IsNullOrWhiteSpace(userPrompt))
                    break;

                var prompt = $"{B_INST} {(first ? $"{B_SYS}{SYS_PROMPT}{E_SYS}" : "")}{userPrompt} {E_INST} ";

                var match = Regex.Match(userPrompt, @"^\/(?<Command>\w+)\s?""?(?<Arg>.*?)""?$");
                if (match.Success)
                {
                    var command = match.Groups["Command"].Value.ToLower();
                    var arg = match.Groups["Arg"].Value;

                    if (command == "load")
                    {
                        var path = Path.GetFullPath(arg);
                        await Console.Out.WriteAsync($"Loading prompt from \"{path}\"...");
                        if (!File.Exists(path))
                        {
                            await Console.Out.WriteLineAsync($" [File not found].");
                            continue;
                        }
                        prompt = File.ReadAllText(arg);
                        var tokenCount = model.Tokenize(prompt, true).Count;
                        await Console.Out.WriteLineAsync($" [{tokenCount} token(s)].");
                        if (tokenCount == 0 || tokenCount >= contextLength - 4)
                        {
                            await Console.Out.WriteLineAsync($"Context limit reached ({contextLength}).");
                            continue;
                        }
                        session.Reset();
                        model.ResetState();
                    }
                    else if (command == "reset")
                    {
                        session.Reset();
                        model.ResetState();
                        await Console.Out.WriteLineAsync($"Context reset.");
                        continue;
                    }
                    else if (command == "dump")
                    {
                        var separator = new String('=', Console.WindowWidth);
                        await Console.Out.WriteLineAsync(separator);
                        await Console.Out.WriteLineAsync(session.GetContextAsText());
                        await Console.Out.WriteLineAsync(separator);
                        continue;
                    }
                }

                await Console.Out.WriteLineAsync("\nOutput:");

                await foreach (var tokenString in session.GenerateTokenStringAsync(prompt, generateOptions, cancellationTokenSource.Token))
                {
                    await Console.Out.WriteAsync(tokenString);
                }

                if (cancellationTokenSource.IsCancellationRequested)
                {
                    await Console.Out.WriteAsync(" [Cancelled]");
                    cancellationTokenSource.Dispose();
                    cancellationTokenSource = new();
                }

                await Console.Out.WriteLineAsync();
                first = false;
            }

            await Console.Out.WriteLineAsync("Quitting...");
        }

        static async Task RunRemoteSampleAsync(string[] args)
        {
            if (args.Length < 2)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 1 base_url model_name [gpu_layers] [ctx_length] [template]");
                return;
            }

            var baseUrl = args[0];
            var modelName = args[1];
            var gpuLayers = args.Length > 2 ? Int32.Parse(args[2]) : 0;
            var contextLength = args.Length > 3 ? Int32.Parse(args[3]) : 4096;
            var template = args.Length > 4 ? args[4] : "{0}";

            var modelOptions = new LlamaCppModelOptions() { Seed = 0, ContextSize = contextLength, GpuLayers = gpuLayers, RopeFrequencyBase = 10000.0f, RopeFrequencyScale = 0.5f };

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            using var httpClient = new HttpClient();

            // Load model
            {
                await Console.Out.WriteAsync("Loading model...");
                var query = HttpUtility.ParseQueryString(String.Empty);
                query["modelName"] = modelName;
                query["modelOptions"] = JsonSerializer.Serialize(modelOptions);
                using var response = (await httpClient.GetAsync($"{baseUrl}/model/load?{query}")).EnsureSuccessStatusCode();
                await Console.Out.WriteLineAsync(" OK.");
            }

            // Create session
            Guid? sessionId;
            {
                await Console.Out.WriteAsync("Creating session...");
                using var response = (await httpClient.GetAsync($"{baseUrl}/session/create")).EnsureSuccessStatusCode();
                sessionId = Guid.Parse(await response.Content.ReadFromJsonAsync<string>() ?? String.Empty);
                await Console.Out.WriteLineAsync($" OK. [{sessionId}]");
            }

            {
                //var query = HttpUtility.ParseQueryString(String.Empty);
                //query["sessionId"] = $"{sessionId}";
                //(await httpClient.GetAsync($"{baseUrl}/session/reset?{query}")).EnsureSuccessStatusCode();
                //(await httpClient.GetAsync($"{baseUrl}/model/reset")).EnsureSuccessStatusCode();
            }

            // Generate token(s)
            {
                var generateOptions = new LlamaCppGenerateOptions { Temperature = 0.0f, Mirostat = Mirostat.Mirostat2 };

                await Console.Out.WriteLineAsync(
                    """

                    Entering interactive mode:
                        * Press <Ctrl+C> to cancel token generation
                        * Press <Enter> on an empty input prompt to quit
                    """
                );

                while (true)
                {
                    try
                    {
                        await Console.Out.WriteLineAsync("\nInput:");

                        var prompt = await Console.In.ReadLineAsync(cancellationTokenSource.Token) ?? String.Empty;
                        if (String.IsNullOrWhiteSpace(prompt))
                            break;

                        var match = Regex.Match(prompt, @"/load\s""?(.*[^""])?""?");
                        if (match.Success)
                        {
                            var path = Path.GetFullPath(match.Groups[1].Value);
                            prompt = File.ReadAllText(match.Groups[1].Value);
                        }

                        await Console.Out.WriteLineAsync("\nOutput:");

                        var content = new
                        {
                            SessionId = $"{sessionId}",
                            GenerateOptions = generateOptions,
                            Prompt = prompt,
                        };

                        var url = $"{baseUrl}/model/generate";
                        using var request = new HttpRequestMessage(HttpMethod.Post, url) { Content = new StringContent(JsonSerializer.Serialize(content), Encoding.UTF8, MediaTypeNames.Application.Json) };
                        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

                        using var response = (await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationTokenSource.Token)).EnsureSuccessStatusCode();

                        await using var stream = await response.Content.ReadAsStreamAsync(cancellationTokenSource.Token);
                        using var reader = new StreamReader(stream);

                        while (!reader.EndOfStream && !cancellationTokenSource.IsCancellationRequested)
                        {
                            var data = await reader.ReadLineAsync(cancellationTokenSource.Token);
                            if (data == null)
                                break;

                            var decodedToken = Regex.Match(data, @"(?<=data:\s).*").Value.Replace("\\n", "\n").Replace("\\t", "\t");
                            await Console.Out.WriteAsync(decodedToken);
                        }

                        await Console.Out.WriteLineAsync();
                        cancellationTokenSource.Token.ThrowIfCancellationRequested();

                        var query = HttpUtility.ParseQueryString(String.Empty);
                        query["sessionId"] = $"{sessionId}";
                        (await httpClient.GetAsync($"{baseUrl}/session/reset?{query}")).EnsureSuccessStatusCode();
                        (await httpClient.GetAsync($"{baseUrl}/model/reset")).EnsureSuccessStatusCode();
                    }
                    catch (Exception ex) when (ex is TaskCanceledException || ex is OperationCanceledException)
                    {
                        await Console.Out.WriteLineAsync(" [Cancelled]");

                        cancellationTokenSource.Dispose();
                        cancellationTokenSource = new();
                    }
                }
            }
        }

        static async Task RunDebugSampleAsync(string[] args)
        {
            _RunDebugSampleAsync(args);
            await Task.CompletedTask;
        }

        static void _RunDebugSampleAsync(string[] args)
        {
            //var path = @"D:\LLM_MODELS\meta-llama\llama-2-13b-chat.ggmlv3.q8_0.bin";
            var path = @"D:\LLM_MODELS\codellama\ggml-codellama-34b-instruct-Q4_K.gguf";
            //var path = @"D:\LLM_MODELS\tiiuae\ggml-falcon-40b-instruct-Q4_K.gguf";
            var prompt = File.ReadAllText(@"..\..\..\PROMPT.txt");

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            {
                LlamaCppInterop.llama_backend_init();

                var cparams = LlamaCppInterop.llama_context_default_params();
                cparams.n_ctx = 16384;
                cparams.n_gpu_layers = 42;

                var n_threads = 12;

                Console.WriteLine($"lparams.n_ctx = {cparams.n_ctx}");
                Console.WriteLine($"lparams.n_batch = {cparams.n_batch}");
                Console.WriteLine($"lparams.n_gpu_layers = {cparams.n_gpu_layers}");
                Console.WriteLine($"lparams.main_gpu = {cparams.main_gpu}");
                Console.WriteLine($"lparams.tensor_split = {cparams.tensor_split}");
                Console.WriteLine($"lparams.low_vram = {cparams.low_vram}");
                Console.WriteLine($"lparams.seed = {cparams.seed}");
                Console.WriteLine($"lparams.f16_kv = {cparams.f16_kv}");
                Console.WriteLine($"lparams.use_mmap = {cparams.use_mmap}");
                Console.WriteLine($"lparams.use_mlock = {cparams.use_mlock}");
                Console.WriteLine($"lparams.logits_all = {cparams.logits_all}");
                Console.WriteLine($"lparams.embedding = {cparams.embedding}");
                Console.WriteLine($"lparams.rope_freq_base = {cparams.rope_freq_base}");
                Console.WriteLine($"lparams.rope_freq_scale = {cparams.rope_freq_scale}");

                var model = LlamaCppInterop.llama_load_model_from_file(path, cparams);
                var ctx = LlamaCppInterop.llama_new_context_with_model(model, cparams);

                Console.WriteLine($"\nsystem_info: n_threads = {n_threads} / {Environment.ProcessorCount} | {LlamaCppInterop.llama_print_system_info()}");
                Console.WriteLine($"sampling: repeat_last_n = 64, repeat_penalty = 1.1, presence_penalty = 0.0, frequency_penalty = 0.0, top_k = 40, tfs_z = 1.0, top_p = 0.95, typical_p = 1.0, temp = 0.8, mirostat = 2, mirostat_lr = 0.1, mirostat_ent = 5.0");
                Console.WriteLine($"generate: n_ctx = {cparams.n_ctx}, n_batch = {cparams.n_batch}, n_predict = -1, n_keep = 0");

                var is_spm = LlamaCppInterop.llama_vocab_type(ctx) == LlamaCppInterop.llama_vocab_type_.LLAMA_VOCAB_TYPE_SPM;
                LlamaCppInterop.llama_tokenize(ctx, prompt, out var tokens, is_spm);

                var tokens_list = new List<llama_token>();
                for (var i = 0; i < tokens.Length; i++)
                    tokens_list.Add(tokens[i]);

                var max_context_size = LlamaCppInterop.llama_n_ctx(ctx);
                var max_tokens_list_size = max_context_size - 4;
                if (tokens_list.Count > max_tokens_list_size)
                {
                    Console.WriteLine($"error: prompt too long ({tokens_list.Count} tokens, max {max_tokens_list_size})");
                    return;
                }

                var tokens_context = new List<llama_token>();
                tokens_context.AddRange(tokens_list);

                Console.WriteLine(new String('=', Console.WindowWidth));
                Console.WriteLine($"tokens_context = {tokens_context.Count}");
                Console.WriteLine($"llama_n_ctx = {LlamaCppInterop.llama_n_ctx(ctx)}");
                Console.WriteLine($"llama_get_kv_cache_token_count = {LlamaCppInterop.llama_get_kv_cache_token_count(ctx)}");
                Console.WriteLine(new String('=', Console.WindowWidth));

                var n_past = 0;

                while (LlamaCppInterop.llama_get_kv_cache_token_count(ctx) < LlamaCppInterop.llama_n_ctx(ctx) && !cancellationTokenSource.IsCancellationRequested)
                {
                    for (var i = 0; i < tokens_list.Count && !cancellationTokenSource.IsCancellationRequested; i += cparams.n_batch)
                    {
                        var n_eval = tokens_list.Count - i;
                        if (n_eval > cparams.n_batch)
                            n_eval = cparams.n_batch;

                        LlamaCppInterop.llama_eval(ctx, tokens_list.Skip(i).ToArray(), n_eval, n_past, n_threads);
                        n_past += n_eval;
                    }

                    if (cancellationTokenSource.IsCancellationRequested)
                        break;

                    tokens_list.Clear();

                    var logits = LlamaCppInterop.llama_get_logits(ctx);
                    var n_vocab = LlamaCppInterop.llama_n_vocab(ctx);

                    var candidates = new LlamaCppInterop.llama_token_data[n_vocab];
                    for (llama_token token_id = 0; token_id < n_vocab && !cancellationTokenSource.IsCancellationRequested; token_id++)
                        candidates[token_id] = new LlamaCppInterop.llama_token_data { id = token_id, logit = logits[token_id], p = 0.0f };

                    if (cancellationTokenSource.IsCancellationRequested)
                        break;

                    var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates, size = (nuint)candidates.Length, sorted = false };

                    var new_token_id = LlamaCppInterop.llama_sample_token_greedy(ctx, candidates_p);
                    if (new_token_id == LlamaCppInterop.llama_token_eos(ctx))
                    {
                        Console.WriteLine(" [end of text]");
                        break;
                    }

                    var token = LlamaCppInterop.llama_token_to_str(ctx, new_token_id);
                    Console.Write(token);

                    tokens_list.Add(new_token_id);
                    tokens_context.Add(new_token_id);
                }

                Console.WriteLine(new String('=', Console.WindowWidth));
                Console.WriteLine($"tokens_context = {tokens_context.Count}");
                Console.WriteLine($"llama_n_ctx = {LlamaCppInterop.llama_n_ctx(ctx)}");
                Console.WriteLine($"llama_get_kv_cache_token_count = {LlamaCppInterop.llama_get_kv_cache_token_count(ctx)}");
                Console.WriteLine(new String('=', Console.WindowWidth));

                LlamaCppInterop.llama_free(ctx);
                LlamaCppInterop.llama_free_model(model);
                LlamaCppInterop.llama_backend_free();
            }
        }

        static async Task RunBertSampleAsync(string[] args)
        {
            //var path = @"D:\LLM_MODELS\sentence-transformers\ggml-all-MiniLM-L12-v2-f32.bin";
            //var path = @"D:\LLM_MODELS\intfloat\ggml-e5-large-v2-f16.bin";
            var path = @"D:\LLM_MODELS\BAAI\ggml-bge-large-en-f32.bin";

            BertCppInterop.bert_params_parse(new[] { "-t", $"8", "-p", "What is the capital of the United States?", "-m", path }, out var bparams);

            var documents = new[]
            {
                "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
                "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
                "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
                "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
                "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
            };

            var ctx = BertCppInterop.bert_load_from_file(path);

            var documentsEmbeddings = documents
                .Select(x => BertCppInterop.bert_tokenize(ctx, x))
                .Select(x => BertCppInterop.bert_eval(ctx, bparams.n_threads, x))
                .ToList();

            var queryEmbeddings = BertCppInterop.bert_eval(ctx, bparams.n_threads, BertCppInterop.bert_tokenize(ctx, bparams.prompt));

            var cosineSimilarities = documentsEmbeddings
                .Select(document => CosineSimilarity(queryEmbeddings, document))
                .ToList();

            var results = documents
                .Zip(cosineSimilarities, (x, similarity) => new { Document = x, CosineSimilarity = similarity })
                .OrderByDescending(x => x.CosineSimilarity)
                .ToList();

            Console.WriteLine($"[{bparams.prompt}]");
            results.ForEach(x => Console.WriteLine($"[{x.CosineSimilarity}][{x.Document}]"));

            BertCppInterop.bert_free(ctx);

            await Task.CompletedTask;
        }

        static float CosineSimilarity(float[] vec1, float[] vec2)
        {
            if (vec1.Length != vec2.Length)
                throw new ArgumentException("Vectors must be of the same size.");

            var dotProduct = vec1.Zip(vec2, (a, b) => a * b).Sum();
            var normA = Math.Sqrt(vec1.Sum(a => Math.Pow(a, 2)));
            var normB = Math.Sqrt(vec2.Sum(b => Math.Pow(b, 2)));

            if (normA == 0.0 || normB == 0.0)
                throw new ArgumentException("Vectors must not be zero vectors.");

            return (float)(dotProduct / (normA * normB));
        }
    }
}
