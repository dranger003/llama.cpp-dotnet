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
using System.Diagnostics;
using System.Data.Common;

namespace LlamaCppCli
{
    using llama_token = System.Int32;

    internal class Program
    {
        static async Task Main(string[] args)
        {
            //#if DEBUG
            //            //args = new[] { "1", "http://localhost:5021", "meta-llama2-chat-13b-v1.0-q8_0", "60", "4096" };
            //            //args = new[] { "1", "http://localhost:5021", "openassistant-llama2-13b-orca-8k-3319-q8_0", "60", "8192" };
            //            //args = new[] { "1", "http://localhost:5021", "codellama-7b-q8_0", "42", "16384" };
            //            args = new[] { "4" };
            //#endif
            //            var samples = new (string Name, Func<string[], Task> Func)[]
            //            {
            //                (nameof(RunLocalSampleAsync), RunLocalSampleAsync),             // Run locally
            //                (nameof(RunRemoteSampleAsync), RunRemoteSampleAsync),           // Run via API
            //                (nameof(RunBertSampleAsync), RunBertSampleAsync),               // BERT
            //                (nameof(RunEmbeddingsSampleAsync), RunEmbeddingsSampleAsync),
            //                (nameof(RunDebugSampleAsync), RunDebugSampleAsync),             // Simple (used for debugging)
            //            }
            //                .Select((sample, index) => (sample, index))
            //                .ToDictionary(k => k.sample.Name, v => (Index: v.index, v.sample.Func));

            //            var PrintAvailableSamples = () =>
            //            {
            //                Console.WriteLine($"Available sample(s):");
            //                foreach (var sample in samples)
            //                    Console.WriteLine($"    [{sample.Value.Index}] = {sample.Key}");
            //            };

            //            if (args.Length < 1)
            //            {
            //                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} <SampleIndex> <SampleArgs>");
            //                PrintAvailableSamples();
            //                return;
            //            }

            //            var sampleIndex = Int32.Parse(args[0]);
            //            var sampleName = samples.SingleOrDefault(sample => sample.Value.Index == sampleIndex).Key;

            //            if (sampleName == default)
            //            {
            //                Console.WriteLine($"Sample not found ({sampleIndex}).");
            //                PrintAvailableSamples();
            //                return;
            //            }

            //            // Required for multi-byte character encoding (e.g. emojis)
            //            Console.OutputEncoding = Encoding.UTF8;

            //            await samples[sampleName].Func(args.Skip(1).ToArray());

            args = new[]
            {
                "D:\\LLM_MODELS\\codellama\\ggml-codeLlama-7b-instruct-q8_0.gguf",
                args.Length > 0 ? args[0] : "1",
            };

            //await RunSimpleSampleAsync(args);
            await RunBatchedSampleAsync(args);
        }

        static async Task RunSimpleSampleAsync(string[] args)
        {
            RunSimpleSample(args);
            await Task.CompletedTask;
        }

        static void RunSimpleSample(string[] args)
        {
            var top_k = 40;
            var top_p = 0.9f;
            var temp = 0.0f;

            // Backend init
            LlamaCppInterop.llama_backend_init(false);

            // Model init
            var mdl_params = LlamaCppInterop.llama_model_default_params();
            mdl_params.n_gpu_layers = 64;
            var mdl = LlamaCppInterop.llama_load_model_from_file(args[0], mdl_params);

            // Context init
            var ctx_params = LlamaCppInterop.llama_context_default_params();
            ctx_params.seed = 1;
            ctx_params.n_ctx = 16384;
            ctx_params.n_batch = 512;
            ctx_params.n_threads = 1;
            ctx_params.n_threads_batch = 1;
            var ctx = LlamaCppInterop.llama_new_context_with_model(mdl, ctx_params);

            // Prompt tokenization
            var tokens_list = new List<llama_token>() { LlamaCppInterop.llama_token_bos(ctx) };
            {
                tokens_list.AddRange(LlamaCppInterop.llama_tokenize(ctx, "[INST] List the planets of the solar system in order from the Sun. [/INST]\nMercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune.").ToArray());
                tokens_list.AddRange(new[] { LlamaCppInterop.llama_token_eos(ctx), LlamaCppInterop.llama_token_bos(ctx) });
                tokens_list.AddRange(LlamaCppInterop.llama_tokenize(ctx, "[INST] Write a table listing the planets of the solar system in reverse order from the Sun. [/INST]\n").ToArray());
            }

            // Prompt batching
            var batch = LlamaCppInterop.llama_batch_init((int)ctx_params.n_batch, 0);
            batch.n_tokens = tokens_list.Count;
            for (var i = 0; i < batch.n_tokens; i++)
            {
                batch.token((int)ctx_params.n_batch)[i] = tokens_list[i];
                batch.pos((int)ctx_params.n_batch)[i] = i;
                batch.seq_id((int)ctx_params.n_batch)[i] = 0;
                batch.logits((int)ctx_params.n_batch)[i] = 0;
            }
            batch.logits((int)ctx_params.n_batch)[batch.n_tokens - 1] = 1;

            // Prompt decoding
            LlamaCppInterop.llama_decode(ctx, batch);

            var candidates = new LlamaCppInterop.llama_token_data[LlamaCppInterop.llama_n_vocab(mdl)];
            var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates, size = (nuint)candidates.Length, sorted = false };

            var n_cur = batch.n_tokens;

            var sw = Stopwatch.StartNew();

            while (true)
            {
                var logits = LlamaCppInterop.llama_get_logits_ith(ctx, batch.n_tokens - 1);
                for (var token_id = 0; token_id < candidates.Length; token_id++)
                    candidates[token_id] = new LlamaCppInterop.llama_token_data { id = token_id, logit = logits[token_id], p = 0.0f };

                LlamaCppInterop.llama_sample_top_k(ctx, candidates_p, top_k, 1);
                LlamaCppInterop.llama_sample_top_p(ctx, candidates_p, top_p, 1);
                LlamaCppInterop.llama_sample_temp(ctx, candidates_p, temp);
                var new_token_id = LlamaCppInterop.llama_sample_token(ctx, candidates_p);

                if (new_token_id == LlamaCppInterop.llama_token_eos(ctx))
                    break;

                Console.Write(Encoding.ASCII.GetString(LlamaCppInterop.llama_token_to_piece(ctx, new_token_id)));

                batch.n_tokens = 1;
                batch.token((int)ctx_params.n_batch)[0] = new_token_id;
                batch.pos((int)ctx_params.n_batch)[0] = n_cur;
                batch.seq_id((int)ctx_params.n_batch)[0] = 0;
                batch.logits((int)ctx_params.n_batch)[0] = 1;

                ++n_cur;

                LlamaCppInterop.llama_decode(ctx, batch);
            }

            Console.WriteLine($"\nElapsed = {sw.Elapsed}\n");

            LlamaCppInterop.llama_batch_free(batch);
            LlamaCppInterop.llama_free(ctx);
            LlamaCppInterop.llama_free_model(mdl);
            LlamaCppInterop.llama_backend_free();
        }

        static async Task RunBatchedSampleAsync(string[] args)
        {
            RunBatchedSample(args);
            await Task.CompletedTask;
        }

        static void RunBatchedSample(string[] args)
        {
            var n_parallel = args.Length > 1 ? Int32.Parse(args[1]) : 1;

            var top_k = 40;
            var top_p = 0.9f;
            var temp = 0.0f;

            // Backend init
            LlamaCppInterop.llama_backend_init(false);

            // Model init
            var mdl_params = LlamaCppInterop.llama_model_default_params();
            mdl_params.n_gpu_layers = 64;
            var mdl = LlamaCppInterop.llama_load_model_from_file(args[0], mdl_params);

            // Context init
            var ctx_params = LlamaCppInterop.llama_context_default_params();
            ctx_params.seed = 1;
            ctx_params.n_ctx = 16384;
            ctx_params.n_batch = 512;
            ctx_params.n_threads = 1;
            ctx_params.n_threads_batch = 1;
            var ctx = LlamaCppInterop.llama_new_context_with_model(mdl, ctx_params);

            // Prompt tokenization
            var tokens_list = new List<llama_token>() { LlamaCppInterop.llama_token_bos(ctx) };
            {
                tokens_list.AddRange(LlamaCppInterop.llama_tokenize(ctx, "[INST] List the planets of the solar system in order from the Sun. [/INST]\nMercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune.").ToArray());
                tokens_list.AddRange(new[] { LlamaCppInterop.llama_token_eos(ctx), LlamaCppInterop.llama_token_bos(ctx) });
                tokens_list.AddRange(LlamaCppInterop.llama_tokenize(ctx, "[INST] Write a table listing the planets of the solar system in reverse order from the Sun. [/INST]\n").ToArray());
            }

            // Prompt batching
            var batch = LlamaCppInterop.llama_batch_init((int)ctx_params.n_batch, 0);
            batch.n_tokens = tokens_list.Count;
            for (var i = 0; i < batch.n_tokens; i++)
            {
                batch.token((int)ctx_params.n_batch)[i] = tokens_list[i];
                batch.pos((int)ctx_params.n_batch)[i] = i;
                batch.seq_id((int)ctx_params.n_batch)[i] = 0;
                batch.logits((int)ctx_params.n_batch)[i] = 0;
            }
            batch.logits((int)ctx_params.n_batch)[batch.n_tokens - 1] = 1;

            // Prompt decoding
            LlamaCppInterop.llama_decode(ctx, batch);

            // Prompt duplication
            for (var i = 1; i < n_parallel; ++i)
                LlamaCppInterop.llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens);

            var candidates = new LlamaCppInterop.llama_token_data[LlamaCppInterop.llama_n_vocab(mdl)];
            var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates, size = (nuint)candidates.Length, sorted = false };

            var streams = new StringBuilder[n_parallel];
            for (var i = 0; i < streams.Length; ++i)
                streams[i] = new StringBuilder();

            var i_batch = Enumerable.Repeat(batch.n_tokens - 1, n_parallel).ToArray();

            var n_cur = batch.n_tokens;

            var sw = Stopwatch.StartNew();

            while (true)
            {
                batch.n_tokens = 0;

                for (var i = 0; i < n_parallel; ++i)
                {
                    if (i_batch[i] < 0)
                        continue;

                    var logits = LlamaCppInterop.llama_get_logits_ith(ctx, i_batch[i]);
                    for (var token_id = 0; token_id < candidates.Length; token_id++)
                        candidates[token_id] = new LlamaCppInterop.llama_token_data { id = token_id, logit = logits[token_id], p = 0.0f };

                    LlamaCppInterop.llama_sample_top_k(ctx, candidates_p, top_k, 1);
                    LlamaCppInterop.llama_sample_top_p(ctx, candidates_p, top_p, 1);
                    LlamaCppInterop.llama_sample_temp(ctx, candidates_p, temp);
                    var new_token_id = LlamaCppInterop.llama_sample_token(ctx, candidates_p);

                    if (new_token_id == LlamaCppInterop.llama_token_eos(ctx))
                    {
                        i_batch[i] = -1;
                        continue;
                    }

                    streams[i].Append(Encoding.ASCII.GetString(LlamaCppInterop.llama_token_to_piece(ctx, new_token_id)));
                    i_batch[i] = batch.n_tokens;

                    batch.token((int)ctx_params.n_batch)[batch.n_tokens] = new_token_id;
                    batch.pos((int)ctx_params.n_batch)[batch.n_tokens] = n_cur;
                    batch.seq_id((int)ctx_params.n_batch)[batch.n_tokens] = i;
                    batch.logits((int)ctx_params.n_batch)[batch.n_tokens] = 1;
                    batch.n_tokens += 1;
                }

                if (batch.n_tokens == 0)
                    break;

                ++n_cur;

                LlamaCppInterop.llama_decode(ctx, batch);
            }

            Console.WriteLine($"\nElapsed = {sw.Elapsed}\n");

            for (var i = 0; i < n_parallel; ++i)
                Console.WriteLine($"[{i}]:\n{streams[i]}\n");

            LlamaCppInterop.llama_batch_free(batch);
            LlamaCppInterop.llama_free(ctx);
            LlamaCppInterop.llama_free_model(mdl);
            LlamaCppInterop.llama_backend_free();
        }

        //static async Task RunLocalSampleAsync(string[] args)
        //{
        //    if (args.Length < 0)
        //    {
        //        await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 0 [model_path] [gpu_layers] [ctx_length] [template]");
        //        return;
        //    }

        //    var modelPath = args.Length > 0 ? args[0] : @"D:\LLM_MODELS\codellama\ggml-codellama-7b-instruct-Q8_0.gguf";
        //    var gpuLayers = args.Length > 1 ? Int32.Parse(args[1]) : 64;
        //    var contextLength = args.Length > 2 ? Int32.Parse(args[2]) : 16384;
        //    var template = args.Length > 3 ? args[3] : "{0}";

        //    var modelOptions = new LlamaCppModelOptions
        //    {
        //        Seed = 0,
        //        ContextSize = contextLength,
        //        GpuLayers = gpuLayers,
        //        //RopeFrequencyBase = 10000.0f,
        //        //RopeFrequencyScale = 0.5f,
        //        //LowVRAM = true,
        //        //UseMemoryLocking = false,
        //        //UseMemoryMapping = false,
        //    };

        //    using var model = new LlamaCppModel();
        //    model.Load(modelPath, modelOptions);

        //    var cancellationTokenSource = new CancellationTokenSource();
        //    Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

        //    var generateOptions = new LlamaCppGenerateOptions { Temperature = 0.0f, Mirostat = Mirostat.Disabled, ThreadCount = 1 };
        //    var session = model.CreateSession();

        //    await Console.Out.WriteLineAsync(
        //        """

        //        Entering interactive mode:
        //            * Press <Ctrl+C> to cancel running predictions
        //            * Press <Enter> on an empty input prompt to quit
        //        """
        //    );

        //    // ------------------------------------------------------------------------------------------------------------------------------
        //    // Llama-2
        //    // ------------------------------------------------------------------------------------------------------------------------------
        //    // <s>[INST] <<SYS>>
        //    // {{ system_prompt }}
        //    // <</SYS>>
        //    //
        //    // {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>\
        //    // <s>[INST] {{ user_msg_2 }} [/INST] {{ model_answer_2 }} </s>\
        //    // <s>[INST] {{ user_msg_3 }} [/INST]
        //    //
        //    // https://github.com/facebookresearch/llama/blob/6c7fe276574e78057f917549435a2554000a876d/llama/generation.py#L250
        //    //
        //    // self.tokenizer.encode(
        //    //     f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
        //    //     bos=True,
        //    //     eos=True,
        //    // )
        //    // ------------------------------------------------------------------------------------------------------------------------------
        //    const string B_INST = "[INST]";
        //    const string E_INST = "[/INST]";
        //    const string B_SYS = "<<SYS>>\n";
        //    const string E_SYS = "\n<</SYS>>\n\n";
        //    const string SYS_PROMPT = "You are a helpful assistant.";
        //    // ------------------------------------------------------------------------------------------------------------------------------

        //    var first = true;

        //    while (true)
        //    {
        //        await Console.Out.WriteLineAsync("\nInput:");

        //        var userPrompt = await Console.In.ReadLineAsync() ?? String.Empty;
        //        if (String.IsNullOrWhiteSpace(userPrompt))
        //            break;

        //        var prompt = $"{B_INST} {(first ? $"{B_SYS}{SYS_PROMPT}{E_SYS}" : "")}{userPrompt} {E_INST} ";

        //        var match = Regex.Match(userPrompt, @"^\/(?<Command>\w+)\s?""?(?<Arg>.*?)""?$");
        //        if (match.Success)
        //        {
        //            var command = match.Groups["Command"].Value.ToLower();
        //            var arg = match.Groups["Arg"].Value;

        //            if (command == "load")
        //            {
        //                var path = Path.GetFullPath(arg);
        //                await Console.Out.WriteAsync($"Loading prompt from \"{path}\"...");
        //                if (!File.Exists(path))
        //                {
        //                    await Console.Out.WriteLineAsync($" [File not found].");
        //                    continue;
        //                }
        //                prompt = File.ReadAllText(arg);
        //                var tokenCount = model.Tokenize(prompt, true).Count;
        //                await Console.Out.WriteLineAsync($" [{tokenCount} token(s)].");
        //                if (tokenCount == 0 || tokenCount >= contextLength - 4)
        //                {
        //                    await Console.Out.WriteLineAsync($"Context limit reached ({contextLength}).");
        //                    continue;
        //                }
        //                session.Reset();
        //                //model.ResetState();
        //            }
        //            else if (command == "reset")
        //            {
        //                session.Reset();
        //                model.ResetState();
        //                await Console.Out.WriteLineAsync($"Context reset.");
        //                continue;
        //            }
        //            else if (command == "dump")
        //            {
        //                var separator = new String('=', Console.WindowWidth);
        //                await Console.Out.WriteLineAsync(separator);
        //                await Console.Out.WriteLineAsync(session.GetContextAsText());
        //                await Console.Out.WriteLineAsync(separator);
        //                continue;
        //            }
        //        }

        //        await Console.Out.WriteLineAsync("\nOutput:");

        //        await foreach (var tokenString in session.GenerateTokenStringAsync(prompt, generateOptions, cancellationTokenSource.Token))
        //        {
        //            await Console.Out.WriteAsync(tokenString);
        //        }

        //        if (cancellationTokenSource.IsCancellationRequested)
        //        {
        //            await Console.Out.WriteAsync(" [Cancelled]");
        //            cancellationTokenSource.Dispose();
        //            cancellationTokenSource = new();
        //        }

        //        await Console.Out.WriteLineAsync();
        //        first = false;
        //    }

        //    await Console.Out.WriteLineAsync("Quitting...");

        //    await Task.CompletedTask;
        //}

        //static async Task RunRemoteSampleAsync(string[] args)
        //{
        //    if (args.Length < 2)
        //    {
        //        await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 1 base_url model_name [gpu_layers] [ctx_length] [template]");
        //        return;
        //    }

        //    var baseUrl = args[0];
        //    var modelName = args[1];
        //    var gpuLayers = args.Length > 2 ? Int32.Parse(args[2]) : 0;
        //    var contextLength = args.Length > 3 ? Int32.Parse(args[3]) : 4096;
        //    var template = args.Length > 4 ? args[4] : "{0}";

        //    var modelOptions = new LlamaCppModelOptions() { Seed = 0, ContextSize = contextLength, GpuLayers = gpuLayers, RopeFrequencyBase = 10000.0f, RopeFrequencyScale = 0.5f };

        //    var cancellationTokenSource = new CancellationTokenSource();
        //    Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

        //    using var httpClient = new HttpClient();

        //    // Load model
        //    {
        //        await Console.Out.WriteAsync("Loading model...");
        //        var query = HttpUtility.ParseQueryString(String.Empty);
        //        query["modelName"] = modelName;
        //        query["modelOptions"] = JsonSerializer.Serialize(modelOptions);
        //        using var response = (await httpClient.GetAsync($"{baseUrl}/model/load?{query}")).EnsureSuccessStatusCode();
        //        await Console.Out.WriteLineAsync(" OK.");
        //    }

        //    // Create session
        //    Guid? sessionId;
        //    {
        //        await Console.Out.WriteAsync("Creating session...");
        //        using var response = (await httpClient.GetAsync($"{baseUrl}/session/create")).EnsureSuccessStatusCode();
        //        sessionId = Guid.Parse(await response.Content.ReadFromJsonAsync<string>() ?? String.Empty);
        //        await Console.Out.WriteLineAsync($" OK. [{sessionId}]");
        //    }

        //    {
        //        //var query = HttpUtility.ParseQueryString(String.Empty);
        //        //query["sessionId"] = $"{sessionId}";
        //        //(await httpClient.GetAsync($"{baseUrl}/session/reset?{query}")).EnsureSuccessStatusCode();
        //        //(await httpClient.GetAsync($"{baseUrl}/model/reset")).EnsureSuccessStatusCode();
        //    }

        //    // Generate token(s)
        //    {
        //        var generateOptions = new LlamaCppGenerateOptions { Temperature = 0.0f, Mirostat = Mirostat.Disabled, ThreadCount = 2 };

        //        await Console.Out.WriteLineAsync(
        //            """

        //            Entering interactive mode:
        //                * Press <Ctrl+Z> on an empty line to submit your prompt
        //                * Press <Ctrl+C> to cancel token generation
        //                * Press <Enter> on an empty input prompt to quit
        //            """
        //        );

        //        while (true)
        //        {
        //            try
        //            {
        //                await Console.Out.WriteLineAsync("\nInput:");

        //                var prompt = (await Console.In.ReadToEndAsync()).Replace("\r", "").Trim();
        //                if (String.IsNullOrWhiteSpace(prompt))
        //                    break;

        //                var match = Regex.Match(prompt, @"/load\s""?(.*[^""])?""?");
        //                if (match.Success)
        //                {
        //                    var path = Path.GetFullPath(match.Groups[1].Value);
        //                    prompt = File.ReadAllText(match.Groups[1].Value);
        //                }

        //                await Console.Out.WriteLineAsync("\nOutput:");

        //                var content = new
        //                {
        //                    SessionId = $"{sessionId}",
        //                    GenerateOptions = generateOptions,
        //                    Prompt = prompt,
        //                };

        //                var url = $"{baseUrl}/model/generate";
        //                using var request = new HttpRequestMessage(HttpMethod.Post, url) { Content = new StringContent(JsonSerializer.Serialize(content), Encoding.UTF8, MediaTypeNames.Application.Json) };
        //                request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        //                using var response = (await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationTokenSource.Token)).EnsureSuccessStatusCode();

        //                await using var stream = await response.Content.ReadAsStreamAsync(cancellationTokenSource.Token);
        //                using var reader = new StreamReader(stream);

        //                while (!reader.EndOfStream && !cancellationTokenSource.IsCancellationRequested)
        //                {
        //                    var data = await reader.ReadLineAsync(cancellationTokenSource.Token);
        //                    if (data == null)
        //                        break;

        //                    var decodedToken = Regex.Match(data, @"(?<=data:\s).*").Value.Replace("\\n", "\n").Replace("\\t", "\t");
        //                    await Console.Out.WriteAsync(decodedToken);
        //                }

        //                await Console.Out.WriteLineAsync();
        //                cancellationTokenSource.Token.ThrowIfCancellationRequested();

        //                // Remove this part if you want to keep your context!
        //                var query = HttpUtility.ParseQueryString(String.Empty);
        //                query["sessionId"] = $"{sessionId}";
        //                (await httpClient.GetAsync($"{baseUrl}/session/reset?{query}")).EnsureSuccessStatusCode();
        //                (await httpClient.GetAsync($"{baseUrl}/model/reset")).EnsureSuccessStatusCode();
        //            }
        //            catch (Exception ex) when (ex is TaskCanceledException || ex is OperationCanceledException)
        //            {
        //                await Console.Out.WriteLineAsync(" [Cancelled]");

        //                cancellationTokenSource.Dispose();
        //                cancellationTokenSource = new();
        //            }
        //        }
        //    }

        //    await Task.CompletedTask;
        //}

        //static async Task RunEmbeddingsSampleAsync(string[] args)
        //{
        //    _RunEmbeddingsSampleAsync(args);
        //    await Task.CompletedTask;
        //}

        //static void _RunEmbeddingsSampleAsync(string[] args)
        //{
        //    var path = @"D:\LLM_MODELS\codellama\ggml-codellama-7b-instruct-Q8_0.gguf";

        //    LlamaCppInterop.llama_backend_init();
        //    {
        //        var cparams = LlamaCppInterop.llama_context_default_params();
        //        cparams.n_ctx = 4096;
        //        cparams.n_gpu_layers = 64;
        //        cparams.embedding = true;

        //        var model = LlamaCppInterop.llama_load_model_from_file(path, cparams);
        //        var ctx = LlamaCppInterop.llama_new_context_with_model(model, cparams);

        //        var documents = new[]
        //        {
        //            "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
        //            "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
        //            "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        //            "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        //            "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        //            "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
        //        };

        //        var query = "What is the capital of the United States?";

        //        var documentsEmbeddings = documents
        //            .Select(x => { LlamaCppInterop.llama_tokenize(ctx, $" {x}", out var tokens, true); return tokens.ToArray(); })
        //            .Select(x => LlamaCppInterop.llama_eval(ctx, x, x.Length, 0, 1))
        //            .Select(x => LlamaCppInterop.llama_get_embeddings(ctx).ToArray())
        //            .ToList();

        //        LlamaCppInterop.llama_tokenize(ctx, $" {query}", out var tokens, true);
        //        LlamaCppInterop.llama_eval(ctx, tokens.ToArray(), tokens.Length, 0, 1);
        //        var queryEmbeddings = LlamaCppInterop.llama_get_embeddings(ctx).ToArray();

        //        var cosineSimilarities = documentsEmbeddings
        //            .Select(document => CosineSimilarity(queryEmbeddings, document))
        //            .ToList();

        //        var results = documents
        //            .Zip(cosineSimilarities, (x, similarity) => new { Document = x, CosineSimilarity = similarity })
        //            .OrderByDescending(x => x.CosineSimilarity)
        //            .ToList();

        //        results.ForEach(x => Console.WriteLine($"[{x.CosineSimilarity}][{x.Document}]"));

        //        //LlamaCppInterop.llama_tokenize(ctx, " Hello, World!", out var tokens, true);
        //        //var embd_inp = tokens.ToArray().ToList();

        //        //while (embd_inp.Any())
        //        //{
        //        //    var n_tokens = Math.Min(cparams.n_batch, embd_inp.Count);
        //        //    LlamaCppInterop.llama_eval(ctx, embd_inp.ToArray(), n_tokens, LlamaCppInterop.llama_get_kv_cache_token_count(ctx), 1);
        //        //    embd_inp.RemoveRange(0, n_tokens);
        //        //}

        //        //var embeddings = LlamaCppInterop.llama_get_embeddings(ctx);
        //        //foreach (var embedding in embeddings)
        //        //    Console.Write($"{embedding:F6} ");
        //        //Console.WriteLine();

        //        LlamaCppInterop.llama_print_timings(ctx);

        //        LlamaCppInterop.llama_free(ctx);
        //        LlamaCppInterop.llama_free_model(model);
        //    }
        //    LlamaCppInterop.llama_backend_free();
        //}

        //static async Task RunBertSampleAsync(string[] args)
        //{
        //    //var path = @"D:\LLM_MODELS\sentence-transformers\ggml-all-MiniLM-L12-v2-f32.bin";
        //    //var path = @"D:\LLM_MODELS\intfloat\ggml-e5-large-v2-f16.bin";
        //    var path = @"D:\LLM_MODELS\BAAI\ggml-bge-large-en-f32.bin";

        //    BertCppInterop.bert_params_parse(new[] { "-t", $"8", "-p", "What is the capital of the United States?", "-m", path }, out var bparams);

        //    var documents = new[]
        //    {
        //        "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
        //        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
        //        "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        //        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        //        "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
        //        "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
        //    };

        //    var ctx = BertCppInterop.bert_load_from_file(path);

        //    var documentsEmbeddings = documents
        //        .Select(x => BertCppInterop.bert_tokenize(ctx, x))
        //        .Select(x => BertCppInterop.bert_eval(ctx, bparams.n_threads, x))
        //        .ToList();

        //    var queryEmbeddings = BertCppInterop.bert_eval(ctx, bparams.n_threads, BertCppInterop.bert_tokenize(ctx, bparams.prompt));

        //    var cosineSimilarities = documentsEmbeddings
        //        .Select(document => CosineSimilarity(queryEmbeddings, document))
        //        .ToList();

        //    var results = documents
        //        .Zip(cosineSimilarities, (x, similarity) => new { Document = x, CosineSimilarity = similarity })
        //        .OrderByDescending(x => x.CosineSimilarity)
        //        .ToList();

        //    Console.WriteLine($"[{bparams.prompt}]");
        //    results.ForEach(x => Console.WriteLine($"[{x.CosineSimilarity}][{x.Document}]"));

        //    BertCppInterop.bert_free(ctx);

        //    await Task.CompletedTask;
        //}

        //static float CosineSimilarity(float[] vec1, float[] vec2)
        //{
        //    if (vec1.Length != vec2.Length)
        //        throw new ArgumentException("Vectors must be of the same size.");

        //    var dotProduct = vec1.Zip(vec2, (a, b) => a * b).Sum();
        //    var normA = Math.Sqrt(vec1.Sum(a => Math.Pow(a, 2)));
        //    var normB = Math.Sqrt(vec2.Sum(b => Math.Pow(b, 2)));

        //    if (normA == 0.0 || normB == 0.0)
        //        throw new ArgumentException("Vectors must not be zero vectors.");

        //    return (float)(dotProduct / (normA * normB));
        //}
    }
}
