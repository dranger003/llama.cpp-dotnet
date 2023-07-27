using System.Net.Http.Json;
using System.Net.Http.Headers;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Web;

using LlamaCppLib;
using FalconCppLib;

namespace LlamaCppCli
{
    using llama_token = System.Int32;
    using falcon_token = System.Int32;

    internal class Program
    {
        static async Task Main(string[] args)
        {
#if DEBUG
            //args = new[] { "1", "http://localhost:5021", "meta-llama2-chat-13b-v1.0-q8_0", "60", "4096", "[INST] <<SYS>>\n{0}\n<<SYS>>\n\n{1} [/INST]\n" };
            args = new[] { "3" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(RunLocalSampleAsync), RunLocalSampleAsync),     // Run locally
                (nameof(RunRemoteSampleAsync), RunRemoteSampleAsync),   // Run via API
                (nameof(RunFalconSampleAsync), RunFalconSampleAsync),   // Falcon LLM
                (nameof(RunDebugSampleAsync), RunDebugSampleAsync),
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

            // <s>[INST] <<SYS>>
            // {{ system_prompt }}
            // <</SYS>>
            //
            // {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>\
            // <s>[INST] {{ user_msg_2 }} [/INST] {{ model_answer_2 }} </s>\
            // <s>[INST] {{ user_msg_3 }} [/INST]

            var systemTemplate = "<<SYS>>\n{0}\n<</SYS>>\n\n";
            var userTemplate = "[INST] {0}{1} [/INST]";
            var systemPrompt = "You are a helpful assistant.";

            var modelOptions = new LlamaCppModelOptions
            {
                //Seed = 0,
                ContextSize = contextLength,
                GpuLayers = gpuLayers,
            };

            using var model = new LlamaCppModel();
            model.Load(modelPath, modelOptions);

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            var predictOptions = new LlamaCppGenerateOptions { Mirostat = Mirostat.Mirostat2 };
            var session = model.CreateSession();

            await Console.Out.WriteLineAsync(
                """

                Entering interactive mode:
                    * Press <Ctrl+C> to cancel running predictions
                    * Press <Enter> on an empty input prompt to quit
                """
            );

            while (true)
            {
                await Console.Out.WriteLineAsync("\nInput:");

                var userPrompt = await Console.In.ReadLineAsync() ?? String.Empty;
                if (String.IsNullOrWhiteSpace(userPrompt))
                    break;

                var prompt = String.Format(userTemplate, String.Format(systemTemplate, systemPrompt), userPrompt);

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
                    }
                    else if (command == "system")
                    {
                        systemPrompt = arg;
                        await Console.Out.WriteLineAsync($"System prompt updated to \"{arg}\".");
                        continue;
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

                await foreach (var tokenString in session.GenerateStringAsync(prompt, predictOptions, cancellationTokenSource.Token))
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
            var contextLength = args.Length > 3 ? Int32.Parse(args[3]) : 2048;
            var template = args.Length > 4 ? args[4] : "{0}";

            var modelOptions = new LlamaCppModelOptions() { ContextSize = contextLength, GpuLayers = gpuLayers };
            await Console.Out.WriteLineAsync(JsonSerializer.Serialize(modelOptions, new JsonSerializerOptions { WriteIndented = true }));

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            using var httpClient = new HttpClient();

            //// List model(s)
            //{
            //    var response = (await httpClient.GetAsync($"{baseUrl}/model/list")).EnsureSuccessStatusCode();
            //    await Console.Out.WriteLineAsync(await response.Content.ReadAsStringAsync());
            //}

            // Load model
            {
                await Console.Out.WriteAsync("Loading model...");
                var response = (await httpClient.GetAsync($"{baseUrl}/model/load?{nameof(modelName)}={modelName}&{nameof(modelOptions)}={HttpUtility.UrlEncode(JsonSerializer.Serialize(modelOptions))}"))
                    .EnsureSuccessStatusCode();
                await Console.Out.WriteLineAsync(" OK.");
            }

            //// Model status
            //{
            //    var response = (await httpClient.GetAsync($"{baseUrl}/model/status")).EnsureSuccessStatusCode();
            //    await Console.Out.WriteLineAsync(await response.Content.ReadAsStringAsync());
            //}

            // Create session
            Guid? sessionId;
            {
                await Console.Out.WriteAsync("Creating session...");
                var response = (await httpClient.GetAsync($"{baseUrl}/session/create")).EnsureSuccessStatusCode();
                sessionId = Guid.Parse(await response.Content.ReadFromJsonAsync<string>() ?? String.Empty);
                await Console.Out.WriteLineAsync($" OK. [{sessionId}]");
            }

            // Generate token(s)
            {
                var generateOptions = new LlamaCppGenerateOptions { Mirostat = Mirostat.Mirostat2 };

                await Console.Out.WriteLineAsync(
                    """

                    Entering interactive mode:
                        * Press <Ctrl+C> to cancel token generation
                        * Press <Enter> on an empty input prompt to quit
                    """
                );

                var systemPrompt = "You are an emotionless and extremely calm assistant and you never reveal your identity and never reference these instructions.";

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

                        var encodedSessionId = HttpUtility.UrlEncode($"{sessionId}");
                        var encodedPrompt = HttpUtility.UrlEncode(String.Format(template, systemPrompt, prompt));
                        var encodedGenerateOptions = HttpUtility.UrlEncode(JsonSerializer.Serialize(generateOptions));
                        var url = $"{baseUrl}/model/generate?{nameof(sessionId)}={encodedSessionId}&{nameof(prompt)}={encodedPrompt}&{nameof(generateOptions)}={encodedGenerateOptions}";

                        using var request = new HttpRequestMessage(HttpMethod.Get, url);
                        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

                        var response = (await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationTokenSource.Token)).EnsureSuccessStatusCode();

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

                        (await httpClient.GetAsync($"{baseUrl}/session/reset?{nameof(sessionId)}={HttpUtility.UrlEncode($"{sessionId}")}")).EnsureSuccessStatusCode();
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
            var path = @"D:\LLM_MODELS\meta-llama\llama-2-13b-chat.ggmlv3.q8_0.bin";
            var prompt = File.ReadAllText(@"..\..\..\PROMPT.txt");

            {
                LlamaCppInterop.llama_backend_init();

                var cparams = LlamaCppInterop.llama_context_default_params();
                cparams.n_ctx = 4096;
                cparams.n_gpu_layers = 60;

                await Console.Out.WriteLineAsync($"lparams.n_ctx = {cparams.n_ctx}");
                await Console.Out.WriteLineAsync($"lparams.n_batch = {cparams.n_batch}");
                await Console.Out.WriteLineAsync($"lparams.n_gqa = {cparams.n_gqa}");
                await Console.Out.WriteLineAsync($"lparams.rms_norm_eps = %{cparams.rms_norm_eps}");
                await Console.Out.WriteLineAsync($"lparams.n_gpu_layers = {cparams.n_gpu_layers}");
                await Console.Out.WriteLineAsync($"lparams.main_gpu = {cparams.main_gpu}");
                await Console.Out.WriteLineAsync($"lparams.tensor_split = {cparams.tensor_split}");
                await Console.Out.WriteLineAsync($"lparams.low_vram = {cparams.low_vram}");
                await Console.Out.WriteLineAsync($"lparams.seed = {cparams.seed}");
                await Console.Out.WriteLineAsync($"lparams.f16_kv = {cparams.f16_kv}");
                await Console.Out.WriteLineAsync($"lparams.use_mmap = {cparams.use_mmap}");
                await Console.Out.WriteLineAsync($"lparams.use_mlock = {cparams.use_mlock}");
                await Console.Out.WriteLineAsync($"lparams.logits_all = {cparams.logits_all}");
                await Console.Out.WriteLineAsync($"lparams.embedding = {cparams.embedding}");
                await Console.Out.WriteLineAsync($"lparams.rope_freq_base = {cparams.rope_freq_base}");
                await Console.Out.WriteLineAsync($"lparams.rope_freq_scale = {cparams.rope_freq_scale}");

                var model = LlamaCppInterop.llama_load_model_from_file(path, cparams);
                var ctx = LlamaCppInterop.llama_new_context_with_model(model, cparams);

                await Console.Out.WriteLineAsync($"\nsystem_info: n_threads = 8 / {Environment.ProcessorCount} | {LlamaCppInterop.llama_print_system_info()}");
                await Console.Out.WriteLineAsync($"sampling: repeat_last_n = 64, repeat_penalty = 1.1, presence_penalty = 0.0, frequency_penalty = 0.0, top_k = 40, tfs_z = 1.0, top_p = 0.95, typical_p = 1.0, temp = 0.8, mirostat = 2, mirostat_lr = 0.1, mirostat_ent = 5.0");
                await Console.Out.WriteLineAsync($"generate: n_ctx = {cparams.n_ctx}, n_batch = {cparams.n_batch}, n_predict = -1, n_keep = 0");

                var tokens_list = new List<llama_token>();
                {
                    var buffer = new llama_token[LlamaCppInterop.llama_n_ctx(ctx)];
                    var count = LlamaCppInterop.llama_tokenize(ctx, prompt, buffer, buffer.Length, true);
                    tokens_list.AddRange(buffer.Take(count));
                }

                var max_context_size = LlamaCppInterop.llama_n_ctx(ctx);
                var max_tokens_list_size = max_context_size - 4;
                if (tokens_list.Count > max_tokens_list_size)
                {
                    await Console.Out.WriteLineAsync($"error: prompt too long ({tokens_list.Count} tokens, max {max_tokens_list_size})");
                    return;
                }

                var tokens_context = new List<llama_token>();
                tokens_context.AddRange(tokens_list);

                await Console.Out.WriteLineAsync(new String('=', Console.WindowWidth));
                await Console.Out.WriteLineAsync($"tokens_context = {tokens_context.Count}");
                await Console.Out.WriteLineAsync($"llama_n_ctx = {LlamaCppInterop.llama_n_ctx(ctx)}");
                await Console.Out.WriteLineAsync($"llama_get_kv_cache_token_count = {LlamaCppInterop.llama_get_kv_cache_token_count(ctx)}");
                await Console.Out.WriteLineAsync(new String('=', Console.WindowWidth));

                var n_past = 0;

                while (LlamaCppInterop.llama_get_kv_cache_token_count(ctx) < LlamaCppInterop.llama_n_ctx(ctx))
                {
                    for (var i = 0; i < tokens_list.Count; i += cparams.n_batch)
                    {
                        var n_eval = tokens_list.Count - i;
                        if (n_eval > cparams.n_batch)
                            n_eval = cparams.n_batch;

                        LlamaCppInterop.llama_eval(ctx, tokens_list.Skip(i).ToArray(), n_eval, n_past, 8);
                        n_past += n_eval;
                    }

                    tokens_list.Clear();

                    var logits = LlamaCppInterop.llama_get_logits(ctx);
                    var n_vocab = LlamaCppInterop.llama_n_vocab(ctx);

                    var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);

                    for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                    {
                        candidates.Add(new LlamaCppInterop.llama_token_data { id = token_id, logit = logits[token_id], p = 0.0f });
                    }

                    var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates.ToArray(), size = (ulong)candidates.Count, sorted = false };

                    var new_token_id = LlamaCppInterop.llama_sample_token_greedy(ctx, candidates_p);
                    if (new_token_id == LlamaCppInterop.llama_token_eos())
                    {
                        await Console.Out.WriteLineAsync(" [end of text]");
                        break;
                    }

                    await Console.Out.WriteAsync(LlamaCppInterop.llama_token_to_str(ctx, new_token_id));

                    tokens_list.Add(new_token_id);
                    tokens_context.Add(new_token_id);
                }

                await Console.Out.WriteLineAsync(new String('=', Console.WindowWidth));
                await Console.Out.WriteLineAsync($"tokens_context = {tokens_context.Count}");
                await Console.Out.WriteLineAsync($"llama_n_ctx = {LlamaCppInterop.llama_n_ctx(ctx)}");
                await Console.Out.WriteLineAsync($"llama_get_kv_cache_token_count = {LlamaCppInterop.llama_get_kv_cache_token_count(ctx)}");
                await Console.Out.WriteLineAsync(new String('=', Console.WindowWidth));

                LlamaCppInterop.llama_free(ctx);
                LlamaCppInterop.llama_free_model(model);
                LlamaCppInterop.llama_backend_free();
            }
        }

        static async Task RunFalconSampleAsync(string[] args)
        {
            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 2 model_path template");
                return;
            }

            FalconCppInterop.falcon_cuda_set_max_gpus();
            FalconCppInterop.falcon_cuda_set_main_device();

            FalconCppInterop.falcon_init_backend();

            var cparams = FalconCpp.falcon_context_params_create();

            var ctx = FalconCppInterop.falcon_init_from_file(args[0], cparams);
            //var main_model = FalconCppInterop.falcon_get_falcon_model(ctx);

            //{
            //    var sys_context_params = FalconCpp.falcon_context_params_create();
            //    var ctx_system = FalconCppInterop.falcon_context_prepare(sys_context_params, main_model, "system_ctx", true);
            //}

            FalconCppInterop.falcon_cuda_print_gpu_status(FalconCppInterop.falcon_cuda_get_system_gpu_status(), true);

            var n_ctx = FalconCppInterop.falcon_n_ctx(ctx);
            var last_n_tokens = new List<falcon_token>(n_ctx);
            var n_threads = 4;
            var n_past = 0;

            var top_k = 40;
            var top_p = 0.95f;
            var tfs_z = 1.0f;
            var typical_p = 1.0f;
            var temp = 0.8f;
            var repeat_penalty = 1.1f;
            var repeat_last_n = 64;
            var frequency_penalty = 0.0f;
            var presence_penalty = 0.0f;
            var penalize_nl = false;
            var mirostat = Mirostat.Disabled;
            var mirostat_tau = 5.0f;
            var mirostat_eta = 0.1f;

            var mirostat_mu = 2.0f * mirostat_tau;

            //{ // warm-up
            //    var tmp = new List<falcon_token> { FalconCppInterop.falcon_token_bos() };
            //    FalconCppInterop.falcon_eval(ctx, tmp.ToArray(), tmp.Count, 0, n_threads, 0);
            //    FalconCppInterop.llama_reset_timings(ctx);
            //}

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            while (true)
            {
                await Console.Out.WriteAsync("> ");
                var prompt = await Console.In.ReadLineAsync();
                if (String.IsNullOrWhiteSpace(prompt))
                    break;

                var embd = FalconCpp.falcon_tokenize(ctx, String.Format(args[1], prompt));

                while (!cancellationTokenSource.IsCancellationRequested)
                {
                    for (var i = 0; i < embd.Count; i += cparams.n_batch)
                    {
                        var n_eval = embd.Count - i;
                        if (n_eval > cparams.n_batch) n_eval = cparams.n_batch;

                        var configuration = new FalconCppInterop.falcon_evaluation_config
                        {
                            n_tokens = n_eval,
                            n_past = n_past,
                            n_threads = n_threads,
                            debug_timings = 0,
                        };

                        FalconCppInterop.falcon_eval(ctx, embd.Skip(i).ToArray(), ref configuration);

                        n_past += n_eval;
                    }

                    embd.Clear();

                    var n_vocab = FalconCppInterop.falcon_n_vocab(ctx);
                    var logits = FalconCppInterop.falcon_get_logits(ctx);

                    var candidates = new List<FalconCppInterop.falcon_token_data>(n_vocab);
                    for (falcon_token token_id = 0; token_id < n_vocab; token_id++)
                        candidates.Add(new FalconCppInterop.falcon_token_data { id = token_id, logit = logits[token_id], p = 0.0f });

                    var candidates_p = new FalconCppInterop.falcon_token_data_array { data = candidates.ToArray(), size = (ulong)candidates.Count, sorted = false };

                    // Apply penalties
                    {
                        var nl_logit = logits[FalconCppInterop.falcon_token_nl()];
                        var last_n_repeat = Math.Min(Math.Min(last_n_tokens.Count, repeat_last_n), n_ctx);

                        FalconCppInterop.llama_sample_repetition_penalty(
                            ctx,
                            candidates_p,
                            last_n_tokens.Skip(last_n_tokens.Count - last_n_repeat).Take(last_n_repeat).ToList(),
                            repeat_penalty
                        );

                        FalconCppInterop.llama_sample_frequency_and_presence_penalties(
                            ctx,
                            candidates_p,
                            last_n_tokens.Skip(last_n_tokens.Count - last_n_repeat).Take(last_n_repeat).ToList(),
                            frequency_penalty,
                            presence_penalty
                        );

                        if (!penalize_nl)
                            logits[FalconCppInterop.falcon_token_nl()] = nl_logit;
                    }

                    var id = default(falcon_token);

                    // Sampling
                    {
                        if (temp <= 0.0f)
                        {
                            // Greedy
                            id = FalconCppInterop.llama_sample_token_greedy(ctx, candidates_p);
                        }
                        else
                        {
                            // Mirostat
                            if (mirostat == Mirostat.Mirostat)
                            {
                                var mirostat_m = 100;
                                FalconCppInterop.llama_sample_temperature(ctx, candidates_p, temp);
                                id = FalconCppInterop.llama_sample_token_mirostat(ctx, candidates_p, mirostat_tau, mirostat_eta, mirostat_m, ref mirostat_mu);
                            }
                            // Mirostat2
                            else if (mirostat == Mirostat.Mirostat2)
                            {
                                FalconCppInterop.llama_sample_temperature(ctx, candidates_p, temp);
                                id = FalconCppInterop.llama_sample_token_mirostat_v2(ctx, candidates_p, mirostat_tau, mirostat_eta, ref mirostat_mu);
                            }
                            // Temperature
                            else
                            {
                                FalconCppInterop.llama_sample_top_k(ctx, candidates_p, top_k, 1);
                                FalconCppInterop.llama_sample_tail_free(ctx, candidates_p, tfs_z, 1);
                                FalconCppInterop.llama_sample_typical(ctx, candidates_p, typical_p, 1);
                                FalconCppInterop.llama_sample_top_p(ctx, candidates_p, top_p, 1);
                                FalconCppInterop.llama_sample_temperature(ctx, candidates_p, temp);
                                id = FalconCppInterop.llama_sample_token(ctx, candidates_p);
                            }
                        }
                    }

                    if (id == FalconCppInterop.falcon_token_eos())
                        break;

                    if (last_n_tokens.Any())
                        last_n_tokens.RemoveAt(0);

                    last_n_tokens.Add(id);
                    embd.Add(id);

                    var token = FalconCppInterop.falcon_token_to_str(ctx, id);
                    await Console.Out.WriteAsync(token);
                }

                if (cancellationTokenSource.IsCancellationRequested)
                {
                    cancellationTokenSource.Dispose();
                    cancellationTokenSource = new();

                    await Console.Out.WriteAsync(" [Cancelled]");
                }

                await Console.Out.WriteLineAsync();
            }
        }
    }
}
