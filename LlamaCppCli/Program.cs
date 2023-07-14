using System.Net.Http.Headers;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Web;

using LlamaCppLib;
using FalconCppLib;

namespace LlamaCppCli
{
    using falcon_token = System.Int32;

    internal class Program
    {
        static async Task Main(string[] args)
        {
#if DEBUG
            //args = new[] { "0", @"C:\LLM_MODELS\WizardLM\ggml-wizardlm-v1.1-13b-q8_0.bin", "60", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n{0}\n\nASSISTANT:\n" };
            //args = new[] { "0", @"C:\LLM_MODELS\WizardLM\wizardlm-30b.ggmlv3.q4_K_M.bin", "60", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n{0}\n\nASSISTANT:\n" };
            //args = new[] { "0", @"C:\LLM_MODELS\psmathur\ggml-orca-mini-v2-13b-q8_0.bin", "60", "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:\n{0}\n\n### Response:\n" };
            //args = new[] { "1", "http://localhost:5021", "wizardlm-v1.1-13b-q8_0", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n{0}\n\nASSISTANT:\n" };
            args = new[] { "2" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(LocalSample), LocalSample),     // Run locally
                (nameof(RemoteSample), RemoteSample),   // Run via API
                (nameof(FalconSample), FalconSample),   // Falcon LLM
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

            await samples[sampleName].Func(args.Skip(1).ToArray());
        }

        static async Task LocalSample(string[] args)
        {
            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 0 model_path [gpu_layers] [template]");
                return;
            }

            var modelPath = args[0];
            var gpuLayers = args.Length > 1 ? Int32.Parse(args[1]) : 0;
            var template = args.Length > 2 ? args[2] : "{0}";

            var modelOptions = new LlamaCppModelOptions
            {
                Seed = 0,
                ContextSize = 2048,
                GpuLayers = gpuLayers,
            };

            using var model = new LlamaCpp();
            model.Load(modelPath, modelOptions);

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            var predictOptions = new LlamaCppPredictOptions
            {
                //ThreadCount = 4,
                //TopK = 40,
                //TopP = 0.95f,
                //Temperature = 0.0f,
                //RepeatPenalty = 1.1f,
                //PenalizeNewLine = false,
                Mirostat = Mirostat.Mirostat2,
                //MirostatTAU = 5.0f,
                //MirostatETA = 0.1f,
                ResetState = true, // No context
                Template = template,
            };

            //await Console.Out.WriteLineAsync($"\n{JsonSerializer.Serialize(modelOptions, new JsonSerializerOptions { WriteIndented = true })}");
            //await Console.Out.WriteLineAsync($"\n{JsonSerializer.Serialize(predictOptions, new JsonSerializerOptions { WriteIndented = true })}");

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

                predictOptions.Prompt = await Console.In.ReadLineAsync() ?? String.Empty;
                if (String.IsNullOrWhiteSpace(predictOptions.Prompt))
                    break;

                await Console.Out.WriteLineAsync("\nOutput:");

                await foreach (var prediction in model.Predict(predictOptions, cancellationTokenSource.Token))
                {
                    var token = prediction.Value;
                    await Console.Out.WriteAsync(token);
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

        static async Task RemoteSample(string[] args)
        {
            if (args.Length < 2)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} 1 base_url model_name [gpu_layers] [template]");
                return;
            }

            var baseUrl = args[0];
            var modelName = args[1];
            var gpuLayers = args.Length > 2 ? Int32.Parse(args[2]) : 0;
            var template = args.Length > 3 ? args[3] : "{0}";

            var modelOptions = new LlamaCppModelOptions() { ContextSize = 2048, GpuLayers = gpuLayers };

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            using var httpClient = new HttpClient();

            //// List model(s)
            //{
            //    var response = await httpClient.GetAsync($"{baseUrl}/model/list");
            //    response.EnsureSuccessStatusCode();
            //    await Console.Out.WriteLineAsync(await response.Content.ReadAsStringAsync());
            //}

            // Load model
            {
                await Console.Out.WriteAsync("Loading model...");
                var response = await httpClient.GetAsync($"{baseUrl}/model/load?{nameof(modelName)}={modelName}&{nameof(modelOptions)}={HttpUtility.UrlEncode(JsonSerializer.Serialize(modelOptions))}");
                response.EnsureSuccessStatusCode();
                await Console.Out.WriteLineAsync(" OK.");
            }

            //// Model status
            //{
            //    var response = await httpClient.GetAsync($"{baseUrl}/model/status");
            //    response.EnsureSuccessStatusCode();
            //    await Console.Out.WriteLineAsync(await response.Content.ReadAsStringAsync());
            //}

            // Run prediction(s)
            {
                var predictOptions = new LlamaCppPredictOptions
                {
                    //ThreadCount = 4,
                    //TopK = 40,
                    //TopP = 0.95f,
                    //Temperature = 0.0f,
                    //RepeatPenalty = 1.1f,
                    //PenalizeNewLine = false,
                    Mirostat = Mirostat.Mirostat2,
                    //MirostatTAU = 5.0f,
                    //MirostatETA = 0.1f,
                    ResetState = true, // No context
                    Template = template,
                };

                await Console.Out.WriteLineAsync(
                    """

                    Entering interactive mode:
                        * Press <Ctrl+C> to cancel running predictions
                        * Press <Enter> on an empty input prompt to quit
                    """
                );

                while (true)
                {
                    try
                    {
                        await Console.Out.WriteLineAsync("\nInput:");

                        predictOptions.Prompt = await Console.In.ReadLineAsync(cancellationTokenSource.Token) ?? String.Empty;
                        if (String.IsNullOrWhiteSpace(predictOptions.Prompt))
                            break;

                        await Console.Out.WriteLineAsync("\nOutput:");

                        using var request = new HttpRequestMessage(HttpMethod.Get, $"{baseUrl}/model/predict?{nameof(predictOptions)}={HttpUtility.UrlEncode(JsonSerializer.Serialize(predictOptions))}");
                        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

                        var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationTokenSource.Token);
                        response.EnsureSuccessStatusCode();

                        await using var stream = await response.Content.ReadAsStreamAsync(cancellationTokenSource.Token);
                        using var reader = new StreamReader(stream);

                        while (!reader.EndOfStream && !cancellationTokenSource.IsCancellationRequested)
                        {
                            var data = await reader.ReadLineAsync(cancellationTokenSource.Token);
                            if (data == null)
                                break;

                            var token = Regex.Match(data, @"(?<=data:\s).*").Value.Replace("\\n", "\n");
                            await Console.Out.WriteAsync(token);
                        }

                        cancellationTokenSource.Token.ThrowIfCancellationRequested();

                        await Console.Out.WriteLineAsync();
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

        static async Task FalconSample(string[] args)
        {
            FalconCppInterop.falcon_cuda_set_max_gpus();
            FalconCppInterop.falcon_cuda_set_main_device();

            FalconCppInterop.falcon_init_backend();

            var cparams = FalconCpp.falcon_context_params_create();

            var ctx = FalconCppInterop.falcon_init_from_file(@"C:\LLM_MODELS\tiiuae\falcon-7b-instruct.ggccv1.q8_0.bin", cparams);
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

            var embd = FalconCpp.falcon_tokenize(ctx, "User: Hello?\nAssistant:");

            //{ // warm-up
            //    var tmp = new List<falcon_token> { FalconCppInterop.falcon_token_bos() };
            //    FalconCppInterop.falcon_eval(ctx, tmp.ToArray(), tmp.Count, 0, n_threads, 0);
            //    FalconCppInterop.llama_reset_timings(ctx);
            //}

            var mirostat_mu = 2.0f * mirostat_tau;

            while (true)
            {
                for (var i = 0; i < embd.Count; i += cparams.n_batch)
                {
                    var n_eval = embd.Count - i;
                    if (n_eval > cparams.n_batch) n_eval = cparams.n_batch;
                    FalconCppInterop.falcon_eval(ctx, embd.Skip(i).ToArray(), n_eval, n_past, n_threads, 0);
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
        }
    }
}
