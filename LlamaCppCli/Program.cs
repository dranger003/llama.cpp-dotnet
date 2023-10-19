using System.Diagnostics;
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
    using llama_seq_id = System.Int32;

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
            //await RunBatchedSampleAsync(args);
            await RunParallelSampleAsync(args);
        }

        //static string __func__ => new StackTrace().GetFrame(1)?.GetMethod()?.Name ?? String.Empty;
        static string __func__ => "main";

        static async Task RunSimpleSampleAsync(string[] args)
        {
            RunSimpleSample(args);
            await Task.CompletedTask;
        }

        static void RunSimpleSample(string[] args)
        {
            var prompts = new Queue<string>(new[]
            {
                "[INST] List the planets of the solar system in order from the Sun. [/INST]",
                "[INST] List the planets of the solar system in reverse order from the Sun. [/INST]",
            });

            // Initialize Backend
            LlamaCppInterop.llama_backend_init(false);

            // Initialize Model
            var mdl_params = LlamaCppInterop.llama_model_default_params();
            mdl_params.n_gpu_layers = 64;
            var mdl = LlamaCppInterop.llama_load_model_from_file(args[0], mdl_params);

            // Initialize Context
            var ctx_params = LlamaCppInterop.llama_context_default_params();
            ctx_params.seed = 1;
            ctx_params.n_ctx = 16384;
            ctx_params.n_batch = 512;
            ctx_params.n_threads = 1;
            ctx_params.n_threads_batch = 1;
            var ctx = LlamaCppInterop.llama_new_context_with_model(mdl, ctx_params);

            // Initialize Batch
            var n_batch = LlamaCppInterop.llama_n_ctx(ctx);
            var batch = LlamaCppInterop.llama_batch_init(n_batch, 0, 1);

            // Initialize Candidates
            var candidates = new LlamaCppInterop.llama_token_data[LlamaCppInterop.llama_n_vocab(mdl)];
            var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates, size = (nuint)candidates.Length, sorted = false };

            var tokens = new Span<llama_token>();
            var n_cur = 0;

            while (true)
            {
                // UpdateBatch()
                // DecodeBatch()
                // GetLogitsAndUpdateCandidates()
                // SampleNextToken()

                if (batch.n_tokens == 0)
                {
                    if (!prompts.TryDequeue(out var prompt))
                        break;

                    tokens = LlamaCppInterop.llama_tokenize(ctx, prompt);

                    for (var i = 0; i < tokens.Length; i++)
                    {
                        batch.token(n_batch)[i] = tokens[i];
                        batch.pos(n_batch)[i] = n_cur;
                        batch.seq_id(n_batch, 0)[i] = 0;
                        batch.logits(n_batch)[i] = 0;
                    }

                    batch.n_tokens = tokens.Length;
                    batch.logits(n_batch)[batch.n_tokens - 1] = 1;

                    LlamaCppInterop.llama_decode(ctx, batch);
                }

                {
                    var logits = LlamaCppInterop.llama_get_logits_ith(ctx, batch.n_tokens - 1);
                    for (var token_id = 0; token_id < candidates.Length; token_id++)
                        candidates[token_id] = new LlamaCppInterop.llama_token_data { id = token_id, logit = logits[token_id], p = 0.0f };
                }

                {
                    LlamaCppInterop.llama_sample_top_k(ctx, candidates_p, 40, 1);
                    LlamaCppInterop.llama_sample_top_p(ctx, candidates_p, 0.9f, 1);
                    LlamaCppInterop.llama_sample_temp(ctx, candidates_p, 0.0f);
                    var new_token_id = LlamaCppInterop.llama_sample_token(ctx, candidates_p);

                    if (new_token_id == LlamaCppInterop.llama_token_eos(ctx))
                        break;

                    Console.Write(Encoding.ASCII.GetString(LlamaCppInterop.llama_token_to_piece(ctx, new_token_id)));

                    tokens = new[] { new_token_id }.AsSpan();
                }

                ++n_cur;
            }

            // Uninitialize
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
            var tfs_z = 1.0f;
            var typical_p = 1.0f;
            var top_p = 0.95f;
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
            var tokens_list = new List<llama_token>();
            {
                tokens_list.AddRange(new[] { LlamaCppInterop.llama_token_bos(ctx) });
                tokens_list.AddRange(LlamaCppInterop.llama_tokenize(
                    ctx,
                    "[INST] The game is set in the year 2330, in a relatively small pocket of the Milky Way, in an area that extends outward from our solar system for approximately 50 light years. [/INST]\n" +
                    "How can I help you with this information?"
                ).ToArray());
                tokens_list.AddRange(new[] { LlamaCppInterop.llama_token_eos(ctx), LlamaCppInterop.llama_token_bos(ctx) });
                tokens_list.AddRange(LlamaCppInterop.llama_tokenize(
                    ctx,
                    "[INST] What is the setting's time and location? [/INST]\n"
                ).ToArray());
            }

            // Prompt batching
            var batch = LlamaCppInterop.llama_batch_init((int)ctx_params.n_batch, 0, 1);
            batch.n_tokens = tokens_list.Count;
            for (var i = 0; i < tokens_list.Count; i++)
                LlamaCppInterop.llama_batch_add((int)ctx_params.n_batch, ref batch, tokens_list[i], i, new[] { 0 }, false);
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
                    LlamaCppInterop.llama_sample_tail_free(ctx, candidates_p, tfs_z, 1);
                    LlamaCppInterop.llama_sample_typical(ctx, candidates_p, typical_p, 1);
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

                    LlamaCppInterop.llama_batch_add((int)ctx_params.n_batch, ref batch, new_token_id, n_cur, new[] { i }, true);
                }

                if (batch.n_tokens == 0)
                    break;

                ++n_cur;

                LlamaCppInterop.llama_decode(ctx, batch);
            }

            Console.WriteLine($"\n\nElapsed = {sw.Elapsed}\n");

            for (var i = 0; i < n_parallel; ++i)
                Console.WriteLine($"[{i}]:\n{streams[i]}\n");

            LlamaCppInterop.llama_batch_free(batch);
            LlamaCppInterop.llama_free(ctx);
            LlamaCppInterop.llama_free_model(mdl);
            LlamaCppInterop.llama_backend_free();
        }

        private struct Client
        {
            public int id;

            public llama_seq_id seq_id = -1;

            public llama_token sampled;

            public long t_start_prompt;
            public long t_start_gen;

            public int n_prompt;
            public int n_decoded;
            public int i_batch = -1;

            public string input = String.Empty;
            public string prompt = String.Empty;
            public StringBuilder response = new();

            public Client() { }
        }

        static async Task RunParallelSampleAsync(string[] args)
        {
            RunParallelSample(args);
            await Task.CompletedTask;
        }

        static unsafe void RunParallelSample(string[] args)
        {
            // parallel.exe -s 1 -t 1 -tb 1 -n 128 -c 2048 -b 512 -np 9 -ns 9 -ngl 64 -m D:\LLM_MODELS\codellama\ggml-codeLlama-7b-instruct-q8_0.gguf

            var k_system = "";
            var k_prompts = new[]
            {
                "What is the meaning of life?",
                "Tell me an interesting fact about llamas.",
                "What is the best way to cook a steak?",
                "Are you familiar with the Special Theory of Relativity and can you explain it to me?",
                "Recommend some interesting books to read.",
                "What is the best way to learn a new language?",
                "How to get a job at Google?",
                "If you could have any superpower, what would it be?",
                "I want to learn how to play the piano.",
            };

            var n_predict = 128;

            //var rand = new Random(1234);
            var rand = 0;

            // number of simultaneous "clients" to simulate
            var n_clients = args.Length > 1 ? Int32.Parse(args[1]) : 1;

            // requests to simulate
            var n_seq = n_clients;

            // insert new requests as soon as the previous one is done
            var cont_batching = false;

            // insert new requests as soon as the previous one is done
            LlamaCppInterop.llama_backend_init(false);

            // load the target model
            var mdl_params = LlamaCppInterop.llama_model_default_params();
            mdl_params.n_gpu_layers = 64;
            var mdl = LlamaCppInterop.llama_load_model_from_file(args[0], mdl_params);

            var ctx_params = LlamaCppInterop.llama_context_default_params();
            ctx_params.seed = 1;
            ctx_params.n_ctx = 4096;
            ctx_params.n_batch = 512;
            ctx_params.n_threads = 1;
            ctx_params.n_threads_batch = 1;
            ctx_params.logits_all = true;
            var ctx = LlamaCppInterop.llama_new_context_with_model(mdl, ctx_params);

            //{
            //    //warming up the model with an empty run
            //    var tmp = new llama_token[] { LlamaCppInterop.llama_token_bos(ctx), LlamaCppInterop.llama_token_eos(ctx) };
            //    LlamaCppInterop.llama_decode(ctx, LlamaCppInterop.llama_batch_get_one(tmp, Math.Min(tmp.Length, (int)ctx_params.n_batch), 0, 0));
            //    LlamaCppInterop.llama_kv_cache_tokens_rm(ctx, -1, -1);
            //    LlamaCppInterop.llama_reset_timings(ctx);
            //}

            Console.WriteLine($"\n\u001b[32mNo new questions so proceed with build-in defaults.\u001b[0m");
            Console.WriteLine($"\n");

            var n_ctx = LlamaCppInterop.llama_n_ctx(ctx);
            var n_vocab = LlamaCppInterop.llama_n_vocab(mdl);

            var clients = Enumerable
                .Range(0, n_clients)
                .Select(i => new Client() { id = i })
                .ToArray();

            var candidates = new LlamaCppInterop.llama_token_data[n_vocab];
            var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates, size = (nuint)candidates.Length, sorted = false };

            var tokens_system = LlamaCppInterop.llama_tokenize(ctx, k_system, true);
            var n_tokens_system = tokens_system.Length;

            var g_seq_id = 0;

            // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
            // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
            var batch = LlamaCppInterop.llama_batch_init(n_ctx, 0, 1);

            var n_total_prompt = 0;
            var n_total_gen = 0;
            var n_cache_miss = 0;

            var t_main_start = LlamaCppInterop.llama_time_us();

            Console.WriteLine($"{__func__}: Simulating parallel requests from clients:");
            Console.WriteLine($"{__func__}: n_parallel = {n_clients}, n_sequences = {n_seq}, cont_batching = {(!cont_batching ? 0 : 1)}, system tokens = {n_tokens_system}\n");

            {
                Console.WriteLine($"{__func__}: Evaluating the system prompt ...");

                for (var i = 0; i < n_tokens_system; i++)
                    LlamaCppInterop.llama_batch_add(n_ctx, ref batch, tokens_system[i], i, new[] { 0 }, false);

                if (LlamaCppInterop.llama_decode(ctx, batch) != 0)
                {
                    Console.WriteLine($"{__func__}: llama_decode() failed");
                    return;
                }

                // assign the system KV cache to all parallel sequences
                for (var i = 1; i < n_clients; ++i)
                    LlamaCppInterop.llama_kv_cache_seq_cp(ctx, 0, i, 0, n_tokens_system);

                Console.WriteLine();
            }

            Console.WriteLine("Processing requests ...\n");

            var t_main_end = 0L;

            while (true)
            {
                LlamaCppInterop.llama_batch_clear(ref batch);

                // decode any currently ongoing sequences
                for (var ci = 0; ci < clients.Length; ci++)
                {
                    ref var client = ref clients[ci];

                    if (client.seq_id == -1)
                        continue;

                    client.i_batch = batch.n_tokens;

                    LlamaCppInterop.llama_batch_add(n_ctx, ref batch, client.sampled, n_tokens_system + client.n_prompt + client.n_decoded, new[] { client.id }, true);

                    client.n_decoded += 1;
                }

                if (batch.n_tokens == 0)
                {
                    // all sequences have ended - clear the entire KV cache
                    for (int i = 0; i < n_clients; ++i)
                        LlamaCppInterop.llama_kv_cache_seq_rm(ctx, i, n_tokens_system, -1);

                    Console.WriteLine($"{__func__}: clearing the KV cache");
                }

                // insert new sequences for decoding
                if (cont_batching || batch.n_tokens == 0)
                {
                    for (var ci = 0; ci < clients.Length; ci++)
                    {
                        ref var client = ref clients[ci];

                        if (client.seq_id == -1 && g_seq_id < n_seq)
                        {
                            client.seq_id = g_seq_id;

                            client.t_start_prompt = LlamaCppInterop.llama_time_us();
                            client.t_start_gen = 0;

                            //client.input = k_prompts[rand.Next(k_prompts.Length)];
                            client.input = k_prompts[rand++ % k_prompts.Length];
                            client.prompt = $"[INST] {client.input} [/INST]";
                            client.response.Clear();

                            // do not prepend BOS because we have a system prompt!
                            var tokens_prompt = LlamaCppInterop.llama_tokenize(ctx, client.prompt, false);

                            for (var i = 0; i < tokens_prompt.Length; i++)
                                LlamaCppInterop.llama_batch_add(n_ctx, ref batch, tokens_prompt[i], i + n_tokens_system, new[] { client.id }, false);

                            // extract the logits only for the last token
                            if (batch.n_tokens > 0)
                                batch.logits(n_ctx)[batch.n_tokens - 1] = 1; // true

                            client.n_prompt = tokens_prompt.Length;
                            client.n_decoded = 0;
                            client.i_batch = batch.n_tokens - 1;

                            Console.WriteLine($"\u001b[31mClient {client.id,3}, seq {client.seq_id,4}, started decoding ...\u001b[0m");

                            g_seq_id += 1;

                            // insert new requests one-by-one
                            //if (cont_batching) {
                            //    break;
                            //}
                        }
                    }
                }

                if (batch.n_tokens == 0)
                    break;

                // process in chunks of params.n_batch
                var n_batch = (int)ctx_params.n_batch;

                for (var i = 0; i < batch.n_tokens; i += n_batch)
                {
                    var n_tokens = Math.Min(n_batch, batch.n_tokens - i);

                    var batch_view = new LlamaCppInterop.llama_batch(
                        n_tokens,
                        batch._token + i,
                        null,
                        batch._pos + i,
                        batch._n_seq_id + i,
                        batch._seq_id + i,
                        batch._logits + i,
                        0, 0, 0 // unused
                    );

                    var ret = LlamaCppInterop.llama_decode(ctx, batch_view);
                    if (ret != 0)
                    {
                        if (n_batch == 1 || ret < 0)
                        {
                            // if you get here, it means the KV cache is full - try increasing it via the context size
                            Console.WriteLine($"{__func__} : failed to decode the batch, n_batch = {n_batch}, ret = {ret}");
                            return;
                        }

                        Console.WriteLine($"{__func__} : failed to decode the batch, retrying with n_batch = n_batch / 2");

                        n_cache_miss += 1;

                        // retry with half the batch size to try to find a free slot in the KV cache
                        n_batch /= 2;
                        i -= n_batch;

                        continue;
                    }

                    //Console.WriteLine($"{__func__} : decoded batch of {n_tokens} tokens\n");

                    for (var ci = 0; ci < clients.Length; ci++)
                    {
                        ref var client = ref clients[ci];

                        if (client.i_batch < i || client.i_batch >= (i + n_tokens))
                            continue;

                        //Console.WriteLine($"client {client.id}, seq {client.seq_id}, token {client.sampled}, pos {client.n_decoded}, batch {client.i_batch}");

                        //var id = LlamaCppInterop.llama_sampling_sample(ctx, null, ctx_sampling, client.tokens_prev, candidates, client.i_batch - i, client.seq_id);
                        var logits = LlamaCppInterop.llama_get_logits_ith(ctx, client.i_batch - i);
                        for (var token_id = 0; token_id < candidates.Length; token_id++)
                            candidates[token_id] = new LlamaCppInterop.llama_token_data { id = token_id, logit = logits[token_id], p = 0.0f };
                        var id = LlamaCppInterop.llama_sample_token_greedy(ctx, candidates_p);

                        if (client.n_decoded == 1)
                        {
                            // start measuring generation time after the first token to make sure all concurrent clients
                            // have their prompt already processed
                            client.t_start_gen = LlamaCppInterop.llama_time_us();
                        }

                        var token_str = Encoding.ASCII.GetString(LlamaCppInterop.llama_token_to_piece(ctx, id));
                        client.response.Append(token_str);
                        client.sampled = id;

                        //Console.WriteLine($"client {client.id}, seq {client.seq_id}, token {id}, pos {client.n_decoded}, batch {client.i_batch}: {token_str}");

                        if (id == LlamaCppInterop.llama_token_eos(ctx) || (n_predict > 0 && client.n_decoded + client.n_prompt >= n_predict))
                        {
                            // basic reverse prompt
                            //var pos = client.response.IndexOf("User:");
                            //if (pos > 0)
                            //    client.response = client.response.Substring(0, pos);

                            // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                            LlamaCppInterop.llama_kv_cache_seq_rm(ctx, client.id, n_tokens_system, -1);

                            t_main_end = LlamaCppInterop.llama_time_us();

                            Console.WriteLine(
                                $"\u001b[31m"
                                + $"Client {client.id,3}, "
                                + $"seq {client.seq_id,3}/{n_seq,3}, "
                                + $"prompt {client.n_prompt,4} t, "
                                + $"response {client.n_decoded,4} t, "
                                + $"time {(t_main_end - client.t_start_prompt) / 1e6,5:F2} s, "
                                + $"speed {(double)(client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,5:F2} t/s, "
                                + $"cache miss {n_cache_miss} "
                                + $"\u001b[0m\n"
                                + $"Input:    {client.input.Trim()}\n"
                                + $"\u001b[35m"
                                + $"Response: {client.response.ToString().Trim()}"
                                + $"\u001b[0m\n"
                            );

                            n_total_prompt += client.n_prompt;
                            n_total_gen += client.n_decoded;

                            client.seq_id = -1;
                        }

                        client.i_batch = -1;
                    }
                }
            }

            t_main_end = LlamaCppInterop.llama_time_us();

            Console.WriteLine($"\n\u001b[35mrun parameters as at {DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}\u001b[0m");

            Console.WriteLine($"\n{__func__}: n_parallel = {n_clients}, n_sequences = {n_seq}, cont_batching = {(!cont_batching ? 0 : 1)}, system tokens = {n_tokens_system}");

            Console.WriteLine($"External prompt file: \u001b[32m{"used built-in defaults"}\u001b[0m");
            Console.WriteLine($"Model and path used:  \u001b[32m{args[0]}\u001b[0m\n");

            Console.WriteLine($"Total prompt tokens: {n_total_prompt,6}, speed: {(double)n_total_prompt / (t_main_end - t_main_start) * 1e6,5:F2} t/s");
            Console.WriteLine($"Total gen tokens:    {n_total_gen,6}, speed: {(double)n_total_gen / (t_main_end - t_main_start) * 1e6,5:F2} t/s");
            Console.WriteLine($"Total speed (AVG):   {"",6}  speed: {(double)(n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6,5:F2} t/s");
            Console.WriteLine($"Cache misses:        {n_cache_miss,6}");
            Console.WriteLine();

            LlamaCppInterop.llama_print_timings(ctx);
            LlamaCppInterop.llama_batch_free(batch);
            LlamaCppInterop.llama_free(ctx);
            LlamaCppInterop.llama_free_model(mdl);
            LlamaCppInterop.llama_backend_free();

            Console.WriteLine($"\n");
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
