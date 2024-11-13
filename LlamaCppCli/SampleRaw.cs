using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

using LlamaCppLib;

using static LlamaCppLib.Native;
using static LlamaCppLib.Interop;

namespace LlamaCppCli
{
    internal partial class Program
    {
        static async Task RunSampleRawAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunSampleRawAsync <ModelPath> [GpuLayers] [CtxLength]");
                return;
            }

            RunSampleRaw(args);
            await Task.CompletedTask;
        }

        [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
        static unsafe sbyte ProgressCallback(float progress, void* state)
        {
            Console.Write($"{new string(' ', 32)}\rLoading model... {(byte)(progress * 100)}%\r");
            return true ? 1 : 0;
        }

        [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
        static unsafe sbyte AbortCallback(void* state)
        {
            var cancel = (bool?)GCHandle.FromIntPtr(new(state)).Target ?? false;
            return (sbyte)(cancel ? 1 : 0);
        }

        static unsafe void RunSampleRaw(string[] args)
        {
            var requests = new List<Request>();
            var assembler = new MultibyteCharAssembler();
            var stream = true;

            var cancel = false;
            var cancel_handle = GCHandle.Alloc(cancel, GCHandleType.Pinned);
            Console.CancelKeyPress += (s, e) => { e.Cancel = cancel = true; };

            var sw = Stopwatch.StartNew();
            var run = 1;
            var tc = 0;

            //================================================================================================================================================================================================

            var seed = unchecked((uint)-1);

            var top_k = 50;
            var top_p = 0.95f;
            var min_p = 0.05f;
            var typical_p = 1.0f;
            var temp = 0.7f;

            //var mirostat = 0;
            //var mirostat_tau = 5.0f;
            //var mirostat_eta = 0.1f;
            //var mirostat_m = 100;

            //var penalty_last_n = 64;
            //var penalty_repeat = 1.0f;
            //var penalty_freq = 0.0f;
            //var penalty_present = 0.0f;

            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = args.Length > 1 ? Int32.Parse(args[1]) : 0;
            mparams.progress_callback = &ProgressCallback;

            var cparams = llama_context_default_params();
            cparams.n_ctx = args.Length > 2 ? UInt32.Parse(args[2]) : 0;
            cparams.flash_attn = true ? 1 : 0;
            cparams.abort_callback = &AbortCallback;
            cparams.abort_callback_data = GCHandle.ToIntPtr(cancel_handle).ToPointer();
            //cparams.n_batch = 512;
            //cparams.n_threads = 8;
            //cparams.n_threads_batch = 8;
            //cparams.rope_freq_base = 8000000;
            //cparams.type_k = ggml_type.GGML_TYPE_F16;
            //cparams.type_v = ggml_type.GGML_TYPE_F16;
            //cparams.logits_all = false ? 1 : 0;

            var sparams = llama_sampler_chain_default_params();
            sparams.no_perf = 0;

            llama_backend_init();
            llama_numa_init(ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED);

            var mdl = llama_load_model_from_file(args[0], mparams);
            var ctx = llama_new_context_with_model(mdl, cparams);
            var bat = llama_batch_init((int)llama_n_ctx(ctx), 0, 1);
            var spl = llama_sampler_chain_init(sparams);

            if (temp > 0.0f)
            {
                llama_sampler_chain_add(spl, llama_sampler_init_top_k(top_k));
                llama_sampler_chain_add(spl, llama_sampler_init_typical(typical_p, 1));
                llama_sampler_chain_add(spl, llama_sampler_init_top_p(top_p, 1));
                llama_sampler_chain_add(spl, llama_sampler_init_min_p(min_p, 1));
                llama_sampler_chain_add(spl, llama_sampler_init_temp(temp));
                llama_sampler_chain_add(spl, llama_sampler_init_dist(seed));
            }
            else
            {
                llama_sampler_chain_add(spl, llama_sampler_init_greedy());
            }

            var messages = new List<LlmMessage> { new() { Role = "system", Content = "You are a helpful assistant." } };

            while (true)
            {
                {
                    cancel = false;

                    Console.Write($"\n[{sw.Elapsed}]{(run++ == 1 ? "" : $"[{tc / sw.Elapsed.TotalSeconds:F2} t/s]")}> ");
                    var line = Console.ReadLine() ?? String.Empty;

                    if (line == "q" || line == "quit")
                    {
                        Console.WriteLine("Bye.");
                        break;
                    }
                    else if (line == "/clear")
                    {
                        messages = new(messages.Take(1));
                        continue;
                    }

                    var prompt = String.Empty;
                    var match = Regex.Match(line, @"\/load\s+("".*?""(?:\s+|$))");
                    if (match.Success)
                    {
                        var fileName = Path.GetFullPath(Regex.Match(line, "\"(.*?)\"").Groups[1].Value);
                        prompt = File.ReadAllText(fileName);
                    }
                    else
                    {
                        prompt = line.Replace("\\n", "\n");
                    }

                    if (String.IsNullOrWhiteSpace(prompt))
                        continue;

                    messages.Add(new() { Role = "user", Content = prompt });
                    prompt = llama_apply_template(ctx, messages);

                    var tokens = llama_tokenize(mdl, Encoding.UTF8.GetBytes(prompt), llama_add_bos_token(mdl) != 0, true);

                    var responseMargin = 512;
                    Console.WriteLine($"{tokens.Length}/{llama_n_ctx(ctx)} token(s)");
                    if (tokens.Length >= llama_n_ctx(ctx) - responseMargin)
                    {
                        Console.WriteLine($"Out of context (with response margin of {responseMargin}.");
                        continue;
                    }

                    requests.Add(new Request((int)llama_n_ctx(ctx), tokens) { Messages = messages });

                    sw.Restart();
                    tc = 0;
                }

                //============================================================================================================================================================================================

                while (true)
                {
                    llama_batch_clear(ref bat);

                    foreach (var request in requests)
                    {
                        for (; request.PosBatch < request.PosToken; request.PosBatch++)
                            llama_batch_add(ref bat, request.Tokens[request.PosBatch], request.PosBatch, [request.Id], false);

                        request.PosLogit = bat.n_tokens - 1;
                        bat.logits[request.PosLogit] = true ? 1 : 0;

                        if (request.T0 == default)
                            request.T0 = DateTime.Now;
                    }

                    if (bat.n_tokens == 0)
                        break;

                    var n_batch = (int)cparams.n_batch;
                    for (var i = 0; i < bat.n_tokens; i += n_batch)
                    {
                        var n_tokens = Math.Min(n_batch, bat.n_tokens - i);

                        var res = llama_decode(
                            ctx,
                            new llama_batch
                            {
                                n_tokens = n_tokens,
                                token = &bat.token[i],
                                embd = null,
                                pos = &bat.pos[i],
                                n_seq_id = &bat.n_seq_id[i],
                                seq_id = &bat.seq_id[i],
                                logits = &bat.logits[i],
                                all_pos_0 = 0,
                                all_pos_1 = 0,
                                all_seq_id = 0,
                            }
                        );

                        if (res != 0)
                        {
                            Console.WriteLine($"llama_decode() = {res}");
                            return;
                        }

                        foreach (var request in requests)
                        {
                            if (stream && n_tokens > 1)
                            {
                                var count = n_tokens + i;
                                var progress = count / (double)bat.n_tokens * 100;
                                var elapsed = DateTime.Now - (request.T0 ?? DateTime.Now);
                                var speed = count / elapsed.TotalSeconds;
                                var remaining = TimeSpan.FromSeconds((bat.n_tokens - count) / speed);
                                Console.Write($"{new String(' ', 32)}\rDecoding... {progress:F2}% [C:{count}/{bat.n_tokens}][S:{speed:F2} t/s][E:{elapsed:hh\\:mm\\:ss\\.fff}][R:{remaining:hh\\:mm\\:ss\\.fff}]\r");
                                if (count == bat.n_tokens) Console.WriteLine();
                            }

                            if (request.PosLogit < i || request.PosLogit >= i + n_tokens)
                                continue;

                            var token = llama_sampler_sample(spl, ctx, request.PosLogit - i);

                            if (request.PosResponse == 0)
                                request.PosResponse = request.PosToken;

                            if (cancel)
                                token = llama_token_eos(mdl); // Override stop token with EOS token

                            if (request.PosToken >= request.Tokens.Length)
                            {
                                if (stream)
                                    Console.Write(" [Out of context]");

                                request.Tokens[request.Tokens.Length - 1] = llama_token_eos(mdl);
                                break;
                            }
                            else
                            {
                                request.Tokens[request.PosToken++] = token;
                                ++tc;

                                var tokenText = assembler.Consume(
                                    //llama_detokenize(mdl, [token])
                                    Interop.llama_token_to_piece(mdl, token, true)
                                );

                                if (request.Messages.Last().Role != "assistant")
                                {
                                    request.Messages.Add(new() { Role = "assistant" });
                                }

                                if (!llama_token_is_eog(mdl, token))
                                {
                                    request.Messages.Last().Content += tokenText;
                                }

                                if (stream)
                                {
                                    if (!llama_token_is_eog(mdl, token))
                                    {
                                        Console.Write(tokenText);
                                    }

                                    if (cancel)
                                        Console.Write(" [Cancelled]");
                                }
                            }

                            if (request.T1 == default)
                                request.T1 = DateTime.Now;

                            if (llama_token_is_eog(mdl, token))
                                request.T2 = DateTime.Now;
                        }
                    }

                    foreach (var r in requests.Where(r => llama_token_is_eog(mdl, r.Tokens[r.PosToken - 1])))
                    {
                        llama_kv_cache_seq_rm(ctx, r.Id, 0, -1);

                        if (!stream)
                        {
                            var promptTokens = r.Tokens.Take(r.PosResponse).SelectMany(token => llama_detokenize(mdl, [token])).ToArray();
                            var responseTokens = r.Tokens.Skip(r.PosResponse).Take(r.PosToken - r.PosResponse).SelectMany(token => llama_detokenize(mdl, [token], false, true).ToArray()).ToArray();

                            Console.WriteLine(new String('=', 128));
                            Console.WriteLine($"request id {r.Id} [{r.PosToken / r.Elapsed.TotalMilliseconds * 1000:F2} t/s]");
                            Console.WriteLine(new String('-', 128));
                            Console.WriteLine(Encoding.UTF8.GetString(promptTokens));
                            Console.WriteLine(new String('-', 128));
                            Console.WriteLine(Encoding.UTF8.GetString(responseTokens));
                            Console.WriteLine(new String('=', 128));
                        }
                        else
                        {
                            Console.WriteLine();
                        }
                    }

                    requests.RemoveAll(r => llama_token_is_eog(mdl, r.Tokens[r.PosToken - 1]));
                }
            }

            cancel_handle.Free();

            llama_sampler_free(spl);
            llama_batch_free(bat);
            llama_free(ctx);
            llama_free_model(mdl);

            llama_backend_free();
        }

        static async Task RunSampleStateRawAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunSampleStateRawAsync <ModelPath>");
                return;
            }

            RunSampleStateRaw(args);
            await Task.CompletedTask;
        }

        static void RunSampleStateRaw(string[] args)
        {
            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = 999;

            var cparams = llama_context_default_params();
            cparams.n_ctx = 4096;
            cparams.flash_attn = true ? 1 : 0;

            var sparams = llama_sampler_chain_default_params();
            sparams.no_perf = true ? 1 : 0;

            var mdl = llama_load_model_from_file(args[0], mparams);

            var seed = 42;

            { // First run
                var ctx = llama_new_context_with_model(mdl, cparams);
                var bat = llama_batch_init((int)llama_n_ctx(ctx), 0, 1);
                var spl = llama_sampler_chain_init(sparams);
                llama_sampler_chain_add(spl, llama_sampler_init_dist((uint)seed));

                Console.Write(new String('=', Console.WindowWidth));

                var messages = new List<LlmMessage>
                {
                    new() { Role = "system", Content = "You are a helpful assistant." },
                    new() { Role = "user", Content = "Hello?" }
                };

                var prompt = llama_apply_template(ctx, messages);
                var tokens = llama_tokenize(mdl, Encoding.UTF8.GetBytes(prompt), llama_add_bos_token(mdl) != 0, true);

                var n_past = 0;
                var run = 0;

                while (true)
                {
                    llama_batch_clear(ref bat);

                    for (var i = 0; i < tokens.Length; i++)
                        llama_batch_add(ref bat, tokens[i], n_past++, [0], i == tokens.Length - 1);

                    llama_decode(ctx, bat);

                    if (++run == 1)
                    {
                        var state_mem = new byte[(int)llama_state_get_size(ctx)];
                        var written = (int)llama_state_get_data(ctx, state_mem, (nuint)state_mem.Length);
                        File.WriteAllBytes($"dump_state_{n_past}.bin", new Span<byte>(state_mem, 0, written).ToArray());
                    }

                    tokens = [llama_sampler_sample(spl, ctx, -1)];
                    if (tokens[0] == llama_token_eos(mdl))
                        break;

                    var piece = llama_token_to_piece(mdl, tokens[0], true);
                    Console.Write(Encoding.UTF8.GetString(piece));
                }

                llama_sampler_free(spl);
                llama_batch_free(bat);
                llama_free(ctx);

                Console.Write($"\n{new String('=', Console.WindowWidth)}");
            }

            { // Second run
                var ctx = llama_new_context_with_model(mdl, cparams);
                var bat = llama_batch_init((int)llama_n_ctx(ctx), 0, 1);
                var spl = llama_sampler_chain_init(sparams);
                llama_sampler_chain_add(spl, llama_sampler_init_dist((uint)seed));

                Console.Write(new String('=', Console.WindowWidth));

                var n_past = 0;
                {
                    var path = Directory.EnumerateFiles(".", "dump_state_*.bin").Single();
                    n_past = Int32.Parse(Regex.Match(path, @"dump_state_(?<n_past>\d+)\.bin").Groups["n_past"].Value);
                    var state_mem = File.ReadAllBytes(path) ?? [];
                    llama_state_set_data(ctx, state_mem, (nuint)state_mem.Length);
                }

                var tokens = new int[0];

                while (true)
                {
                    tokens = [llama_sampler_sample(spl, ctx, -1)];
                    if (tokens[0] == llama_token_eos(mdl))
                        break;

                    var piece = llama_token_to_piece(mdl, tokens[0], true);
                    Console.Write(Encoding.UTF8.GetString(piece));

                    llama_batch_clear(ref bat);
                    llama_batch_add(ref bat, tokens[0], n_past++, [0], true);

                    llama_decode(ctx, bat);
                }

                llama_sampler_free(spl);
                llama_batch_free(bat);
                llama_free(ctx);

                Console.Write($"\n{new String('=', Console.WindowWidth)}");
            }

            llama_free_model(mdl);
        }
    }

    file class Request : IEquatable<Request>
    {
        public int Id { get; set; }
        public List<LlmMessage> Messages { get; set; } = [];

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosResponse { get; set; }
        public int PosToken { get; set; }
        public int[] Tokens { get; set; }

        public float MirostatMU = 0.0f;

        public DateTime? T0 { get; set; }   // Decoding
        public DateTime? T1 { get; set; }   // Sampling
        public DateTime? T2 { get; set; }   // End
        public TimeSpan Elapsed => (T2 - T1) ?? TimeSpan.FromSeconds(0);

        public Request(int n_ctx, ReadOnlySpan<int> tokens)
        {
            this.Tokens = new int[n_ctx];
            tokens.CopyTo(Tokens);
            this.PosToken += tokens.Length;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is Request request && Equals(request);
        public override int GetHashCode() => Id.GetHashCode();

        // IEquatable<T>
        public bool Equals(Request? other) => other?.Id == this.Id;
    }
}
