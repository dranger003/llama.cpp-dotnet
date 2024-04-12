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
            var extraStopTokens = new[] { "<|EOT|>", "<|end_of_turn|>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>", "<step>" };
            var assembler = new MultibyteCharAssembler();
            var stream = true;

            //var template = "<s>Source: system\n\n You are very formal and precise. <step> Source: user\n\n {0} <step> Source: assistant\nDestination: user\n\n ";
            //var template = "<s>[INST] {0} [/INST]";
            //var template = "<s>user\n{0}</s>\n<s>assistant\n";
            //var template = "<bos><start_of_turn>user\n{0}<end_of_turn>\n<start_of_turn>model\n";

            var cancel = false;
            var cancel_handle = GCHandle.Alloc(cancel, GCHandleType.Pinned);
            Console.CancelKeyPress += (s, e) => { e.Cancel = cancel = true; };

            var sw = Stopwatch.StartNew();
            var run = 1;
            var tc = 0;

            //================================================================================================================================================================================================

            var top_k = 50;
            var top_p = 0.95f;
            var min_p = 0.05f;
            var tfs_z = 1.0f;
            var typical_p = 1.0f;
            var temp = 0.7f;

            var mirostat = 0;
            var mirostat_tau = 5.0f;
            var mirostat_eta = 0.1f;
            var mirostat_m = 100;

            var penalty_last_n = 64;
            var penalty_repeat = 1.0f;
            var penalty_freq = 0.0f;
            var penalty_present = 0.0f;

            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = args.Length > 1 ? Int32.Parse(args[1]) : 128;
            mparams.progress_callback = &ProgressCallback;

            var cparams = llama_context_default_params();
            cparams.n_ctx = args.Length > 2 ? UInt32.Parse(args[2]) : 0;
            cparams.abort_callback = &AbortCallback;
            cparams.abort_callback_data = GCHandle.ToIntPtr(cancel_handle).ToPointer();
            //cparams.seed = unchecked((uint)-1);
            //cparams.n_batch = 512;
            //cparams.n_threads = 8;
            //cparams.n_threads_batch = 8;
            //cparams.rope_freq_base = 1000000;
            //cparams.type_k = ggml_type.GGML_TYPE_F16;
            //cparams.type_v = ggml_type.GGML_TYPE_F16;
            //cparams.logits_all = false ? 1 : 0;

            llama_backend_init();
            llama_numa_init(ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED);

            var mdl = llama_load_model_from_file(args[0], mparams);
            var ctx = llama_new_context_with_model(mdl, cparams);
            var bat = llama_batch_init((int)llama_n_ctx(ctx), 0, 1);

            var bat_view = new llama_batch();
            var candidates = new llama_token_data[llama_n_vocab(mdl)];

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

                    var prompt = String.Empty;
                    var match = Regex.Match(line, @"\/load\s+("".*?""(?:\s+|$))");
                    if (match.Success)
                    {
                        var fileName = Path.GetFullPath(Regex.Match(line, "\"(.*?)\"").Groups[1].Value);
                        prompt = File.ReadAllText(fileName);
                    }
                    else
                    {
                        //prompt = String.IsNullOrWhiteSpace(line) ? File.ReadAllText(args[1]).Replace("\r\n", "\n") : line;
                        prompt = line.Replace("\\n", "\n");
                    }

                    //if (args.Length > 4 && Int32.Parse(args[4]) > 0)
                    //    prompt = String.Format(template, prompt);

                    if (String.IsNullOrWhiteSpace(prompt))
                        continue;

                    //var add_bos = llama_add_bos_token(mdl) > 0;
                    //if (!add_bos) add_bos = llama_vocab_type(mdl) == llama_vocab_type_t.LLAMA_VOCAB_TYPE_SPM;

                    //Console.WriteLine($"[{prompt}]");
                    var tokens = llama_tokenize(mdl, prompt, false, true, false);

                    Console.WriteLine($"{tokens.Length}/{llama_n_ctx(ctx)} token(s)");
                    if (tokens.Length >= llama_n_ctx(ctx))
                    {
                        Console.WriteLine("Out of context.");
                        continue;
                    }

                    //// Debug
                    //Console.WriteLine(tokens.ToArray().Select(x => $"[{x}:{Encoding.UTF8.GetString(llama_token_to_piece(mdl, x))}]").Aggregate((a, b) => $"{a}{b}"));

                    requests.Add(new Request((int)llama_n_ctx(ctx), tokens));

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

                        bat_view.n_tokens = n_tokens;
                        bat_view.token = bat.token + i;
                        bat_view.embd = null;
                        bat_view.pos = bat.pos + i;
                        bat_view.n_seq_id = bat.n_seq_id + i;
                        bat_view.seq_id = bat.seq_id + i;
                        bat_view.logits = bat.logits + i;
                        bat_view.all_pos_0 = 0;
                        bat_view.all_pos_1 = 0;
                        bat_view.all_seq_id = 0;

                        var res = llama_decode(ctx, bat_view);
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

                            var logits = llama_get_logits_ith(ctx, request.PosLogit - i);

                            for (var token = 0; token < candidates.Length; token++)
                            {
                                candidates[token].id = token;
                                candidates[token].logit = logits[token];
                                candidates[token].p = 0.0f;
                            }

                            fixed (llama_token_data* ptr1 = &candidates[0])
                            {
                                var candidates_p = new llama_token_data_array
                                {
                                    data = ptr1,
                                    size = (nuint)candidates.Length,
                                    sorted = false ? 1 : 0,
                                };

                                if (penalty_repeat != 1.0f)
                                {
                                    var index = Math.Max(0, request.PosToken - penalty_last_n);
                                    llama_sample_repetition_penalties(
                                        ctx,
                                        ref candidates_p,
                                        new Span<int>(request.Tokens, index, request.Tokens.Length - index),
                                        (nuint)penalty_last_n,
                                        penalty_repeat,
                                        penalty_freq,
                                        penalty_present
                                    );
                                }

                                var token = llama_token_eos(mdl);

                                if (temp < 0.0f)
                                {
                                    llama_sample_softmax(ctx, ref candidates_p);
                                    token = candidates_p.data[0].id;
                                }
                                else if (temp == 0.0f)
                                {
                                    token = llama_sample_token_greedy(ctx, ref candidates_p);
                                }
                                else if (mirostat == 1)
                                {
                                    llama_sample_temp(ctx, ref candidates_p, temp);
                                    token = llama_sample_token_mirostat(ctx, ref candidates_p, mirostat_tau, mirostat_eta, mirostat_m, ref request.MirostatMU);
                                }
                                else if (mirostat == 2)
                                {
                                    llama_sample_temp(ctx, ref candidates_p, temp);
                                    token = llama_sample_token_mirostat_v2(ctx, ref candidates_p, mirostat_tau, mirostat_eta, ref request.MirostatMU);
                                }
                                else
                                {
                                    llama_sample_top_k(ctx, ref candidates_p, top_k, 1);
                                    llama_sample_tail_free(ctx, ref candidates_p, tfs_z, 1);
                                    llama_sample_typical(ctx, ref candidates_p, typical_p, 1);
                                    llama_sample_top_p(ctx, ref candidates_p, top_p, 1);
                                    llama_sample_min_p(ctx, ref candidates_p, min_p, 1);
                                    llama_sample_temp(ctx, ref candidates_p, temp);
                                    token = llama_sample_token(ctx, ref candidates_p);
                                }

                                if (request.PosResponse == 0)
                                    request.PosResponse = request.PosToken;

                                var stop = extraStopTokens
                                    .Select(extraStopToken => llama_tokenize(mdl, extraStopToken, false, true).ToArray())
                                    .Where(tokens => tokens.Length == 1)
                                    .Select(tokens => tokens[0])
                                    .Contains(token);

                                if (cancel || stop)
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

                                    if (stream)
                                    {
                                        var tokenText = assembler.Consume(llama_token_to_piece(mdl, token));
                                        Console.Write(tokenText);

                                        // Debug
                                        //Console.Write($"[{token}:{tokenText}]");

                                        if (cancel)
                                            Console.Write(" [Cancelled]");
                                    }
                                }

                                if (request.T1 == default)
                                    request.T1 = DateTime.Now;

                                if (token == llama_token_eos(mdl))
                                    request.T2 = DateTime.Now;
                            }
                        }
                    }

                    foreach (var r in requests.Where(r => r.Tokens[r.PosToken - 1] == llama_token_eos(mdl)))
                    {
                        llama_kv_cache_seq_rm(ctx, r.Id, 0, -1);

                        if (!stream)
                        {
                            var promptTokens = r.Tokens.Take(r.PosResponse).SelectMany(token => llama_token_to_piece(mdl, token).ToArray()).ToArray();
                            var responseTokens = r.Tokens.Skip(r.PosResponse).Take(r.PosToken - r.PosResponse).SelectMany(token => llama_token_to_piece(mdl, token).ToArray()).ToArray();

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

                    requests.RemoveAll(r => r.Tokens[r.PosToken - 1] == llama_token_eos(mdl));
                }
            }

            cancel_handle.Free();

            llama_batch_free(bat);
            llama_free(ctx);
            llama_free_model(mdl);

            llama_backend_free();
        }
    }

    file class Request : IEquatable<Request>
    {
        public int Id { get; set; }

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
