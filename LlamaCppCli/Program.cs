using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

using LlamaCppLib;

namespace LlamaCppCli
{
    using llama_token = int;

    internal class Program
    {
        static async Task Main(string[] args)
        {
            args = new[]
            {
                "D:\\LLM_MODELS\\teknium\\ggml-openhermes-2-mistral-7b-q8_0.gguf",
                //"D:\\LLM_MODELS\\codellama\\ggml-codeLlama-34b-instruct-q4_k.gguf",
            };

            await RunSampleAsync(args);
        }

        static async Task RunSampleAsync(string[] args)
        {
            RunSample(args);
            await Task.CompletedTask;
        }

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        static unsafe void ProgressCallback(float progress, void* state) => Console.Write($"{new string(' ', 32)}\rLoading model... {(byte)(progress * 100)}%\r");

        static unsafe void RunSample(string[] args)
        {
            var top_k = 40;
            var top_p = 0.95f;
            var tfs_z = 1.0f;
            var typical_p = 1.0f;
            var temp = 0.0f;

            var mirostat = 0;
            var mirostat_tau = 5.0f;
            var mirostat_eta = 0.1f;

            var penalty_last_n = 64;
            var penalty_repeat = 1.1f;
            var penalty_freq = 0.0f;
            var penalty_present = 0.0f;

            var mparams = PInvoke.llama_model_default_params();
            mparams.n_gpu_layers = 64;
            mparams.progress_callback = &ProgressCallback;

            var cparams = PInvoke.llama_context_default_params();
            cparams.seed = 0;
            cparams.n_ctx = 0;
            cparams.n_batch = 64;
            cparams.n_threads = 1;
            cparams.n_threads_batch = 1;

            PInvoke.llama_backend_init(false);

            var mdl = PInvoke.llama_load_model_from_file(args[0], mparams);
            Console.Write($"\nCreating new model context...");
            var ctx = PInvoke.llama_new_context_with_model(mdl, cparams);
            Console.Write($" 100%");
            var bat = PInvoke.llama_batch_init(PInvoke.llama_n_ctx(ctx), 0, 1);

            var requests = new List<Request>();
            var bat_view = stackalloc PInvoke.llama_batch[1];

            var candidates = new PInvoke.llama_token_data[PInvoke.llama_n_vocab(mdl)];
            var candidates_p = stackalloc PInvoke.llama_token_data_array[1];

            for (var token = 0; token < candidates.Length; token++)
                candidates[token] = new PInvoke.llama_token_data();

            var min = (int a, int b) => a < b ? a : b;
            var max = (int a, int b) => a > b ? a : b;

            while (true)
            {
                if (!requests.Any())
                {
                    Console.Write("\n> ");
                    var prompt = Console.ReadLine() ?? String.Empty;

                    prompt = Regex.Replace(prompt, "\\\\n", "\n");

                    if (String.IsNullOrWhiteSpace(prompt))
                        break;

                    var match = Regex.Match(prompt, @"^/load (.+)$");
                    if (match.Success)
                        prompt = File.ReadAllText(match.Groups[1].Value);

                    var tokens = Interop.llama_tokenize(mdl, prompt, true, true);
                    Console.WriteLine($"{tokens.Length} token(s)");

                    requests.Add(new Request(PInvoke.llama_n_ctx(ctx), tokens));

                    PInvoke.llama_kv_cache_seq_rm(ctx, 0, 0, -1);
                }

                bat.n_tokens = 0;

                foreach (var request in requests)
                {
                    for (; request.PosBatch < request.PosTokens; request.PosBatch++)
                        Interop.llama_batch_add(ref bat, request.Tokens[request.PosBatch], request.PosBatch, new[] { request.Id }, false);

                    request.PosLogit = bat.n_tokens - 1;
                    bat.logits[request.PosLogit] = true ? 1 : 0;
                }

                if (bat.n_tokens == 0)
                    break;

                var n_batch = (int)cparams.n_batch;
                for (var i = 0; i < bat.n_tokens; i += n_batch)
                {
                    var n_tokens = min(n_batch, bat.n_tokens - i);

                    bat_view->n_tokens = n_tokens;
                    bat_view->token = bat.token + i;
                    bat_view->embd = null;
                    bat_view->pos = bat.pos + i;
                    bat_view->n_seq_id = bat.n_seq_id + i;
                    bat_view->seq_id = bat.seq_id + i;
                    bat_view->logits = bat.logits + i;
                    bat_view->all_pos_0 = 0;
                    bat_view->all_pos_1 = 0;
                    bat_view->all_seq_id = 0;

                    PInvoke.llama_decode(ctx, *bat_view);

                    foreach (var request in requests)
                    {
                        if (request.PosLogit < i || request.PosLogit >= i + n_tokens)
                            continue;

                        var logits = PInvoke.llama_get_logits_ith(ctx, request.PosLogit - i);

                        for (var token = 0; token < candidates.Length; token++)
                        {
                            candidates[token].id = token;
                            candidates[token].logit = logits[token];
                            candidates[token].p = 0.0f;
                        }

                        fixed (PInvoke.llama_token_data* p1 = &candidates[0])
                        {
                            candidates_p->data = p1;
                            candidates_p->size = (nuint)candidates.Length;
                            candidates_p->sorted = false ? 1 : 0;

                            fixed (llama_token* p2 = &request.Tokens[max(0, request.PosTokens - penalty_last_n)])
                            {
                                PInvoke.llama_sample_repetition_penalties(
                                    ctx,
                                    candidates_p,
                                    p2,
                                    (nuint)penalty_last_n,
                                    penalty_repeat,
                                    penalty_freq,
                                    penalty_present);
                            }

                            llama_token token;
                            if (temp <= 0)
                            {
                                token = PInvoke.llama_sample_token_greedy(ctx, candidates_p);
                            }
                            else if (mirostat == 1)
                            {
                                PInvoke.llama_sample_temp(ctx, candidates_p, temp);
                                token = PInvoke.llama_sample_token_mirostat(ctx, candidates_p, mirostat_tau, mirostat_eta, 100, ref request.MirostatMU);
                            }
                            else if (mirostat == 2)
                            {
                                PInvoke.llama_sample_temp(ctx, candidates_p, temp);
                                token = PInvoke.llama_sample_token_mirostat_v2(ctx, candidates_p, mirostat_tau, mirostat_eta, ref request.MirostatMU);
                            }
                            else
                            {
                                PInvoke.llama_sample_top_k(ctx, candidates_p, top_k, 1);
                                PInvoke.llama_sample_tail_free(ctx, candidates_p, tfs_z, 1);
                                PInvoke.llama_sample_typical(ctx, candidates_p, typical_p, 1);
                                PInvoke.llama_sample_top_p(ctx, candidates_p, top_p, 1);
                                PInvoke.llama_sample_temp(ctx, candidates_p, temp);
                                token = PInvoke.llama_sample_token(ctx, candidates_p);
                            }

                            request.Tokens[request.PosTokens++] = token;
                            Console.Write(Encoding.ASCII.GetString(Interop.llama_token_to_piece(mdl, token)));

                            if (request.T1 == default)
                                request.T1 = DateTime.Now;

                            if (token == PInvoke.llama_token_eos(mdl))
                                request.T2 = DateTime.Now;
                        }
                    }
                }

                requests
                    .Where(r => r.Tokens[r.PosTokens - 1] == PInvoke.llama_token_eos(mdl))
                    .ToList()
                    .ForEach(r => Console.WriteLine($"\n{r.PosTokens / (double)(((r.T2 - r.T1)?.TotalSeconds) ?? 1):F2} t/s"));

                requests.RemoveAll(r => r.Tokens[r.PosTokens - 1] == PInvoke.llama_token_eos(mdl));
            }

            PInvoke.llama_batch_free(bat);
            PInvoke.llama_free(ctx);
            PInvoke.llama_free_model(mdl);
            PInvoke.llama_backend_free();
        }
    }

    internal class Request : IEquatable<Request>
    {
        public int Id { get; set; }

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosTokens { get; set; }
        public llama_token[] Tokens { get; set; }

        public float MirostatMU;

        public DateTime? T1 { get; set; }
        public DateTime? T2 { get; set; }

        public Request(int n_ctx, Span<llama_token> tokens)
        {
            Tokens = new llama_token[n_ctx];
            tokens.CopyTo(Tokens);
            PosTokens += tokens.Length;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is Request request && Equals(request);
        public override int GetHashCode() => Id.GetHashCode();

        // IEquatable<T>
        public bool Equals(Request? other) => other?.Id == this.Id;
    }
}
