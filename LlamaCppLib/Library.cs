using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    public class LlmModel : IDisposable
    {
        private bool _disposed;

        private UnmanagedResource _backend = new();
        private UnmanagedResource<nint> _model = new();
        private UnmanagedResource<nint> _context = new();
        private UnmanagedResource<PInvoke.llama_batch> _batch = new();

        private BackendOptions _backendOptions = new();

        public LlmModel() { }
        public LlmModel(BackendOptions backendOptions) => _backendOptions = backendOptions;

        ~LlmModel() => Dispose(disposing: false);

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Managed
                }

                // Unmanaged
                _batch.Dispose();
                _context.Dispose();
                _model.Dispose();
                _backend.Dispose();

                _disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe void _ProgressCallback(float progress, void* state)
        {
            var callback = (Action<float>?)GCHandle.FromIntPtr(new(state)).Target;
            callback?.Invoke(progress * 100);
        }

        public unsafe void Load(string modelPath, ModelOptions? modelOptions = default, Action<float>? progressCallback = default)
        {
            if (_model.Created)
                throw new InvalidOperationException("Model already loaded.");

            if (modelOptions == default)
                modelOptions = new();

            if (!_backend.Created)
                _backend.Create(() => PInvoke.llama_backend_init(_backendOptions.NumaOptimizations), PInvoke.llama_backend_free);

            using var progressCallbackHandle = new UnmanagedResource<GCHandle>();
            progressCallbackHandle.Create(() => GCHandle.Alloc(progressCallback), handle => handle.Free());

            var mparams = PInvoke.llama_model_default_params();
            mparams.n_gpu_layers = modelOptions.GpuLayers;
            mparams.use_mmap = (byte)(modelOptions.UseMemoryMap ? 1 : 0);
            mparams.progress_callback = &LlmModel._ProgressCallback;
            mparams.progress_callback_user_data = GCHandle.ToIntPtr((GCHandle)progressCallbackHandle).ToPointer();

            _model.Create(() => PInvoke.llama_load_model_from_file(modelPath, mparams), PInvoke.llama_free_model);

            var cparams = PInvoke.llama_context_default_params();
            cparams.seed = (uint)modelOptions.Seed;
            cparams.n_ctx = (uint)modelOptions.ContextLength;
            cparams.n_batch = (uint)modelOptions.BatchSize;
            cparams.n_threads = (uint)modelOptions.ThreadCount;
            cparams.n_threads_batch = (uint)modelOptions.BatchThreadCount;

            _context.Create(() => PInvoke.llama_new_context_with_model((nint)_model, cparams), PInvoke.llama_free);

            _batch.Create(() => PInvoke.llama_batch_init(PInvoke.llama_n_ctx((nint)_context), 0, 1), PInvoke.llama_batch_free);
        }

        public void Unload()
        {
            _batch.Dispose();
            _context.Dispose();
            _model.Dispose();
        }

        //public IEnumerable<byte[]> NextToken(SamplingOptions? samplingOptions = default)
        //{
        //    var requests = new List<LlmRequest>();
        //    var bat_view = stackalloc PInvoke.llama_batch[1];

        //    var candidates = new PInvoke.llama_token_data[PInvoke.llama_n_vocab((nint)_model)];
        //    var candidates_p = stackalloc PInvoke.llama_token_data_array[1];

        //    for (var token = 0; token < candidates.Length; token++)
        //        candidates[token] = new PInvoke.llama_token_data();

        //    while (true)
        //    {
        //        if (!requests.Any())
        //        {
        //            Console.Write("\n> ");
        //            var prompt = Console.ReadLine() ?? String.Empty;

        //            prompt = Regex.Replace(prompt, "\\\\n", "\n");

        //            if (String.IsNullOrWhiteSpace(prompt))
        //                break;

        //            var match = Regex.Match(prompt, @"^/load (.+)$");
        //            if (match.Success)
        //                prompt = File.ReadAllText(match.Groups[1].Value);

        //            var tokens = Interop.llama_tokenize(mdl, prompt, true, true);
        //            Console.WriteLine($"{tokens.Length} token(s)");

        //            requests.Add(new LlmRequest(PInvoke.llama_n_ctx(ctx), tokens));

        //            PInvoke.llama_kv_cache_seq_rm(ctx, 0, 0, -1);
        //        }

        //        bat.n_tokens = 0;

        //        foreach (var request in requests)
        //        {
        //            for (; request.PosBatch < request.PosTokens; request.PosBatch++)
        //                Interop.llama_batch_add(ref bat, request.Tokens[request.PosBatch], request.PosBatch, new[] { request.Id }, false);

        //            request.PosLogit = bat.n_tokens - 1;
        //            bat.logits[request.PosLogit] = true ? 1 : 0;
        //        }

        //        if (bat.n_tokens == 0)
        //            break;

        //        var n_batch = (int)cparams.n_batch;
        //        for (var i = 0; i < bat.n_tokens; i += n_batch)
        //        {
        //            var n_tokens = Math.Min(n_batch, bat.n_tokens - i);

        //            bat_view->n_tokens = n_tokens;
        //            bat_view->token = bat.token + i;
        //            bat_view->embd = null;
        //            bat_view->pos = bat.pos + i;
        //            bat_view->n_seq_id = bat.n_seq_id + i;
        //            bat_view->seq_id = bat.seq_id + i;
        //            bat_view->logits = bat.logits + i;
        //            bat_view->all_pos_0 = 0;
        //            bat_view->all_pos_1 = 0;
        //            bat_view->all_seq_id = 0;

        //            PInvoke.llama_decode(ctx, *bat_view);

        //            foreach (var request in requests)
        //            {
        //                if (request.PosLogit < i || request.PosLogit >= i + n_tokens)
        //                    continue;

        //                var logits = PInvoke.llama_get_logits_ith(ctx, request.PosLogit - i);

        //                for (var token = 0; token < candidates.Length; token++)
        //                {
        //                    candidates[token].id = token;
        //                    candidates[token].logit = logits[token];
        //                    candidates[token].p = 0.0f;
        //                }

        //                fixed (PInvoke.llama_token_data* p1 = &candidates[0])
        //                {
        //                    candidates_p->data = p1;
        //                    candidates_p->size = (nuint)candidates.Length;
        //                    candidates_p->sorted = false ? 1 : 0;

        //                    fixed (llama_token* p2 = &request.Tokens[Math.Max(0, request.PosTokens - samplingOptions.PenaltyLastN)])
        //                    {
        //                        PInvoke.llama_sample_repetition_penalties(
        //                            ctx,
        //                            candidates_p,
        //                            p2,
        //                            (nuint)samplingOptions.PenaltyLastN,
        //                            samplingOptions.PenaltyRepeat,
        //                            samplingOptions.PenaltyFreq,
        //                            samplingOptions.PenaltyPresent);
        //                    }

        //                    llama_token token;
        //                    if (samplingOptions.Temperature < 0.0f)
        //                    {
        //                        PInvoke.llama_sample_softmax(ctx, candidates_p);
        //                        token = candidates_p->data[0].id;
        //                    }
        //                    else if (samplingOptions.Temperature == 0.0f)
        //                    {
        //                        token = PInvoke.llama_sample_token_greedy(ctx, candidates_p);
        //                    }
        //                    else if (samplingOptions.Mirostat == Mirostat.MirostatV1)
        //                    {
        //                        PInvoke.llama_sample_temp(ctx, candidates_p, samplingOptions.Temperature);
        //                        token = PInvoke.llama_sample_token_mirostat(ctx, candidates_p, samplingOptions.MirostatTau, samplingOptions.MirostatEta, 100, ref request.MirostatMU);
        //                    }
        //                    else if (samplingOptions.Mirostat == Mirostat.MirostatV2)
        //                    {
        //                        PInvoke.llama_sample_temp(ctx, candidates_p, samplingOptions.Temperature);
        //                        token = PInvoke.llama_sample_token_mirostat_v2(ctx, candidates_p, samplingOptions.MirostatTau, samplingOptions.MirostatEta, ref request.MirostatMU);
        //                    }
        //                    else
        //                    {
        //                        PInvoke.llama_sample_top_k(ctx, candidates_p, samplingOptions.TopK, 1);
        //                        PInvoke.llama_sample_tail_free(ctx, candidates_p, samplingOptions.TfsZ, 1);
        //                        PInvoke.llama_sample_typical(ctx, candidates_p, samplingOptions.TypicalP, 1);
        //                        PInvoke.llama_sample_top_p(ctx, candidates_p, samplingOptions.TopP, 1);
        //                        PInvoke.llama_sample_temp(ctx, candidates_p, samplingOptions.Temperature);
        //                        token = PInvoke.llama_sample_token(ctx, candidates_p);
        //                    }

        //                    request.Tokens[request.PosTokens++] = token;
        //                    Console.Write(Encoding.ASCII.GetString(Interop.llama_token_to_piece(mdl, token)));

        //                    if (request.T1 == default)
        //                        request.T1 = DateTime.Now;

        //                    if (token == PInvoke.llama_token_eos(mdl))
        //                        request.T2 = DateTime.Now;
        //                }
        //            }
        //        }

        //        requests
        //            .Where(r => r.Tokens[r.PosTokens - 1] == PInvoke.llama_token_eos(mdl))
        //            .ToList()
        //            .ForEach(r => Console.WriteLine($"\n{r.PosTokens / (double)(((r.T2 - r.T1)?.TotalSeconds) ?? 1):F2} t/s"));

        //        requests.RemoveAll(r => r.Tokens[r.PosTokens - 1] == PInvoke.llama_token_eos(mdl));
        //    }
        //}
    }

    internal class LlmRequest : IEquatable<LlmRequest>
    {
        public int Id { get; set; }

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosTokens { get; set; }
        public int[] Tokens { get; set; }

        public float MirostatMU = 0.0f;

        public DateTime? T1 { get; set; }
        public DateTime? T2 { get; set; }

        public LlmRequest(int n_ctx, Span<int> tokens)
        {
            this.Tokens = new int[n_ctx];
            tokens.CopyTo(Tokens);
            PosTokens += tokens.Length;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is LlmRequest request && Equals(request);
        public override int GetHashCode() => Id.GetHashCode();

        // IEquatable<T>
        public bool Equals(LlmRequest? other) => other?.Id == this.Id;
    }
}
