using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

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
        private ModelOptions _modelOptions = new();

        private List<LlmRequest> _requests = new();

        private CancellationTokenSource _cancellationTokenSource = new();
        private Task? _task;

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
                    StopAsync().Wait();
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

            if (modelOptions != default)
                _modelOptions = modelOptions;

            if (!_backend.Created)
                _backend.Create(() => PInvoke.llama_backend_init(_backendOptions.NumaOptimizations), PInvoke.llama_backend_free);

            using var progressCallbackHandle = new UnmanagedResource<GCHandle>();
            progressCallbackHandle.Create(() => GCHandle.Alloc(progressCallback), handle => handle.Free());

            var mparams = PInvoke.llama_model_default_params();
            mparams.n_gpu_layers = _modelOptions.GpuLayers;
            mparams.use_mmap = (byte)(_modelOptions.UseMemoryMap ? 1 : 0);
            mparams.progress_callback = &LlmModel._ProgressCallback;
            mparams.progress_callback_user_data = GCHandle.ToIntPtr(progressCallbackHandle.Handle).ToPointer();

            _model.Create(() => PInvoke.llama_load_model_from_file(modelPath, mparams), PInvoke.llama_free_model);

            var cparams = PInvoke.llama_context_default_params();
            cparams.seed = (uint)_modelOptions.Seed;
            cparams.n_ctx = (uint)_modelOptions.ContextLength;
            cparams.n_batch = (uint)_modelOptions.BatchSize;
            cparams.n_threads = (uint)_modelOptions.ThreadCount;
            cparams.n_threads_batch = (uint)_modelOptions.BatchThreadCount;

            _context.Create(() => PInvoke.llama_new_context_with_model(_model.Handle, cparams), PInvoke.llama_free);

            _batch.Create(() => PInvoke.llama_batch_init(PInvoke.llama_n_ctx(_context.Handle), 0, 1), PInvoke.llama_batch_free);
        }

        public void Unload()
        {
            _batch.Dispose();
            _context.Dispose();
            _model.Dispose();
        }

        public Span<int> Tokenize(string prompt, bool addBos = false, bool specialTokens = false) => Interop.llama_tokenize(_model.Handle, prompt, addBos, specialTokens);

        public Task StartAsync()
        {
            if (_task != default)
                throw new InvalidOperationException("Already running.");

            _task = new Task(_Run);
            _task.Start();

            return _task;
        }

        public async Task StopAsync()
        {
            if (_task == default)
                return;

            _cancellationTokenSource.Cancel();
            await (_task ?? Task.CompletedTask);
            _cancellationTokenSource = new();

            _task = default;
        }

        private unsafe void _Run()
        {
            var batchView = stackalloc PInvoke.llama_batch[1];

            var candidates = new PInvoke.llama_token_data[PInvoke.llama_n_vocab(_model.Handle)];
            var candidates_p = stackalloc PInvoke.llama_token_data_array[1];

            for (var token = 0; token < candidates.Length; token++)
                candidates[token] = new PInvoke.llama_token_data();

            _batch.GetResource(out var batch);

            var cancellationToken = _cancellationTokenSource.Token;
            while (!cancellationToken.IsCancellationRequested)
            {
                while (!_requests.Any())
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    Thread.Sleep(10);
                }

                batch.n_tokens = 0;

                foreach (var request in _requests)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    for (; request.PosBatch < request.PosTokens; request.PosBatch++)
                        Interop.llama_batch_add(ref batch, request.Tokens[request.PosBatch], request.PosBatch, new[] { request.Id }, false);

                    request.PosLogit = batch.n_tokens - 1;
                    batch.logits[request.PosLogit] = true ? 1 : 0;
                }

                if (batch.n_tokens == 0)
                    break;

                var nbatch = _modelOptions.BatchSize;
                for (var i = 0; i < batch.n_tokens; i += nbatch)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    var n_tokens = Math.Min(nbatch, batch.n_tokens - i);

                    batchView->n_tokens = n_tokens;
                    batchView->token = batch.token + i;
                    batchView->embd = null;
                    batchView->pos = batch.pos + i;
                    batchView->n_seq_id = batch.n_seq_id + i;
                    batchView->seq_id = batch.seq_id + i;
                    batchView->logits = batch.logits + i;
                    batchView->all_pos_0 = 0;
                    batchView->all_pos_1 = 0;
                    batchView->all_seq_id = 0;

                    PInvoke.llama_decode(_context.Handle, *batchView);

                    foreach (var request in _requests)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

                        if (request.PosLogit < i || request.PosLogit >= i + n_tokens)
                            continue;

                        var logits = PInvoke.llama_get_logits_ith(_context.Handle, request.PosLogit - i);

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

                            fixed (int* p2 = &request.Tokens[Math.Max(0, request.PosTokens - request.SamplingOptions.PenaltyLastN)])
                            {
                                PInvoke.llama_sample_repetition_penalties(
                                    _context.Handle,
                                    candidates_p,
                                    p2,
                                    (nuint)request.SamplingOptions.PenaltyLastN,
                                    request.SamplingOptions.PenaltyRepeat,
                                    request.SamplingOptions.PenaltyFreq,
                                    request.SamplingOptions.PenaltyPresent);
                            }

                            int token;
                            if (request.SamplingOptions.Temperature < 0.0f)
                            {
                                PInvoke.llama_sample_softmax(_context.Handle, candidates_p);
                                token = candidates_p->data[0].id;
                            }
                            else if (request.SamplingOptions.Temperature == 0.0f)
                            {
                                token = PInvoke.llama_sample_token_greedy(_context.Handle, candidates_p);
                            }
                            else if (request.SamplingOptions.Mirostat == Mirostat.MirostatV1)
                            {
                                PInvoke.llama_sample_temp(_context.Handle, candidates_p, request.SamplingOptions.Temperature);
                                token = PInvoke.llama_sample_token_mirostat(
                                    _context.Handle,
                                    candidates_p,
                                    request.SamplingOptions.MirostatTau,
                                    request.SamplingOptions.MirostatEta,
                                    100, ref request.MirostatMU
                                );
                            }
                            else if (request.SamplingOptions.Mirostat == Mirostat.MirostatV2)
                            {
                                PInvoke.llama_sample_temp(_context.Handle, candidates_p, request.SamplingOptions.Temperature);
                                token = PInvoke.llama_sample_token_mirostat_v2(
                                    _context.Handle,
                                    candidates_p,
                                    request.SamplingOptions.MirostatTau,
                                    request.SamplingOptions.MirostatEta,
                                    ref request.MirostatMU
                                );
                            }
                            else
                            {
                                PInvoke.llama_sample_top_k(_context.Handle, candidates_p, request.SamplingOptions.TopK, 1);
                                PInvoke.llama_sample_tail_free(_context.Handle, candidates_p, request.SamplingOptions.TfsZ, 1);
                                PInvoke.llama_sample_typical(_context.Handle, candidates_p, request.SamplingOptions.TypicalP, 1);
                                PInvoke.llama_sample_top_p(_context.Handle, candidates_p, request.SamplingOptions.TopP, 1);
                                PInvoke.llama_sample_temp(_context.Handle, candidates_p, request.SamplingOptions.Temperature);
                                token = PInvoke.llama_sample_token(_context.Handle, candidates_p);
                            }

                            request.Tokens[request.PosTokens++] = token;
                            Console.Write(Encoding.ASCII.GetString(Interop.llama_token_to_piece(_model.Handle, token)));

                            if (request.T1 == default)
                                request.T1 = DateTime.Now;

                            if (token == PInvoke.llama_token_eos(_model.Handle))
                                request.T2 = DateTime.Now;
                        }
                    }
                }

                _requests.RemoveAll(r => r.Tokens[r.PosTokens - 1] == PInvoke.llama_token_eos(_model.Handle));
            }
        }
    }

    internal class LlmRequest : IEquatable<LlmRequest>
    {
        public int Id { get; set; }

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosTokens { get; set; }
        public int[] Tokens { get; set; }

        public SamplingOptions SamplingOptions { get; set; } = new();
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
