using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    public class LlmEngine : IDisposable
    {
        private bool _disposed;

        private UnmanagedResource _backend = new();
        private UnmanagedResource<nint> _model = new();
        private UnmanagedResource<nint> _context = new();
        private UnmanagedResource<PInvoke.llama_batch> _batch = new();

        private EngineOptions _engineOptions = new();
        private ModelOptions _modelOptions = new();

        private ConcurrentQueue<LlmRequest> _requests = new();

        private CancellationTokenSource _cancellationTokenSource = new();
        private Task? _task;

        public LlmEngine() { }
        public LlmEngine(EngineOptions backendOptions) => _engineOptions = backendOptions;

        ~LlmEngine() => Dispose(disposing: false);

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

        public unsafe void LoadModel(string modelPath, ModelOptions? modelOptions = default, Action<float>? progressCallback = default)
        {
            if (_model.Created)
                throw new InvalidOperationException("Model already loaded.");

            if (modelOptions != default)
                _modelOptions = modelOptions;

            if (!_backend.Created)
                _backend.Create(() => PInvoke.llama_backend_init(_engineOptions.NumaOptimizations), PInvoke.llama_backend_free);

            using var progressCallbackHandle = new UnmanagedResource<GCHandle>();
            progressCallbackHandle.Create(() => GCHandle.Alloc(progressCallback), handle => handle.Free());

            var mparams = PInvoke.llama_model_default_params();
            mparams.n_gpu_layers = _modelOptions.GpuLayers;
            mparams.use_mmap = (byte)(_modelOptions.UseMemoryMap ? 1 : 0);
            mparams.progress_callback = &LlmEngine._ProgressCallback;
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

        public void UnloadModel()
        {
            _batch.Dispose();
            _context.Dispose();
            _model.Dispose();
        }

        public Span<int> Tokenize(string prompt, bool prependBosToken = false, bool processSpecialTokens = false) =>
            Interop.llama_tokenize(_model.Handle, prompt, prependBosToken, processSpecialTokens);

        public Task RunAsync()
        {
            if (_task != default)
                throw new InvalidOperationException("Already running.");

            return _task = Task.Run(_Run);
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

        public bool IsRunning => _task?.Status == TaskStatus.Running;

        public async Task WaitForRunningAsync(int pollingRateMs = 10)
        {
            while (!this.IsRunning)
                await Task.Delay(pollingRateMs);
        }

        public LlmRequest NewRequest(string prompt, bool prependBosToken = false, bool processSpecialTokens = false)
        {
            var request = new LlmRequest(prompt, prependBosToken, processSpecialTokens);
            _requests.Enqueue(request);
            return request;
        }

        private unsafe void _Run()
        {
            var contextLength = PInvoke.llama_n_ctx(_context.Handle);
            var eosToken = PInvoke.llama_token_eos(_model.Handle);

            var candidates = new PInvoke.llama_token_data[PInvoke.llama_n_vocab(_model.Handle)];
            var candidatesPtr = stackalloc PInvoke.llama_token_data_array[1];

            for (var token = 0; token < candidates.Length; token++)
                candidates[token] = new PInvoke.llama_token_data();

            var batchPtr = stackalloc PInvoke.llama_batch[1];
            _batch.GetResource(out var batch);

            var sequences = new Slots<LlmSequence>(_engineOptions.MaxParallel);

            var cancellationToken = _cancellationTokenSource.Token;
            while (!cancellationToken.IsCancellationRequested)
            {
                // Fill as many sequence slots as possible given pending requests
                while (sequences.HasFreeSlot && _requests.Count > 0)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    if (_requests.TryDequeue(out var request))
                    {
                        var sequence = new LlmSequence(request, contextLength, Tokenize(request.Prompt, request.PrependBosToken, request.ProcessSpecialTokens));
                        var id = sequences.Add(sequence);
                        sequence.Id = id;
                    }
                }

                batch.n_tokens = 0;

                foreach (var sequence in sequences)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    for (; sequence.PosBatch < sequence.PosTokens; sequence.PosBatch++)
                        Interop.llama_batch_add(ref batch, sequence.Tokens[sequence.PosBatch], sequence.PosBatch, new[] { sequence.Id }, false);

                    sequence.PosLogit = batch.n_tokens - 1;
                    batch.logits[sequence.PosLogit] = true ? 1 : 0;
                }

                // Idle
                if (batch.n_tokens == 0)
                {
                    Thread.Sleep(10);
                    continue;
                }

                var batchSize = _modelOptions.BatchSize;
                for (var i = 0; i < batch.n_tokens; i += batchSize)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    var n_tokens = Math.Min(batchSize, batch.n_tokens - i);

                    batchPtr->n_tokens = n_tokens;
                    batchPtr->token = batch.token + i;
                    batchPtr->embd = null;
                    batchPtr->pos = batch.pos + i;
                    batchPtr->n_seq_id = batch.n_seq_id + i;
                    batchPtr->seq_id = batch.seq_id + i;
                    batchPtr->logits = batch.logits + i;
                    batchPtr->all_pos_0 = 0;
                    batchPtr->all_pos_1 = 0;
                    batchPtr->all_seq_id = 0;

                    PInvoke.llama_decode(_context.Handle, *batchPtr);

                    foreach (var sequence in sequences)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

                        if (sequence.PosLogit < i || sequence.PosLogit >= i + n_tokens)
                            continue;

                        var logits = PInvoke.llama_get_logits_ith(_context.Handle, sequence.PosLogit - i);

                        for (var token = 0; token < candidates.Length; token++)
                        {
                            candidates[token].id = token;
                            candidates[token].logit = logits[token];
                            candidates[token].p = 0.0f;
                        }

                        fixed (PInvoke.llama_token_data* ptrCandidates = &candidates[0])
                        {
                            candidatesPtr->data = ptrCandidates;
                            candidatesPtr->size = (nuint)candidates.Length;
                            candidatesPtr->sorted = false ? 1 : 0;

                            fixed (int* ptrTokens = &sequence.Tokens[Math.Max(0, sequence.PosTokens - sequence.SamplingOptions.PenaltyLastN)])
                            {
                                PInvoke.llama_sample_repetition_penalties(
                                    _context.Handle,
                                    candidatesPtr,
                                    ptrTokens,
                                    (nuint)sequence.SamplingOptions.PenaltyLastN,
                                    sequence.SamplingOptions.PenaltyRepeat,
                                    sequence.SamplingOptions.PenaltyFreq,
                                    sequence.SamplingOptions.PenaltyPresent);
                            }

                            var token = default(int);

                            if (sequence.SamplingOptions.Temperature < 0.0f)
                            {
                                PInvoke.llama_sample_softmax(_context.Handle, candidatesPtr);
                                token = candidatesPtr->data[0].id;
                            }
                            else if (sequence.SamplingOptions.Temperature == 0.0f)
                            {
                                token = PInvoke.llama_sample_token_greedy(_context.Handle, candidatesPtr);
                            }
                            else if (sequence.SamplingOptions.Mirostat == Mirostat.MirostatV1)
                            {
                                PInvoke.llama_sample_temp(_context.Handle, candidatesPtr, sequence.SamplingOptions.Temperature);
                                token = PInvoke.llama_sample_token_mirostat(
                                    _context.Handle,
                                    candidatesPtr,
                                    sequence.SamplingOptions.MirostatTau,
                                    sequence.SamplingOptions.MirostatEta,
                                    sequence.MirostatM,
                                    ref sequence.MirostatMu
                                );
                            }
                            else if (sequence.SamplingOptions.Mirostat == Mirostat.MirostatV2)
                            {
                                PInvoke.llama_sample_temp(_context.Handle, candidatesPtr, sequence.SamplingOptions.Temperature);
                                token = PInvoke.llama_sample_token_mirostat_v2(
                                    _context.Handle,
                                    candidatesPtr,
                                    sequence.SamplingOptions.MirostatTau,
                                    sequence.SamplingOptions.MirostatEta,
                                    ref sequence.MirostatMu
                                );
                            }
                            else
                            {
                                PInvoke.llama_sample_top_k(_context.Handle, candidatesPtr, sequence.SamplingOptions.TopK, 1);
                                PInvoke.llama_sample_tail_free(_context.Handle, candidatesPtr, sequence.SamplingOptions.TfsZ, 1);
                                PInvoke.llama_sample_typical(_context.Handle, candidatesPtr, sequence.SamplingOptions.TypicalP, 1);
                                PInvoke.llama_sample_top_p(_context.Handle, candidatesPtr, sequence.SamplingOptions.TopP, 1);
                                PInvoke.llama_sample_temp(_context.Handle, candidatesPtr, sequence.SamplingOptions.Temperature);
                                token = PInvoke.llama_sample_token(_context.Handle, candidatesPtr);
                            }

                            sequence.Tokens[sequence.PosTokens++] = token;

                            if (!sequence.Request.Tokens.Writer.TryWrite(Interop.llama_token_to_piece(_model.Handle, token).ToArray()))
                                throw new Exception("Unable to write next token to request channel.");

                            if (token == eosToken)
                                sequence.Request.Tokens.Writer.Complete();
                        }
                    }
                }

                sequences.RemoveAll(r => r.Tokens[r.PosTokens - 1] == eosToken);
            }
        }
    }
}
