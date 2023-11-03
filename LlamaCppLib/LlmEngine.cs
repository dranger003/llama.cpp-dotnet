using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using static LlamaCppLib.Native;
using static LlamaCppLib.Interop;

namespace LlamaCppLib
{
    public class LlmEngine : IDisposable
    {
        private bool _disposed = default;

        private UnmanagedResource _backend = new();
        private UnmanagedResource<nint> _model = new();
        private UnmanagedResource<nint> _context = new();
        private UnmanagedResource<llama_batch> _batch = new();

        private EngineOptions _engineOptions = new();
        private ModelOptions _modelOptions = new();

        private ConcurrentQueue<LlmPrompt> _requests = new();

        private CancellationTokenSource _cancellationTokenSource = new();
        private Task? _mainTask = default;

        public LlmEngine(EngineOptions? engineOptions = default)
        {
            if (engineOptions != default)
                _engineOptions = engineOptions;
        }

        ~LlmEngine() => Dispose(false);

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
            Dispose(true);
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
                _backend.Create(() => llama_backend_init(_engineOptions.NumaOptimizations), llama_backend_free);

            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = _modelOptions.GpuLayers;
            mparams.use_mmap = (byte)(_modelOptions.UseMemoryMap ? 1 : 0);

            using var progressCallbackHandle = new UnmanagedResource<GCHandle>();
            if (progressCallback != default)
            {
                progressCallbackHandle.Create(() => GCHandle.Alloc(progressCallback), handle => handle.Free());
                mparams.progress_callback = &LlmEngine._ProgressCallback;
                mparams.progress_callback_user_data = GCHandle.ToIntPtr(progressCallbackHandle.Handle).ToPointer();
            }

            _model.Create(() => llama_load_model_from_file(modelPath, mparams), llama_free_model);

            var cparams = llama_context_default_params();
            cparams.seed = (uint)_modelOptions.Seed;
            cparams.n_ctx = (uint)_modelOptions.ContextLength;
            cparams.n_batch = (uint)_modelOptions.BatchSize;
            cparams.n_threads = (uint)_modelOptions.ThreadCount;
            cparams.n_threads_batch = (uint)_modelOptions.BatchThreadCount;

            _context.Create(() => llama_new_context_with_model(_model.Handle, cparams), llama_free);

            _batch.Create(() => llama_batch_init(llama_n_ctx(_context.Handle), 0, 1), llama_batch_free);
        }

        public void UnloadModel()
        {
            _batch.Dispose();
            _context.Dispose();
            _model.Dispose();
        }

        public Span<int> Tokenize(string prompt, bool prependBosToken = false, bool processSpecialTokens = false) =>
            llama_tokenize(_model.Handle, prompt, prependBosToken, processSpecialTokens);

        public void StartAsync()
        {
            if (_mainTask != default)
                throw new InvalidOperationException("Already running.");

            _mainTask = Task.Run(_Run);
        }

        public async Task StopAsync()
        {
            if (_mainTask == default)
                return;

            _cancellationTokenSource.Cancel();
            await (_mainTask ?? Task.CompletedTask);
            _cancellationTokenSource = new();

            _mainTask = default;
        }

        public bool IsRunning => _mainTask?.Status == TaskStatus.Running;

        public async Task WaitForRunningAsync(int pollingRateMs = 10)
        {
            while (!this.IsRunning)
                await Task.Delay(pollingRateMs);
        }

        public LlmPrompt Prompt(
            string prompt,
            SamplingOptions? samplingOptions = default,
            bool prependBosToken = false,
            bool processSpecialTokens = false,
            int[]? extraStopTokens = default
        )
        {
            var request = new LlmPrompt(prompt, samplingOptions ?? new(), prependBosToken, processSpecialTokens) { ExtraStopTokens = extraStopTokens };
            _requests.Enqueue(request);
            return request;
        }

        private unsafe void _Run()
        {
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
                        var sequence = new LlmSequence(request, llama_n_ctx(_context.Handle), Tokenize(request.Prompt, request.PrependBosToken, request.ProcessSpecialTokens))
                        {
                            T1 = DateTime.Now,
                        };

                        var id = sequences.Add(sequence);
                        sequence.Id = id;
                    }
                }

                if (cancellationToken.IsCancellationRequested)
                    continue;

                batch.n_tokens = 0;

                foreach (var sequence in sequences)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    for (; sequence.PosBatch < sequence.PosTokens; sequence.PosBatch++)
                        llama_batch_add(ref batch, sequence.Tokens[sequence.PosBatch], sequence.PosBatch, new[] { sequence.Id }, false);

                    sequence.PosLogit = batch.n_tokens - 1;
                    batch.logits[sequence.PosLogit] = true ? 1 : 0;
                }

                if (cancellationToken.IsCancellationRequested)
                    continue;

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

                    var result = llama_decode(
                        _context.Handle,
                        new llama_batch
                        {
                            n_tokens = n_tokens,
                            token = batch.token + i,
                            embd = null,
                            pos = batch.pos + i,
                            n_seq_id = batch.n_seq_id + i,
                            seq_id = batch.seq_id + i,
                            logits = batch.logits + i,
                            all_pos_0 = 0,
                            all_pos_1 = 0,
                            all_seq_id = 0,
                        }
                    );

                    if (result != 0)
                    {
                        foreach (var sequence in sequences)
                            sequence.Request.Tokens.Writer.Complete(new InsufficientMemoryException());

                        sequences.RemoveAll(sequence => true);
                        llama_kv_cache_clear(_context.Handle);

                        continue;
                    }

                    foreach (var sequence in sequences)
                    {
                        if (cancellationToken.IsCancellationRequested)
                            break;

                        if (sequence.PosLogit < i || sequence.PosLogit >= i + n_tokens)
                            continue;

                        var logits = llama_get_logits_ith(_context.Handle, sequence.PosLogit - i);

                        var candidates = new llama_token_data[llama_n_vocab(_model.Handle)];
                        for (var token = 0; token < candidates.Length; token++)
                        {
                            if (cancellationToken.IsCancellationRequested)
                                break;

                            candidates[token].id = token;
                            candidates[token].logit = logits[token];
                            candidates[token].p = 0.0f;
                        }

                        if (cancellationToken.IsCancellationRequested)
                            continue;

                        fixed (llama_token_data* ptrCandidates = &candidates[0])
                        {
                            var candidates_p = new llama_token_data_array
                            {
                                data = ptrCandidates,
                                size = (nuint)candidates.Length,
                                sorted = false ? 1 : 0,
                            };

                            fixed (int* ptrTokens = &sequence.Tokens[Math.Max(0, sequence.PosTokens - sequence.SamplingOptions.PenaltyLastN)])
                            {
                                llama_sample_repetition_penalties(
                                    _context.Handle,
                                    ref candidates_p,
                                    ptrTokens,
                                    (nuint)sequence.SamplingOptions.PenaltyLastN,
                                    sequence.SamplingOptions.PenaltyRepeat,
                                    sequence.SamplingOptions.PenaltyFreq,
                                    sequence.SamplingOptions.PenaltyPresent
                                );
                            }

                            var token = llama_token_eos(_model.Handle);

                            if (sequence.SamplingOptions.Temperature < 0.0f)
                            {
                                llama_sample_softmax(_context.Handle, ref candidates_p);
                                token = candidates_p.data[0].id;
                            }
                            else if (sequence.SamplingOptions.Temperature == 0.0f)
                            {
                                token = llama_sample_token_greedy(_context.Handle, ref candidates_p);
                            }
                            else if (sequence.SamplingOptions.Mirostat == Mirostat.MirostatV1)
                            {
                                llama_sample_temp(_context.Handle, ref candidates_p, sequence.SamplingOptions.Temperature);
                                token = llama_sample_token_mirostat(
                                    _context.Handle,
                                    ref candidates_p,
                                    sequence.SamplingOptions.MirostatTau,
                                    sequence.SamplingOptions.MirostatEta,
                                    sequence.MirostatM,
                                    ref sequence.MirostatMu
                                );
                            }
                            else if (sequence.SamplingOptions.Mirostat == Mirostat.MirostatV2)
                            {
                                llama_sample_temp(_context.Handle, ref candidates_p, sequence.SamplingOptions.Temperature);
                                token = llama_sample_token_mirostat_v2(
                                    _context.Handle,
                                    ref candidates_p,
                                    sequence.SamplingOptions.MirostatTau,
                                    sequence.SamplingOptions.MirostatEta,
                                    ref sequence.MirostatMu
                                );
                            }
                            else
                            {
                                llama_sample_top_k(_context.Handle, ref candidates_p, sequence.SamplingOptions.TopK, 1);
                                llama_sample_tail_free(_context.Handle, ref candidates_p, sequence.SamplingOptions.TfsZ, 1);
                                llama_sample_typical(_context.Handle, ref candidates_p, sequence.SamplingOptions.TypicalP, 1);
                                llama_sample_top_p(_context.Handle, ref candidates_p, sequence.SamplingOptions.TopP, 1);
                                llama_sample_temp(_context.Handle, ref candidates_p, sequence.SamplingOptions.Temperature);
                                token = llama_sample_token(_context.Handle, ref candidates_p);
                            }

                            sequence.T2 ??= DateTime.Now;

                            sequence.Tokens[sequence.PosTokens++] = token;
                            sequence.Request.Tokens.Writer.TryWrite(llama_token_to_piece(_model.Handle, token).ToArray());

                            if (
                                token == llama_token_eos(_model.Handle)
                                || sequence.Request.Cancelled
                                || (sequence.Request.ExtraStopTokens?.Contains(token) ?? false)
                                || sequence.PosTokens >= sequence.SamplingOptions.MaxSequenceLength
                            )
                            {
                                sequence.T3 = DateTime.Now;

                                sequence.Request.PromptingTime = (sequence.T2 - sequence.T1) ?? new();
                                sequence.Request.SamplingTime = (sequence.T3 - sequence.T2) ?? new();
                                sequence.Request.Tokens.Writer.Complete();

                                llama_kv_cache_seq_rm(_context.Handle, sequence.Id, -1, -1);
                            }
                        }
                    }
                }

                sequences.RemoveAll(
                    sequence =>
                        sequence.Tokens[sequence.PosTokens - 1] == llama_token_eos(_model.Handle)
                        || sequence.Request.Cancelled
                        || (sequence.Request.ExtraStopTokens?.Contains(sequence.Tokens[sequence.PosTokens - 1]) ?? false)
                );
            }

            if (cancellationToken.IsCancellationRequested)
            {
                // Notify outstanding requests of cancellation
                foreach (var sequence in sequences)
                    sequence.Request.Tokens.Writer.Complete(new OperationCanceledException());
            }
        }
    }
}
