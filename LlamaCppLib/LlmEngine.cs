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

        private LlmEngineOptions _engineOptions = new();
        private LlmModelOptions _modelOptions = new();

        private BlockingQueue<LlmPrompt> _prompts = new();

        private CancellationTokenSource _cancellationTokenSource = new();
        private Task? _mainLoop = default;

        public LlmEngine(LlmEngineOptions? engineOptions = default)
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
                    _StopAsync().Wait();
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

        public unsafe void LoadModel(string modelPath, LlmModelOptions? modelOptions = default, Action<float>? progressCallback = default)
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

            _StartAsync();
        }

        public void UnloadModel()
        {
            _StopAsync().Wait();

            _batch.Dispose();
            _context.Dispose();
            _model.Dispose();
        }

        public Span<int> Tokenize(string prompt, bool prependBosToken = false, bool processSpecialTokens = false) => llama_tokenize(_model.Handle, prompt, prependBosToken, processSpecialTokens);

        public bool Loaded => _mainLoop?.Status == TaskStatus.Running;

        public LlmPrompt Prompt(
            string promptText,
            SamplingOptions? samplingOptions = default,
            bool? prependBosToken = default,
            bool? processSpecialTokens = default,
            int[]? extraStopTokens = default
        )
        {
            var prompt = new LlmPrompt(
                promptText,
                samplingOptions ?? new(),
                prependBosToken ?? llama_add_bos_token(_model.Handle) > 0 ? true : llama_vocab_type(_model.Handle) == llama_vocab_type_t.LLAMA_VOCAB_TYPE_SPM,
                processSpecialTokens ?? true
            )
            { ExtraStopTokens = extraStopTokens };

            _prompts.Enqueue(prompt);
            return prompt;
        }

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static unsafe void _ProgressCallback(float progress, void* state)
        {
            var callback = (Action<float>?)GCHandle.FromIntPtr(new(state)).Target;
            callback?.Invoke(progress * 100);
        }

        private void _StartAsync()
        {
            if (_mainLoop != default)
                return;

            _mainLoop = Task.Run(_Run);
        }

        private async Task _StopAsync()
        {
            if (_mainLoop == default)
                return;

            _cancellationTokenSource.Cancel();
            await (_mainLoop ?? Task.CompletedTask);
            _cancellationTokenSource = new();

            _mainLoop = default;
        }

        private unsafe void _Run()
        {
            _batch.GetResource(out var batch);

            var sequences = new Slots<LlmSequence>(_engineOptions.MaxParallel);

            var candidates = new llama_token_data[llama_n_vocab(_model.Handle)];
            var batchView = new llama_batch();

            var cancellationToken = _cancellationTokenSource.Token;
            while (!cancellationToken.IsCancellationRequested)
            {
                // Fill as many sequence slots as possible given pending requests
                while (sequences.HasFreeSlot && _prompts.Any())
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    var prompt = _prompts.Dequeue();

                    var sequence = new LlmSequence(
                        prompt,
                        llama_n_ctx(_context.Handle),
                        Tokenize(prompt.PromptText, prompt.PrependBosToken, prompt.ProcessSpecialTokens)
                    )
                    { T1 = DateTime.Now };

                    var id = sequences.Add(sequence);
                    sequence.Id = id;
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

                if (batch.n_tokens == 0)
                {
                    _prompts.WaitForNext(cancellationToken);
                    continue;
                }

                var batchSize = _modelOptions.BatchSize;
                for (var i = 0; i < batch.n_tokens; i += batchSize)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    var n_tokens = Math.Min(batchSize, batch.n_tokens - i);

                    batchView.n_tokens = n_tokens;
                    batchView.token = batch.token + i;
                    batchView.embd = null;
                    batchView.pos = batch.pos + i;
                    batchView.n_seq_id = batch.n_seq_id + i;
                    batchView.seq_id = batch.seq_id + i;
                    batchView.logits = batch.logits + i;
                    batchView.all_pos_0 = 0;
                    batchView.all_pos_1 = 0;
                    batchView.all_seq_id = 0;

                    var result = llama_decode(_context.Handle, batchView);

                    if (result != 0)
                    {
                        foreach (var sequence in sequences)
                            sequence.Prompt.TokenChannel.Writer.Complete(new InsufficientMemoryException());

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

                            if (sequence.T2 == default)
                            {
                                sequence.T2 = DateTime.Now;
                                sequence.Prompt.PromptingSpeed = sequence.PosResponse / ((sequence.T2 - sequence.T1) ?? new()).TotalSeconds;
                            }

                            var stop = false
                                || sequence.Prompt.Cancelled
                                || sequence.PosTokens >= sequence.Tokens.Length - 1
                                || sequence.PosTokens - sequence.PosResponse >= sequence.SamplingOptions.ResponseMaxTokenCount;

                            if (!stop)
                            {
                                sequence.Prompt.TokenChannel.Writer.TryWrite(llama_token_to_piece(_model.Handle, token).ToArray());
                                sequence.Tokens[sequence.PosTokens++] = token;
                            }

                            if (stop || token == llama_token_eos(_model.Handle) || (sequence.Prompt.ExtraStopTokens?.Contains(token) ?? false))
                            {
                                sequence.T3 = DateTime.Now;
                                sequence.Prompt.SamplingSpeed = (sequence.PosTokens - sequence.PosResponse - 1) / ((sequence.T3 - sequence.T2) ?? new()).TotalSeconds;

                                if (stop)
                                    sequence.Prompt.TokenChannel.Writer.Complete(new OperationCanceledException());
                                else
                                    sequence.Prompt.TokenChannel.Writer.Complete();

                                llama_kv_cache_seq_rm(_context.Handle, sequence.Id, -1, -1);
                                sequences.Remove(sequence.Id);
                            }
                        }
                    }
                }
            }

            if (cancellationToken.IsCancellationRequested)
            {
                // Notify outstanding requests of cancellation
                foreach (var sequence in sequences)
                    sequence.Prompt.TokenChannel.Writer.Complete(new OperationCanceledException());
            }
        }
    }
}
