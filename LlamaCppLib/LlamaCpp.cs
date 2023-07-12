using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaModel = System.IntPtr;
    using LlamaContext = System.IntPtr;
    using LlamaToken = System.Int32;

    public class LlamaCpp : IDisposable
    {
        private LlamaModel _model;
        private LlamaContext _context;
        private LlamaCppModelOptions _options = new();
        private byte[]? _state;

        public LlamaCpp()
        { }

        public void Dispose()
        {
            if (_context != IntPtr.Zero)
            {
                LlamaCppInterop.llama_free(_context);
                _context = IntPtr.Zero;
            }

            if (_model != IntPtr.Zero)
            {
                LlamaCppInterop.llama_free_model(_model);
                _model = IntPtr.Zero;
            }
        }

        public LlamaContext Handle => _context;

        public LlamaCppModelOptions Options { get => _options; }

        public List<LlamaToken> Tokenize(string text, bool addBos = false)
        {
            var tokens = new LlamaToken[LlamaCppInterop.llama_n_ctx(_context)];
            var count = LlamaCppInterop.llama_tokenize(_context, text, tokens, tokens.Length, addBos);
            return new(tokens.Take(count));
        }

        public string Detokenize(IEnumerable<LlamaToken> vocabIds)
        {
            if (!vocabIds.Any())
                return String.Empty;

            return vocabIds
                .Select(vocabId => LlamaCppInterop.llama_token_to_str(_context, vocabId))
                .Aggregate((a, b) => $"{a}{b}");
        }

        public void ResetState()
        {
            if (_state == null)
                return;

            LlamaCppInterop.llama_set_state_data(_context, _state);
        }

        /// <summary>
        /// Load a model and optionally load a LoRA adapter
        /// </summary>
        /// <param name="modelPath">Specify the path to the LLaMA model file</param>
        /// <param name="options">Model options</param>
        /// <param name="loraPath">Apply a LoRA (Low-Rank Adaptation) adapter to the model (will override use_mmap=false). This allows you to adapt the pretrained model to specific tasks or domains.</param>
        /// <param name="loraBaseModelPath">Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the lora model path, and specifies the base model for the adaptation.</param>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="InvalidOperationException"></exception>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="Exception"></exception>
        public void Load(string modelPath, LlamaCppModelOptions options, string? loraPath = null, string? loraBaseModelPath = null)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found \"{modelPath}\".");

            if (_context != IntPtr.Zero)
                throw new InvalidOperationException($"Model already loaded.");

            var useLora = loraPath != null;

            if (useLora && !File.Exists(loraPath))
                throw new FileNotFoundException($"LoRA adapter file not found \"{loraPath}\".");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = options.ContextSize;
            cparams.n_batch = 512;
            cparams.n_gpu_layers = options.GpuLayers;
            cparams.seed = options.Seed;
            cparams.f16_kv = options.UseHalf;
            cparams.use_mmap = useLora ? false : options.UseMemoryMapping;
            cparams.use_mlock = options.UseMemoryLocking;

            _model = LlamaCppInterop.llama_load_model_from_file(modelPath, cparams);
            _context = LlamaCppInterop.llama_new_context_with_model(_model, cparams);
            _state = LlamaCppInterop.llama_copy_state_data(_context);
            _options = options;

            if (useLora)
            {
                if (loraBaseModelPath == null)
                    loraBaseModelPath = modelPath;

                if (loraPath == null)
                    throw new ArgumentNullException(nameof(loraPath));

                if (loraBaseModelPath == null)
                    throw new ArgumentNullException(loraPath, nameof(loraBaseModelPath));

                var result = LlamaCppInterop.llama_model_apply_lora_from_file(_model, loraPath, loraBaseModelPath, 2);
                if (result != 0)
                    throw new Exception($"Unable to load LoRA file (return code: {result}).");
            }
        }

        public async IAsyncEnumerable<KeyValuePair<LlamaToken, string>> Predict(LlamaCppPredictOptions options, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (_context == IntPtr.Zero)
                throw new InvalidOperationException("You must load a model.");

            if (options.ResetState)
                ResetState();

            var embd = Tokenize(options.Prompt, true);

            var n_ctx = LlamaCppInterop.llama_n_ctx(_context);
            var last_n_tokens = new List<LlamaToken>(n_ctx);

            var n_threads = options.ThreadCount;
            var n_batch = 512;
            var n_past = 0;

            var top_k = options.TopK;
            var top_p = options.TopP;
            var tfs_z = options.TfsZ;
            var typical_p = options.TypicalP;
            var temp = options.Temperature;
            var repeat_penalty = options.RepeatPenalty;
            var repeat_last_n = options.LastTokenCountPenalty;
            var frequency_penalty = options.FrequencyPenalty;
            var presence_penalty = options.PresencePenalty;
            var mirostat_tau = options.MirostatTAU;
            var mirostat_eta = options.MirostatETA;

            var mirostat_mu = 2.0f * mirostat_tau;

            while (LlamaCppInterop.llama_get_kv_cache_token_count(_context) < LlamaCppInterop.llama_n_ctx(_context) && !cancellationToken.IsCancellationRequested)
            {
                for (var i = 0; i < embd.Count; i += n_batch)
                {
                    var n_eval = embd.Count - i;
                    if (n_eval > n_batch) n_eval = n_batch;
                    LlamaCppInterop.llama_eval(_context, embd.Skip(i).ToArray(), n_eval, n_past, n_threads);
                    n_past += n_eval;
                }

                embd.Clear();

                var logits = LlamaCppInterop.llama_get_logits(_context);
                var n_vocab = LlamaCppInterop.llama_n_vocab(_context);

                var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);
                for (LlamaToken tokenId = 0; tokenId < n_vocab; tokenId++)
                    candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });

                var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates.ToArray(), size = (ulong)candidates.Count, sorted = false };

                // Apply penalties
                {
                    var nl_logit = logits[LlamaCppInterop.llama_token_nl()];
                    var last_n_repeat = Math.Min(Math.Min(last_n_tokens.Count, repeat_last_n), n_ctx);

                    LlamaCppInterop.llama_sample_repetition_penalty(
                        _context,
                        candidates_p,
                        last_n_tokens.Skip(last_n_tokens.Count - last_n_repeat).Take(last_n_repeat).ToList(),
                        repeat_penalty
                    );

                    LlamaCppInterop.llama_sample_frequency_and_presence_penalties(
                        _context,
                        candidates_p,
                        last_n_tokens.Skip(last_n_tokens.Count - last_n_repeat).Take(last_n_repeat).ToList(),
                        frequency_penalty,
                        presence_penalty
                    );

                    if (!options.PenalizeNewLine)
                        logits[LlamaCppInterop.llama_token_nl()] = nl_logit;
                }

                var id = default(LlamaToken);

                // Sampling
                {
                    if (temp <= 0.0f)
                    {
                        // Greedy
                        id = LlamaCppInterop.llama_sample_token_greedy(_context, candidates_p);
                    }
                    else
                    {
                        // Mirostat
                        if (options.Mirostat == Mirostat.Mirostat)
                        {
                            var mirostat_m = 100;
                            LlamaCppInterop.llama_sample_temperature(_context, candidates_p, temp);
                            id = LlamaCppInterop.llama_sample_token_mirostat(_context, candidates_p, mirostat_tau, mirostat_eta, mirostat_m, ref mirostat_mu);
                        }
                        // Mirostat2
                        else if (options.Mirostat == Mirostat.Mirostat2)
                        {
                            LlamaCppInterop.llama_sample_temperature(_context, candidates_p, temp);
                            id = LlamaCppInterop.llama_sample_token_mirostat_v2(_context, candidates_p, mirostat_tau, mirostat_eta, ref mirostat_mu);
                        }
                        // Temperature
                        else
                        {
                            LlamaCppInterop.llama_sample_top_k(_context, candidates_p, top_k, 1);
                            LlamaCppInterop.llama_sample_tail_free(_context, candidates_p, tfs_z, 1);
                            LlamaCppInterop.llama_sample_typical(_context, candidates_p, typical_p, 1);
                            LlamaCppInterop.llama_sample_top_p(_context, candidates_p, top_p, 1);
                            LlamaCppInterop.llama_sample_temperature(_context, candidates_p, temp);
                            id = LlamaCppInterop.llama_sample_token(_context, candidates_p);
                        }
                    }
                }

                if (id == LlamaCppInterop.llama_token_eos())
                    break;

                if (last_n_tokens.Any())
                    last_n_tokens.RemoveAt(0);

                last_n_tokens.Add(id);
                embd.Add(id);

                var token = LlamaCppInterop.llama_token_to_str(_context, id);
                yield return new(id, token);
            }

            await Task.CompletedTask;
        }
    }
}
