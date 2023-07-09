using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaModel = System.IntPtr;
    using LlamaContext = System.IntPtr;
    using LlamaToken = System.Int32;

    public class LlamaCpp : IDisposable
    {
        private LlamaModel _model;
        private LlamaContext _modelContext;
        private string _modelName;
        private LlamaCppOptions _options = new();
        private byte[]? _state;

        public LlamaCpp(string name, LlamaCppOptions options)
        {
            _modelName = name;
            _options = options;
        }

        public void Dispose()
        {
            if (_model != IntPtr.Zero)
            {
                LlamaCppInterop.llama_free(_modelContext);
                _modelContext = IntPtr.Zero;
                LlamaCppInterop.llama_free_model(_model);
                _model = IntPtr.Zero;
            }
        }

        public LlamaContext Handle => _modelContext;

        public string ModelName { get => _modelName; }

        public LlamaCppOptions Options { get => _options; }

        public List<LlamaToken> Tokenize(string text, bool addBos = false)
        {
            var tokens = new LlamaToken[_options.ContextSize ?? 1];
            var count = LlamaCppInterop.llama_tokenize(_modelContext, text, tokens, tokens.Length, addBos);
            return new(tokens.Take(count));
        }

        public string Detokenize(IEnumerable<LlamaToken> vocabIds) => vocabIds.Any() ?
            vocabIds
                .Select(vocabId => LlamaCppInterop.llama_token_to_str(_modelContext, vocabId))
                .Aggregate((a, b) => $"{a}{b}") : String.Empty;

        public void ResetState()
        {
            if (_state == null)
                return;

            LlamaCppInterop.llama_set_state_data(_modelContext, _state);
        }

        /// <summary>
        /// Load a model and optionally load a LoRA adapter
        /// </summary>
        /// <param name="modelPath">Specify the path to the LLaMA model file</param>
        /// <param name="contextSize">The context option allows you to set the size of the prompt context used by the LLaMA models during text generation. A larger context size helps the model to better comprehend and generate responses for longer input or conversations. Set the size of the prompt context (default: 2048). The LLaMA models were built with a context of 2048, which will yield the best results on longer input/inference. However, increasing the context size beyond 2048 may lead to unpredictable results.</param>
        /// <param name="seed">The RNG seed is used to initialize the random number generator that influences the text generation process. By setting a specific seed value, you can obtain consistent and reproducible results across multiple runs with the same input and settings. This can be helpful for testing, debugging, or comparing the effects of different options on the generated text to see when they diverge. If the seed is set to a value less than or equal to 0, a random seed will be used, which will result in different outputs on each run.</param>
        /// <param name="loraPath">Apply a LoRA (Low-Rank Adaptation) adapter to the model (will override use_mmap=false). This allows you to adapt the pretrained model to specific tasks or domains.</param>
        /// <param name="loraBaseModelPath">Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the lora model path, and specifies the base model for the adaptation.</param>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="InvalidOperationException"></exception>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="Exception"></exception>
        public void Load(string modelPath, string? loraPath = null, string? loraBaseModelPath = null)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found \"{modelPath}\".");

            if (_modelContext != IntPtr.Zero)
                throw new InvalidOperationException($"Model already loaded.");

            var useLora = loraPath != null;

            if (useLora && !File.Exists(loraPath))
                throw new FileNotFoundException($"LoRA adapter file not found \"{loraPath}\".");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = _options.ContextSize ?? 512;
            cparams.n_batch = 512;
            cparams.n_gpu_layers = _options.GpuLayers ?? 0;
            cparams.seed = _options.Seed ?? unchecked((uint)-1);
            cparams.f16_kv = _options.UseHalf ?? true;
            cparams.use_mmap = useLora ? false : (_options.UseMemoryMapping ?? cparams.use_mmap);
            cparams.use_mlock = _options.UseMemoryLocking ?? false;

            _model = LlamaCppInterop.llama_load_model_from_file(modelPath, cparams);
            _modelContext = LlamaCppInterop.llama_new_context_with_model(_model, cparams);
            _state = LlamaCppInterop.llama_copy_state_data(_modelContext);

            if (useLora)
            {
                if (loraBaseModelPath == null)
                    loraBaseModelPath = modelPath;

                if (loraPath == null)
                    throw new ArgumentNullException(nameof(loraPath));

                if (loraBaseModelPath == null)
                    throw new ArgumentNullException(loraPath, nameof(loraBaseModelPath));

                var result = LlamaCppInterop.llama_model_apply_lora_from_file(_model, loraPath, loraBaseModelPath, Options.ThreadCount ?? 0);
                if (result != 0)
                    throw new Exception($"Unable to load LoRA file (return code: {result}).");
            }
        }

        public async IAsyncEnumerable<KeyValuePair<LlamaToken, string>> Predict(PredictOptions options, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (_modelContext == IntPtr.Zero)
                throw new InvalidOperationException("You must load a model.");

            if (!_options.IsConfigured)
                throw new InvalidOperationException("You must configure the model.");

            var embd = options.PromptVocabIds;

            var n_ctx = LlamaCppInterop.llama_n_ctx(_modelContext);
            var last_n_tokens = new List<LlamaToken>(n_ctx);

            var n_threads = _options.ThreadCount ?? 4;
            var n_batch = 512;
            var n_past = 0;

            var top_k = _options.TopK ?? 40;
            var top_p = _options.TopP ?? 0.95f;
            var tfs_z = _options.TfsZ ?? 1.0f;
            var typical_p = _options.TypicalP ?? 1.0f;
            var temp = _options.Temperature ?? 0.8f;
            var repeat_penalty = _options.RepeatPenalty ?? 1.1f;
            var repeat_last_n = _options.LastTokenCountPenalty ?? 64;
            var frequency_penalty = _options.FrequencyPenalty ?? 0.0f;
            var presence_penalty = _options.PresencePenalty ?? 0.0f;
            var mirostat_tau = _options.MirostatTAU ?? 5.0f;
            var mirostat_eta = _options.MirostatETA ?? 0.1f;

            while (LlamaCppInterop.llama_get_kv_cache_token_count(_modelContext) < LlamaCppInterop.llama_n_ctx(_modelContext) && !cancellationToken.IsCancellationRequested)
            {
                for (var i = 0; i < embd.Count; i += n_batch)
                {
                    var n_eval = embd.Count - i;
                    if (n_eval > n_batch) n_eval = n_batch;
                    LlamaCppInterop.llama_eval(_modelContext, embd.Skip(i).ToArray(), n_eval, n_past, n_threads);
                    n_past += n_eval;
                }

                embd.Clear();

                var logits = LlamaCppInterop.llama_get_logits(_modelContext);
                var n_vocab = LlamaCppInterop.llama_n_vocab(_modelContext);

                var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);
                for (LlamaToken tokenId = 0; tokenId < n_vocab; tokenId++)
                    candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });

                var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates.ToArray(), size = (ulong)candidates.Count, sorted = false };

                { // Apply penalties
                    var nl_logit = logits[LlamaCppInterop.llama_token_nl()];
                    var last_n_repeat = Math.Min(Math.Min(last_n_tokens.Count, repeat_last_n), n_ctx);

                    LlamaCppInterop.llama_sample_repetition_penalty(
                        _modelContext,
                        candidates_p,
                        last_n_tokens.Skip(last_n_tokens.Count - last_n_repeat).Take(last_n_repeat).ToList(),
                        repeat_penalty
                    );

                    LlamaCppInterop.llama_sample_frequency_and_presence_penalties(
                        _modelContext,
                        candidates_p,
                        last_n_tokens.Skip(last_n_tokens.Count - last_n_repeat).Take(last_n_repeat).ToList(),
                        frequency_penalty,
                        presence_penalty
                    );

                    if (!_options.PenalizeNewLine ?? false)
                        logits[LlamaCppInterop.llama_token_nl()] = nl_logit;
                }

                var id = default(LlamaToken);

                { // Sampling
                    if (temp <= 0.0f)
                    {
                        // Greedy
                        id = LlamaCppInterop.llama_sample_token_greedy(_modelContext, candidates_p);
                    }
                    else
                    {
                        // Mirostat
                        if (_options.Mirostat == Mirostat.Mirostat)
                        {
                            var mirostat_mu = 2.0f * mirostat_tau;
                            var mirostat_m = 100;
                            LlamaCppInterop.llama_sample_temperature(_modelContext, candidates_p, temp);
                            id = LlamaCppInterop.llama_sample_token_mirostat(_modelContext, candidates_p, mirostat_tau, mirostat_eta, mirostat_m, ref mirostat_mu);
                        }
                        // Mirostat2
                        else if (_options.Mirostat == Mirostat.Mirostat2)
                        {
                            var mirostat_mu = 2.0f * mirostat_tau;
                            LlamaCppInterop.llama_sample_temperature(_modelContext, candidates_p, _options.Temperature ?? 1.0f);
                            id = LlamaCppInterop.llama_sample_token_mirostat_v2(_modelContext, candidates_p, mirostat_tau, mirostat_eta, ref mirostat_mu);
                        }
                        // Temperature
                        else
                        {
                            LlamaCppInterop.llama_sample_top_k(_modelContext, candidates_p, top_k, 1);
                            LlamaCppInterop.llama_sample_tail_free(_modelContext, candidates_p, tfs_z, 1);
                            LlamaCppInterop.llama_sample_typical(_modelContext, candidates_p, typical_p, 1);
                            LlamaCppInterop.llama_sample_top_p(_modelContext, candidates_p, top_p, 1);
                            LlamaCppInterop.llama_sample_temperature(_modelContext, candidates_p, temp);
                            id = LlamaCppInterop.llama_sample_token(_modelContext, candidates_p);
                        }
                    }
                }

                if (id == LlamaCppInterop.llama_token_eos())
                    break;

                if (last_n_tokens.Any())
                    last_n_tokens.RemoveAt(0);

                last_n_tokens.Add(id);
                embd.Add(id);

                yield return new(id, LlamaCppInterop.llama_token_to_str(_modelContext, id));
            }

            await Task.CompletedTask;
        }
    }
}
