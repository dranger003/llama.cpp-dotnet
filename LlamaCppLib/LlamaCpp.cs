using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaContext = System.IntPtr;
    using LlamaToken = System.Int32;

    public class LlamaCpp : IDisposable
    {
        private LlamaContext _model;
        private string _modelName;
        private LlamaCppOptions _options = new();

        public LlamaCpp(string name, LlamaCppOptions options)
        {
            _modelName = name;
            _options = options;
        }

        public void Dispose()
        {
            if (_model != nint.Zero)
            {
                LlamaCppInterop.llama_free(_model);
                _model = nint.Zero;
            }
        }

        public string ModelName { get => _modelName; }

        public LlamaCppOptions Options { get => _options; }

        public LlamaCppSession CreateSession(string sessionName) =>
            new(this, sessionName);

        public List<LlamaToken> Tokenize(string text, bool addBos = false)
        {
            var tokens = new LlamaToken[_options.ContextSize ?? 1];
            var count = LlamaCppInterop.llama_tokenize(_model, text, tokens, tokens.Length, addBos);
            return new(tokens.Take(count));
        }

        public string Detokenize(IEnumerable<LlamaToken> vocabIds) =>
            vocabIds
                .Select(vocabId => LlamaCppInterop.llama_token_to_str(_model, vocabId))
                .DefaultIfEmpty()
                .Aggregate((a, b) => $"{a}{b}") ?? String.Empty;

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

            if (_model != nint.Zero)
                throw new InvalidOperationException($"Model already loaded.");

            var useLora = loraPath != null;

            if (useLora && !File.Exists(loraPath))
                throw new FileNotFoundException($"LoRA adapter file not found \"{loraPath}\".");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = _options.ContextSize ?? 512;
            cparams.n_gpu_layers = _options.GpuLayers ?? 0;
            cparams.seed = _options.Seed ?? 0;
            cparams.f16_kv = _options.UseHalf ?? true;
            cparams.logits_all = false;
            cparams.vocab_only = false;
            cparams.use_mmap = useLora ? false : (_options.UseMemoryMapping ?? cparams.use_mmap);
            cparams.use_mlock = _options.UseMemoryLocking ?? false;
            cparams.embedding = false;

            _model = LlamaCppInterop.llama_init_from_file(modelPath, cparams);

            if (useLora)
            {
                if (loraBaseModelPath == null)
                    loraBaseModelPath = modelPath;

                if (loraPath == null)
                    throw new ArgumentNullException(nameof(loraPath));

                if (loraBaseModelPath == null)
                    throw new ArgumentNullException(loraPath, nameof(loraBaseModelPath));

                var result = LlamaCppInterop.llama_apply_lora_from_file(_model, loraPath, loraBaseModelPath, Options.ThreadCount ?? 0);
                if (result != 0)
                    throw new Exception($"Unable to load LoRA file (return code: {result}).");
            }
        }

        public async IAsyncEnumerable<KeyValuePair<LlamaToken, string>> Predict(PredictOptions options, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (_model == nint.Zero)
                throw new InvalidOperationException("You must load a model.");

            if (!_options.IsConfigured)
                throw new InvalidOperationException("You must configure the model.");

            var sampledVocabIds = new List<int>();
            sampledVocabIds.AddRange(options.PromptVocabIds);

            var endOfStream = false;
            while (!endOfStream && !cancellationToken.IsCancellationRequested)
            {
                if (options.ContextVocabIds.Count >= LlamaCppInterop.llama_n_ctx(_model))
                    throw new NotImplementedException($"Context rotation not yet implemented (max context reached: {options.ContextVocabIds.Count}).");

                LlamaCppInterop.llama_eval(_model, sampledVocabIds.ToArray(), sampledVocabIds.Count, options.ContextVocabIds.Count, _options.ThreadCount ?? 1);
                options.ContextVocabIds.AddRange(sampledVocabIds);

                var mirostatMU = options.MirostatMU;
                var id = Sample(options.ContextVocabIds, ref mirostatMU);
                options.MirostatMU = mirostatMU;

                sampledVocabIds.ClearAdd(id);
                yield return new(id, LlamaCppInterop.llama_token_to_str(_model, id));
                endOfStream = id == LlamaCppInterop.llama_token_eos();
            }

            await Task.CompletedTask;
        }

        private LlamaToken Sample(List<LlamaToken> contextVocabIds, ref float mirostatMU)
        {
            var logits = LlamaCppInterop.llama_get_logits(_model);
            var vocabCount = LlamaCppInterop.llama_n_vocab(_model);

            // Apply logit biases
            foreach (var logit in _options.LogitBias)
                logits[logit.Key] += logit.Value;

            var candidates = new List<LlamaCppInterop.llama_token_data>(vocabCount);
            for (LlamaToken tokenId = 0; tokenId < vocabCount; tokenId++)
                candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });

            var candidates_p = new LlamaCppInterop.llama_token_data_array
            {
                data = candidates.ToArray(),
                size = (ulong)candidates.Count,
                sorted = false
            };

            // Apply penalties
            var nl_logit = logits[LlamaCppInterop.llama_token_nl()];

            LlamaCppInterop.llama_sample_repetition_penalty(_model, candidates_p, contextVocabIds, _options.RepeatPenalty ?? 0);
            LlamaCppInterop.llama_sample_frequency_and_presence_penalties(_model, candidates_p, contextVocabIds, _options.FrequencyPenalty ?? 0, _options.PresencePenalty ?? 0);

            if (!(_options.PenalizeNewLine ?? true))
                logits[LlamaCppInterop.llama_token_nl()] = nl_logit;

            var id = default(LlamaToken);

            if ((_options.Temperature ?? 0) <= 0)
            {
                // Greedy sampling
                id = LlamaCppInterop.llama_sample_token_greedy(_model, candidates_p);
            }
            else
            {
                if (_options.Mirostat == Mirostat.Mirostat)
                {
                    var mirostat_m = 100;
                    LlamaCppInterop.llama_sample_temperature(_model, candidates_p, _options.Temperature ?? 0);
                    id = LlamaCppInterop.llama_sample_token_mirostat(_model, candidates_p, _options.MirostatTAU ?? 0, _options.MirostatETA ?? 0, mirostat_m, ref mirostatMU);
                }
                else if (_options.Mirostat == Mirostat.Mirostat2)
                {
                    LlamaCppInterop.llama_sample_temperature(_model, candidates_p, _options.Temperature ?? 0);
                    id = LlamaCppInterop.llama_sample_token_mirostat_v2(_model, candidates_p, _options.MirostatTAU ?? 0, _options.MirostatETA ?? 0, ref mirostatMU);
                }
                else
                {
                    // Temperature sampling
                    LlamaCppInterop.llama_sample_top_k(_model, candidates_p, _options.TopK ?? 0);
                    LlamaCppInterop.llama_sample_tail_free(_model, candidates_p, _options.TfsZ ?? 0);
                    LlamaCppInterop.llama_sample_typical(_model, candidates_p, _options.TypicalP ?? 0);
                    LlamaCppInterop.llama_sample_top_p(_model, candidates_p, _options.TopP ?? 0);
                    LlamaCppInterop.llama_sample_temperature(_model, candidates_p, _options.Temperature ?? 0);
                    id = LlamaCppInterop.llama_sample_token(_model, candidates_p);
                }
            }

            return id;
        }
    }
}
