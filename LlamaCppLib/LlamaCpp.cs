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
            cparams.seed = _options.Seed ?? 0;
            cparams.f16_kv = _options.UseHalf ?? true;
            cparams.use_mmap = useLora ? false : (_options.UseMemoryMapping ?? cparams.use_mmap);
            cparams.use_mlock = _options.UseMemoryLocking ?? false;

            _model = LlamaCppInterop.llama_load_model_from_file(modelPath, cparams);
            _modelContext = LlamaCppInterop.llama_new_context_with_model(_model, cparams);

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

            var tokens_list = options.PromptVocabIds;

            while (LlamaCppInterop.llama_get_kv_cache_token_count(_modelContext) < LlamaCppInterop.llama_n_ctx(_modelContext) && !cancellationToken.IsCancellationRequested)
            {
                LlamaCppInterop.llama_eval(
                    _modelContext,
                    tokens_list.ToArray(),
                    tokens_list.Count,
                    LlamaCppInterop.llama_get_kv_cache_token_count(_modelContext),
                    _options.ThreadCount ?? 4
                );

                tokens_list.Clear();

                var logits = LlamaCppInterop.llama_get_logits(_modelContext);
                var n_vocab = LlamaCppInterop.llama_n_vocab(_modelContext);

                var candidates = new List<LlamaCppInterop.llama_token_data>(n_vocab);
                for (LlamaToken tokenId = 0; tokenId < n_vocab; tokenId++)
                    candidates.Add(new LlamaCppInterop.llama_token_data { id = tokenId, logit = logits[tokenId], p = 0.0f });

                var candidates_p = new LlamaCppInterop.llama_token_data_array { data = candidates.ToArray(), size = (ulong)candidates.Count, sorted = false };
                var new_token_id = LlamaCppInterop.llama_sample_token_greedy(_modelContext, candidates_p);

                if (new_token_id == LlamaCppInterop.llama_token_eos())
                    break;

                var token = LlamaCppInterop.llama_token_to_str(_modelContext, new_token_id);
                tokens_list.Add(new_token_id);

                yield return new(new_token_id, token);
            }

            await Task.CompletedTask;
        }
    }
}
