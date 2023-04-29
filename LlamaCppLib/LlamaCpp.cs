using System.Reflection;
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

        public LlamaCpp(string name) => _modelName = name;

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

        public IEnumerable<LlamaToken> Tokenize(string text, bool addBos = false) =>
            LlamaCppInterop.llama_tokenize(_model, $"{(addBos ? " " : String.Empty)}{text}", addBos);

        public string Detokenize(IEnumerable<LlamaToken> vocabIds) =>
            vocabIds
                .Select(vocabId => LlamaCppInterop.llama_token_to_str(_model, vocabId))
                .DefaultIfEmpty()
                .Aggregate((a, b) => $"{a}{b}") ?? String.Empty;

        /// <summary>
        /// Load model, possibly include loading a LoRA adapter
        /// </summary>
        /// <param name="modelPath">Specify the path to the LLaMA model file</param>
        /// <param name="contextSize">The context option allows you to set the size of the prompt context used by the LLaMA models during text generation. A larger context size helps the model to better comprehend and generate responses for longer input or conversations. Set the size of the prompt context (default: 2048). The LLaMA models were built with a context of 2048, which will yield the best results on longer input/inference. However, increasing the context size beyond 2048 may lead to unpredictable results.</param>
        /// <param name="seed">The RNG seed is used to initialize the random number generator that influences the text generation process. By setting a specific seed value, you can obtain consistent and reproducible results across multiple runs with the same input and settings. This can be helpful for testing, debugging, or comparing the effects of different options on the generated text to see when they diverge. If the seed is set to a value less than or equal to 0, a random seed will be used, which will result in different outputs on each run.</param>
        /// <param name="keyValuesF16">Use 32 bit floats instead of 16 bit floats for memory key+value, allowing higher quality inference at the cost of memory.</param>
        /// <param name="loraPath">Apply a LoRA (Low-Rank Adaptation) adapter to the model (will override use_mmap=false). This allows you to adapt the pretrained model to specific tasks or domains.</param>
        /// <param name="loraBaseModelPath">Optional model to use as a base for the layers modified by the LoRA adapter. This flag is used in conjunction with the lora model path, and specifies the base model for the adaptation.</param>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="InvalidOperationException"></exception>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="Exception"></exception>
        public void Load(string modelPath, int contextSize = 2048, int seed = 0, bool keyValuesF16 = true, string? loraPath = null, string? loraBaseModelPath = null)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found \"{modelPath}\".");

            if (_model != nint.Zero)
                throw new InvalidOperationException($"Model already loaded.");

            var useLora = loraPath != null;

            if (useLora && !File.Exists(loraPath))
                throw new FileNotFoundException($"LoRA adapter file not found \"{loraPath}\".");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = contextSize;
            cparams.seed = seed;
            cparams.f16_kv = keyValuesF16;
            cparams.use_mmap = useLora ? false : cparams.use_mmap; // Override to false for LoRA

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

        public void Configure(LlamaCppOptions options) => _options = options;

        public void Configure(Action<LlamaCppOptions> configure) =>
            configure(_options);

        public async IAsyncEnumerable<KeyValuePair<LlamaToken, string>> Predict(
            List<LlamaToken> contextVocabIds,
            IEnumerable<LlamaToken> promptVocabIds,
            [EnumeratorCancellation] CancellationToken cancellationToken = default
        )
        {
            // New sampling API not fully supported yet
            // https://github.com/ggerganov/llama.cpp/commit/dd7eff57d8491792010b1002b8de6a4b54912e5c

            yield return new(0, String.Empty);
            await Task.CompletedTask;

            throw new NotImplementedException();

            //if (_model == nint.Zero)
            //    throw new InvalidOperationException("You must load a model.");

            //if (!_options.IsConfigured)
            //    throw new InvalidOperationException("You must configure the model.");

            //var sampledVocabIds = new List<int>();
            //sampledVocabIds.AddRange(promptVocabIds);

            //var endOfStream = false;
            //while (!endOfStream && !cancellationToken.IsCancellationRequested)
            //{
            //    if (contextVocabIds.Count >= LlamaCppInterop.llama_n_ctx(_model))
            //        throw new NotImplementedException($"Context rotation not yet implemented (max context reached: {contextVocabIds.Count}).");

            //    LlamaCppInterop.llama_eval(_model, sampledVocabIds, contextVocabIds.Count, _options.ThreadCount ?? 0);
            //    contextVocabIds.AddRange(sampledVocabIds);

            //    var id = LlamaCppInterop.llama_sample_top_p_top_k(
            //        _model,
            //        contextVocabIds,
            //        _options.TopK ?? 0,
            //        _options.TopP ?? 0,
            //        _options.Temperature ?? 0,
            //        _options.RepeatPenalty ?? 0
            //    );

            //    sampledVocabIds.ClearAdd(id);
            //    yield return new(id, LlamaCppInterop.llama_token_to_str(_model, id));
            //    endOfStream = id == LlamaCppInterop.llama_token_eos();
            //}
        }
    }
}
