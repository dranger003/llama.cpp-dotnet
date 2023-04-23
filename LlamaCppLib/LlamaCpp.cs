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

        public void Load(string modelPath, int contextSize = 2048, int seed = 0, bool keyValuesF16 = true)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found \"{modelPath}\".");

            if (_model != nint.Zero)
                throw new InvalidOperationException($"Model already loaded.");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = contextSize;
            cparams.seed = seed;
            cparams.f16_kv = keyValuesF16;
            _model = LlamaCppInterop.llama_init_from_file(modelPath, cparams);
        }

        public void Configure(LlamaCppOptions options) =>  _options = options;

        public void Configure(Action<LlamaCppOptions> configure) =>
            configure(_options);

        public async IAsyncEnumerable<KeyValuePair<LlamaToken, string>> Predict(
            List<LlamaToken> contextVocabIds,
            IEnumerable<LlamaToken> promptVocabIds,
            [EnumeratorCancellation] CancellationToken cancellationToken = default
        )
        {
            if (_model == nint.Zero)
                throw new InvalidOperationException("You must load a model.");

            if (!_options.IsConfigured)
                throw new InvalidOperationException("You must configure the model.");

            var sampledVocabIds = new List<int>();
            sampledVocabIds.AddRange(promptVocabIds);

            var endOfStream = false;
            while (!endOfStream && !cancellationToken.IsCancellationRequested)
            {
                if (contextVocabIds.Count >= LlamaCppInterop.llama_n_ctx(_model))
                    throw new NotImplementedException($"Context rotation not yet implemented (max context reached: {contextVocabIds.Count}).");

                LlamaCppInterop.llama_eval(_model, sampledVocabIds, contextVocabIds.Count, _options.ThreadCount ?? 0);
                contextVocabIds.AddRange(sampledVocabIds);

                var id = LlamaCppInterop.llama_sample_top_p_top_k(
                    _model,
                    contextVocabIds,
                    _options.TopK ?? 0,
                    _options.TopP ?? 0,
                    _options.Temperature ?? 0,
                    _options.RepeatPenalty ?? 0
                );

                sampledVocabIds.ClearAdd(id);
                yield return new(id, LlamaCppInterop.llama_token_to_str(_model, id));
                endOfStream = id == LlamaCppInterop.llama_token_eos();
            }

            await Task.CompletedTask;
        }
    }
}
