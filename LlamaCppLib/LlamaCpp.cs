using System.Runtime.CompilerServices;
using System.Text;

namespace LlamaCppLib
{
    public class LlamaCpp : IDisposable
    {
        private nint _handle = nint.Zero;
        private string _modelName;
        private string _modelPath = string.Empty;
        private LlamaCppOptions _options = new();

        public LlamaCpp(string name) => _modelName = name;

        public void Dispose()
        {
            if (_handle != nint.Zero)
            {
                LlamaCppInterop.llama_free(_handle);
                _handle = nint.Zero;
            }
        }

        public string ModelName { get => _modelName; }
        public string ModelPath { get => _modelPath; }

        public void Load(string modelPath, int contextSize = 2048, int seed = 0, bool useFloat32 = true)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found \"{modelPath}\".");

            if (_handle != nint.Zero)
                throw new InvalidOperationException($"Model already laoded.");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = contextSize;
            cparams.n_parts = -1;
            cparams.seed = seed;
            cparams.f16_kv = !useFloat32;
            cparams.use_mlock = false;
            _handle = LlamaCppInterop.llama_init_from_file(modelPath, cparams);

            _modelPath = modelPath;
        }

        public void Configure(LlamaCppOptions options) => _options = options;

        public void Configure(Action<LlamaCppOptions> configure) => configure(_options);

        public async IAsyncEnumerable<string> Predict(
            StringBuilder context,
            string prompt,
            bool updateContext = true,
            [EnumeratorCancellation] CancellationToken cancellationToken = default
        )
        {
            if (_handle == nint.Zero)
                throw new InvalidOperationException("You must load a model first.");

            if (!String.IsNullOrEmpty(_options.InstructionPrompt))
                prompt = $"{_options.InstructionPrompt} {prompt}";

            if (updateContext)
                context.AppendNewLineIfMissing().Append(prompt);

            var contextVocabIds = new List<int>();
            var sampledVocabIds = new List<int>();

            contextVocabIds.AddRange(LlamaCppInterop.llama_tokenize(_handle, $"{context}"));
            contextVocabIds.AddRange(LlamaCppInterop.llama_tokenize(_handle, $"{prompt}"));

            sampledVocabIds.AddRange(contextVocabIds);

            var evaluatedVocabIdCount = 0;

            var instructionPromptVocabIds = LlamaCppInterop.llama_tokenize(_handle, _options.InstructionPrompt);

            while (true)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                LlamaCppInterop.llama_eval(_handle, sampledVocabIds, sampledVocabIds.Count, evaluatedVocabIdCount, _options.ThreadCount);

                evaluatedVocabIdCount += sampledVocabIds.Count;
                sampledVocabIds.Clear();

                var id = LlamaCppInterop.llama_sample_top_p_top_k(
                    _handle,
                    contextVocabIds,
                    contextVocabIds.Count,
                    _options.TopK,
                    _options.TopP,
                    _options.Temperature,
                    _options.RepeatPenalty
                );

                contextVocabIds.Add(id);
                sampledVocabIds.Add(id);

                var token = LlamaCppInterop.llama_token_to_str(_handle, id);

                if (updateContext)
                    context.Append(token);

                yield return token;

                if (!_options.IgnoreEndOfStream && id == LlamaCppInterop.llama_token_eos())
                    break;

                if (_options.StopOnInstructionPrompt && contextVocabIds.TakeLast(instructionPromptVocabIds.Length).SequenceEqual(instructionPromptVocabIds))
                    break;
            }

            await Task.CompletedTask;
        }
    }
}
