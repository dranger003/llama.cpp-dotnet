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

        public LlamaCppSession NewSession(string sessionName) => new(this, sessionName);

        public List<int> Tokenize(string text) => LlamaCppInterop.llama_tokenize(_handle, text).ToList();

        public void Load(string modelPath, int contextSize = 2048, int seed = 0, bool useFloat32 = false)
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

        public async IAsyncEnumerable<string> Predict(List<int> contextVocabIds, string prompt, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (_handle == nint.Zero)
                throw new InvalidOperationException("You must load a model first.");

            var sampledVocabIds = new List<int>();
            sampledVocabIds.AddRange(LlamaCppInterop.llama_tokenize(_handle, $"{prompt}"));

            var endOfStream = false;
            while (!endOfStream && !cancellationToken.IsCancellationRequested)
            {
                LlamaCppInterop.llama_eval(_handle, sampledVocabIds, sampledVocabIds.Count, contextVocabIds.Count, _options.ThreadCount);

                contextVocabIds.AddRange(sampledVocabIds);

                var id = LlamaCppInterop.llama_sample_top_p_top_k(
                    _handle,
                    contextVocabIds,
                    contextVocabIds.Count,
                    _options.TopK,
                    _options.TopP,
                    _options.Temperature,
                    _options.RepeatPenalty
                );

                sampledVocabIds.Clear();
                sampledVocabIds.Add(id);

                endOfStream = EndOfStream(contextVocabIds);

                if (endOfStream)
                {
                    var newLineVocabId = LlamaCppInterop.llama_tokenize(_handle, "\n");
                    contextVocabIds.AddRange(newLineVocabId);
                    sampledVocabIds.AddRange(newLineVocabId);
                }

                //DumpContext(contextVocabIds); // DBG

                var token = LlamaCppInterop.llama_token_to_str(_handle, id);
                yield return token;

                if (endOfStream && token != "\n")
                    yield return "\n";
            }

            await Task.CompletedTask;
        }

        private bool EndOfStream(IList<int> vocabIds)
        {
            // Vector based
            var eosVocabIds = LlamaCppInterop.llama_tokenize(_handle, _options.EndOfStreamToken);
            if (vocabIds.TakeLast(eosVocabIds.Length).SequenceEqual(eosVocabIds))
                return true;

            // String based
            var c = vocabIds.Count;
            var s = new StringBuilder();

            while (--c >= 0 && s.Length <= _options.EndOfStreamToken.Length)
            {
                s.Insert(0, LlamaCppInterop.llama_token_to_str(_handle, vocabIds[c]));

                if (s.Length > _options.EndOfStreamToken.Length && s.ToString(s.Length - _options.EndOfStreamToken.Length, _options.EndOfStreamToken.Length) == _options.EndOfStreamToken)
                    return true;
            }

            return false;
        }

        //private void DumpContext(IEnumerable<int> contextVocabIds, string fileName = "context.txt")
        //{
        //    var sb = new StringBuilder();
        //    foreach (var t in contextVocabIds)
        //        sb.Append($"{LlamaCppInterop.llama_token_to_str(_handle, t)}");
        //    File.WriteAllText(fileName, $"{sb}");
        //}
    }
}
