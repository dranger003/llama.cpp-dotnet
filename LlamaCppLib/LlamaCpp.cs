using System.Runtime.CompilerServices;
using System.Text;

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

        public LlamaCppSession CreateSession(string sessionName) => new(this, sessionName);

        public IEnumerable<LlamaToken> Tokenize(string text, bool addBos = false) => LlamaCppInterop.llama_tokenize(_model, $"{(addBos ? " " : String.Empty)}{text}", addBos);
        public string Detokenize(IEnumerable<LlamaToken> vocabIds) => vocabIds.Select(vocabId => LlamaCppInterop.llama_token_to_str(_model, vocabId)).Aggregate((a, b) => $"{a}{b}");

        public LlamaToken EosToken { get => LlamaCppInterop.llama_token_eos(); }

        public void Load(string modelPath, int contextSize = 2048, int seed = 0, bool useFloat32 = false)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException($"Model file not found \"{modelPath}\".");

            if (_model != nint.Zero)
                throw new InvalidOperationException($"Model already laoded.");

            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = contextSize;
            cparams.seed = seed;
            cparams.f16_kv = !useFloat32;
            _model = LlamaCppInterop.llama_init_from_file(modelPath, cparams);
        }

        public void Configure(LlamaCppOptions options) => _options = options;

        public void Configure(Action<LlamaCppOptions> configure) => configure(_options);

        public async IAsyncEnumerable<KeyValuePair<LlamaToken, string>> Predict(
            List<LlamaToken> contextVocabIds,
            IEnumerable<LlamaToken> promptVocabIds,
            [EnumeratorCancellation] CancellationToken cancellationToken = default
        )
        {
            if (_model == nint.Zero)
                throw new InvalidOperationException("You must load a model first.");

            var sampledVocabIds = new List<int>();
            sampledVocabIds.AddRange(promptVocabIds);

            var endOfStream = false;
            while (!endOfStream && !cancellationToken.IsCancellationRequested)
            {
                LlamaCppInterop.llama_eval(_model, sampledVocabIds, contextVocabIds.Count, _options.ThreadCount);
                contextVocabIds.AddRange(sampledVocabIds);

                var id = LlamaCppInterop.llama_sample_top_p_top_k(_model, contextVocabIds, _options.TopK, _options.TopP, _options.Temperature, _options.RepeatPenalty);
                sampledVocabIds.ClearAdd(id);

                //DumpContext(contextVocabIds); // DBG

                yield return new(id, LlamaCppInterop.llama_token_to_str(_model, id));

                endOfStream = id == LlamaCppInterop.llama_token_eos();
            }

            await Task.CompletedTask;
        }

        //private bool EndOfStream(IList<int> vocabIds)
        //{
        //    if (!String.IsNullOrEmpty(_options.EndOfStreamToken))
        //    {
        //        // Indice based
        //        var eosVocabIds = LlamaCppInterop.llama_tokenize(_model, _options.EndOfStreamToken);
        //        if (vocabIds.TakeLast(eosVocabIds.Count).SequenceEqual(eosVocabIds))
        //            return true;

        //        // String based
        //        var c = vocabIds.Count;
        //        var s = new StringBuilder();

        //        while (--c >= 0 && s.Length <= _options.EndOfStreamToken.Length)
        //        {
        //            s.Insert(0, LlamaCppInterop.llama_token_to_str(_model, vocabIds[c]));

        //            if (s.Length > _options.EndOfStreamToken.Length && s.ToString(s.Length - _options.EndOfStreamToken.Length, _options.EndOfStreamToken.Length) == _options.EndOfStreamToken)
        //                return true;
        //        }
        //    }

        //    return false;
        //}

        //private void DumpContext(IEnumerable<int> contextVocabIds, string fileName = "context.txt")
        //{
        //    var sb = new StringBuilder();
        //    foreach (var t in contextVocabIds)
        //        sb.Append($"{LlamaCppInterop.llama_token_to_str(_model, t)}");
        //        //sb.Append($"[{t}]");
        //    File.WriteAllText(fileName, $"{sb}");
        //}
    }
}
