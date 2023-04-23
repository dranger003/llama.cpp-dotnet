using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public class LlamaCppSession
    {
        private LlamaCpp _model;
        private string _name;
        private List<int> _contextVocabIds = new();
        private LlamaCppSessionOptions _options = new();

        public LlamaCppSession(LlamaCpp model, string name)
        {
            _model = model;
            _name = name;
        }

        public string Name { get => _name; }

        public List<LlamaToken> TokenizedContext => _contextVocabIds;

        public string Conversation => _model.Detokenize(_contextVocabIds);

        public void Configure(Action<LlamaCppSessionOptions> configure) => configure(_options);

        public void Reset() => _contextVocabIds.Clear();

        public async IAsyncEnumerable<string> Predict(string prompt, string? context = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var start = !_contextVocabIds.Any();

            if (_options.Template != null)
            {
                if (context != null)
                    prompt = String.Format(_options.Template, prompt, context);
                else
                    prompt = String.Format(_options.Template, prompt);

                //prompt = $"USER:\n{prompt}\n\nASSISTANT:\n";
                //prompt = $"### Instruction:\n{prompt}\n\n### Response:\n";
            }

            var promptVocabIds = _model.Tokenize(prompt, start);

            await foreach (var token in _model.Predict(_contextVocabIds, promptVocabIds, cancellationToken))
                yield return token.Value;

            _contextVocabIds.AddRange(_model.Tokenize("\n\n"));
            yield return "\n";
        }
    }
}
