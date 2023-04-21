using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public class LlamaCppSession
    {
        private LlamaCpp _model;
        private string _name;

        private List<int> _contextVocabIds = new();
        private List<string> _initialContext = new();

        public LlamaCppSession(LlamaCpp model, string name)
        {
            _model = model;
            _name = name;
        }

        public string Name { get => _name; }

        public List<string> InitialContext { get => _initialContext; }

        public List<LlamaToken> TokenizedContext => _contextVocabIds;

        public string Conversation => _model.Detokenize(_contextVocabIds.Skip(1)).Substring(1); // Skip BOS

        public void Configure(Action<LlamaCppSession> configure) => configure(this);

        public void Reset() => _contextVocabIds.Clear();

        public async IAsyncEnumerable<string> Predict(string prompt, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var start = !_contextVocabIds.Any();

            prompt = $"USER:\n{prompt}\n\nASSISTANT:\n";

            if (start && _initialContext.Any())
            {
                var context = _initialContext
                    .Select((x, i) => $"{(i % 2 == 0 ? "ASSISTANT" : "USER")}:\n{x}\n")
                    .Aggregate((a, b) => $"{a}\n{b}");

                prompt = $"{context}\n{prompt}";
            }

            var promptVocabIds = _model.Tokenize(prompt, start);

            await foreach (var token in _model.Predict(_contextVocabIds, promptVocabIds, cancellationToken))
                yield return token.Value;

            _contextVocabIds.AddRange(_model.Tokenize("\n\n"));
            yield return "\n";
        }
    }
}
