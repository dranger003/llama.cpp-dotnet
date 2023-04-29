using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public class LlamaCppSession
    {
        private LlamaCpp _model;
        private string _name;
        private List<int> _contextVocabIds = new();

        public LlamaCppSession(LlamaCpp model, string name)
        {
            _model = model;
            _name = name;
        }

        public string Name { get => _name; }

        public LlamaCppSessionOptions Options { get; } = new();

        public List<LlamaToken> TokenizedContext => _contextVocabIds;

        public string Conversation => _model.Detokenize(_contextVocabIds);

        public void Configure(Action<LlamaCppSessionOptions> configure) => configure(this.Options);

        public void Reset() => _contextVocabIds.Clear();

        public async IAsyncEnumerable<string> Predict(string prompt, string? context = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var start = !_contextVocabIds.Any();
            var template = this.Options.Template;

            if (template != null)
            {
                template = template
                    .Replace("{prompt}", "{0}")
                    .Replace("{context}", "{1}");

                if (context != null)
                    prompt = String.Format(template, prompt, context);
                else
                    prompt = String.Format(template, prompt);
            }

            var promptVocabIds = _model.Tokenize(prompt, start);

            await foreach (var token in _model.Predict(_contextVocabIds, promptVocabIds, cancellationToken))
                yield return token.Value;

            // TODO
            _contextVocabIds.AddRange(_model.Tokenize("\n\n"));
            yield return "\n";
        }
    }
}
