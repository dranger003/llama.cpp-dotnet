using System.Runtime.CompilerServices;

namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public class LlamaCppSession
    {
        private LlamaCpp _model;
        private string _name;
        private PredictOptions _predictOptions = new();

        public LlamaCppSession(LlamaCpp model, string name)
        {
            _model = model;
            _name = name;

            Reset(); // Initialize mirostat
        }

        public string Name { get => _name; }

        public LlamaCppSessionOptions Options { get; } = new();

        public List<LlamaToken> TokenizedContext => _predictOptions.ContextVocabIds;

        public string Conversation => _model.Detokenize(_predictOptions.ContextVocabIds);

        public void Configure(Action<LlamaCppSessionOptions> configure) => configure(this.Options);

        public void Reset()
        {
            _predictOptions.ContextVocabIds.Clear();
            _predictOptions.MirostatMU = 2.0f * _model.Options.MirostatTAU ?? 0.0f;
        }

        public async IAsyncEnumerable<string> Predict(string prompt, string? context = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
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

            _predictOptions.PromptVocabIds = _model.Tokenize(prompt, false); //!_predictOptions.ContextVocabIds.Any()

            await foreach (var token in _model.Predict(_predictOptions, cancellationToken))
                yield return token.Value;

            _predictOptions.ContextVocabIds.AddRange(_model.Tokenize("\n\n"));
            yield return "\n";
        }
    }
}
