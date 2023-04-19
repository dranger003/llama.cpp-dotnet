using System.Runtime.CompilerServices;
using System.Text;

namespace LlamaCppLib
{
    public class LlamaCppSession
    {
        private LlamaCpp _model;
        private string _name;

        private List<int> _contextVocabIds = new();
        private List<string> _initialContext = new();
        private List<string> _conversation = new();

        public LlamaCppSession(LlamaCpp model, string name)
        {
            _model = model;
            _name = name;
        }

        public string Name { get => _name; }

        public List<string> InitialContext { get => _initialContext; }

        public List<string> Roles { get; set; } = new();

        public IEnumerable<string> Conversation { get => _conversation; }

        public void Configure(Action<LlamaCppSession> configure) => configure(this);

        public void Reset() => _contextVocabIds.Clear();

        public async IAsyncEnumerable<string> Predict(string prompt, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (!_contextVocabIds.Any())
            {
                _conversation.Clear();
                _conversation.AddRange(_initialContext);

                prompt = $"{_initialContext.DefaultIfEmpty().Aggregate((a, b) => $"{a}\n{b}")}\n{prompt}";
            }

            _conversation.Add(prompt);

            var topic = new StringBuilder();

            await foreach (var token in _model.Predict(_contextVocabIds, $"{prompt}\n", cancellationToken))
            {
                topic.Append(token);
                yield return token;
            }

            _conversation.Add(topic.ToString().TrimEnd('\n'));
        }
    }
}
