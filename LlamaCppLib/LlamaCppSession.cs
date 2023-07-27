namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    internal class LlamaCppSessionState
    {
        public List<LlamaToken> TokenIds { get; } = new();
        public int EvalOffset { get; set; } = 0;
    }

    public class LlamaCppSession
    {
        private LlamaCppModel _model;
        private LlamaCppSessionState _state = new();
        private Guid _id = Guid.NewGuid();

        private static LlamaCppSession? _lastSessionToGenerate;

        internal LlamaCppSession(LlamaCppModel model) => _model = model;

        public Guid Id => _id;

        public void Reset()
        {
            _state.TokenIds.Clear();
            _state.EvalOffset = 0;
        }

        public string GetContextAsText() => _model.UntokenizeToText(_state.TokenIds);

        public IAsyncEnumerable<byte[]> GenerateBytesAsync(string prompt, LlamaCppGenerateOptions? options = default, CancellationToken cancellationToken = default)
        {
            if (_lastSessionToGenerate != null && _lastSessionToGenerate != this)
                _state.EvalOffset = 0;

            _lastSessionToGenerate = this;

            if (options == default)
                options = new();

            _state.TokenIds.AddRange(_model.Tokenize(prompt, !_state.TokenIds.Any()));
            return _model.GenerateTokenBytesAsync(options, _state, cancellationToken);
        }

        public IAsyncEnumerable<byte[]> GenerateBytesAsync(string prompt, CancellationToken cancellationToken = default) =>
            GenerateBytesAsync(prompt, default, cancellationToken);

        public IAsyncEnumerable<string> GenerateStringAsync(string prompt, LlamaCppGenerateOptions? options = default, CancellationToken cancellationToken = default)
        {
            if (_lastSessionToGenerate != null && _lastSessionToGenerate != this)
                _state.EvalOffset = 0;

            _lastSessionToGenerate = this;

            if (options == default)
                options = new();

            var tokens = _model.Tokenize(prompt, !_state.TokenIds.Any());
            _state.TokenIds.AddRange(tokens);

            return _model.GenerateTokenStringAsync(options, _state, cancellationToken);
        }

        public IAsyncEnumerable<string> GenerateStringAsync(string prompt, CancellationToken cancellationToken = default) =>
            GenerateStringAsync(prompt, default, cancellationToken);
    }
}
