using LlamaCppWeb;

namespace LlamaCppLib
{
    public class LlamaCppLoader
    {
        private LlamaCppConfiguration _configuration;
        private LlamaCpp? _model;
        private List<LlamaCppSession> _sessions = new();

        public LlamaCppLoader(IConfiguration configuration)
        {
            _configuration = new LlamaCppConfiguration(configuration);
            _configuration.Load();
        }

        public IEnumerable<string> Models => _configuration.Models.Select(x => x.Name).ToList();

        public LlamaCpp? Model => _model;

        public IEnumerable<LlamaCppSession> Sessions { get => _sessions; }

        public void Load(string modelName, out string? initialContext)
        {
            initialContext = null;

            var modelIndex = _configuration.Models
                .Select((model, index) => (Model: model, Index: index))
                .Where(x => x.Model.Name == modelName)
                .Single()
                .Index;

            var modelPath = _configuration.Models[modelIndex].Path;

            // Different model requested? If so, dispose current
            if (_model != null && _model.ModelPath != modelPath)
                Unload();

            // No model loaded, load it
            if (_model == null)
            {
                _model = new LlamaCpp(modelName);
                _model.Load(modelPath);

                var modelOptions = _configuration.Models[modelIndex].Options;

                // Use model options, with fallback on global options
                _model.Configure(configure =>
                {
                    configure.ThreadCount = modelOptions.ThreadCount ?? _configuration.ThreadCount;
                    configure.TopK = modelOptions.TopK ?? _configuration.TopK;
                    configure.TopP = modelOptions.TopP ?? _configuration.TopP;
                    configure.Temperature = modelOptions.Temperature ?? _configuration.Temperature;
                    configure.RepeatPenalty = modelOptions.RepeatPenalty ?? _configuration.RepeatPenalty;
                    configure.EndOfStreamToken = modelOptions.EndOfStreamToken ?? _configuration.EndOfStreamToken;
                });
            }
        }

        public void Unload()
        {
            _model?.Dispose();
            _model = null;
        }

        public LlamaCppSession NewSession(string sessionName)
        {
            if (_model == null)
                throw new InvalidOperationException("No model loaded.");

            var session = new LlamaCppSession(_model, sessionName);
            _sessions.Add(session);

            return session;
        }

        public void DeleteSession(string sessionName) => _sessions.Remove(_sessions.Single(x => x.Name == sessionName));

        public LlamaCppSession GetSession(string sessionName) => _sessions.Single(session => session.Name == sessionName);
    }
}
