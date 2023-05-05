using LlamaCppWeb;

namespace LlamaCppLib
{
    public enum LlamaCppModelStatus { Unloaded, Loaded }

    public class LlamaCppManager
    {
        private LlamaCppConfiguration _configuration;
        private LlamaCpp? _model;
        private List<LlamaCppSession> _sessions = new();

        public LlamaCppManager(IConfiguration configuration)
        {
            _configuration = new LlamaCppConfiguration(configuration);
            _configuration.Load();
        }

        public IEnumerable<string> Models { get => _configuration.Models.Select(x => x.Name ?? String.Empty); }

        public IEnumerable<string> Sessions { get => _sessions.Select(session => session.Name); }

        public LlamaCppModelStatus Status { get => _model == null ? LlamaCppModelStatus.Unloaded : LlamaCppModelStatus.Loaded; }

        public string? ModelName { get => _model?.ModelName; }

        public void LoadModel(string modelName)
        {
            var modelIndex = _configuration.Models
                .Select((model, index) => (Model: model, Index: index))
                .Where(x => x.Model.Name == modelName)
                .Single()
                .Index;

            if (_model != null)
                UnloadModel();

            // No model loaded, load it
            if (_model == null)
            {
                var modelPath = _configuration.Models[modelIndex].Path ?? String.Empty;
                var modelOptions = _configuration.Models[modelIndex].Options;

                // Use model options, with fallback on global options
                var options = new LlamaCppOptions
                {
                    ThreadCount = modelOptions.ThreadCount ?? _configuration.ThreadCount,
                    TopK = modelOptions.TopK ?? _configuration.TopK,
                    TopP = modelOptions.TopP ?? _configuration.TopP,
                    Temperature = modelOptions.Temperature ?? _configuration.Temperature,
                    RepeatPenalty = modelOptions.RepeatPenalty ?? _configuration.RepeatPenalty,
                };

                _model = new LlamaCpp(modelName, options);
                _model.Load(modelPath);
            }
        }

        public void UnloadModel()
        {
            _model?.Dispose();
            _model = null;
        }

        public void ConfigureModel(Action<LlamaCppOptions> configure)
        {
            if (_model == null)
                throw new InvalidOperationException("No model loaded.");

            configure(_model.Options);
        }

        public LlamaCppSession CreateSession(string sessionName)
        {
            if (_model == null)
                throw new InvalidOperationException("No model loaded.");

            var session = new LlamaCppSession(_model, sessionName);
            _sessions.Add(session);

            return session;
        }

        public void DestroySession(string sessionName)
        {
            var session = _sessions.FirstOrDefault(session => session.Name == sessionName);

            if (session == null)
                return;

            _sessions.Remove(session);
        }

        public LlamaCppSession GetSession(string sessionName)
        {
            var session = _sessions.FirstOrDefault(session => session.Name == sessionName);

            if (session == null)
                throw new NullReferenceException($"No such session ({sessionName})");

            return session;
        }
    }
}
