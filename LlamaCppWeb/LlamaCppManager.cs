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

        public IEnumerable<string> Models => _configuration.Models.Select(x => x.Name);

        public IEnumerable<string> Sessions => _sessions.Select(session => session.Name);

        public LlamaCppModelStatus Status => _model == null ? LlamaCppModelStatus.Unloaded : LlamaCppModelStatus.Loaded;

        public void LoadModel(string modelName)
        {
            var modelIndex = _configuration.Models
                .Select((model, index) => (Model: model, Index: index))
                .Where(x => x.Model.Name == modelName)
                .Single()
                .Index;

            var modelPath = _configuration.Models[modelIndex].Path;

            // Different model requested? If so, dispose current
            if (_model != null && _model.ModelPath != modelPath)
                UnloadModel();

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

        public void UnloadModel()
        {
            _model?.Dispose();
            _model = null;
        }

        public LlamaCppSession CreateSession(string sessionName)
        {
            if (_model == null)
                throw new InvalidOperationException("No model loaded.");

            var session = new LlamaCppSession(_model, sessionName);
            _sessions.Add(session);

            return session;
        }

        public void DestroySession(string sessionName) => _sessions.Remove(_sessions.Single(x => x.Name == sessionName));

        public void ConfigureSession(string sessionName, List<string> initialContext) => GetSession(sessionName).Configure(options => options.InitialContext.AddRange(initialContext));

        public LlamaCppSession GetSession(string sessionName) => _sessions.Single(session => session.Name == sessionName);
    }
}
