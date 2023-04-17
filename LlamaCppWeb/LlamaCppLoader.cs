using LlamaCppWeb;

namespace LlamaCppLib
{
    public class LlamaCppLoader
    {
        private LlamaCppConfiguration _configuration;
        private LlamaCpp? _model;

        public LlamaCppLoader(IConfiguration configuration)
        {
            _configuration = new LlamaCppConfiguration(configuration);
            _configuration.Load();
        }

        public IEnumerable<string> Models => _configuration.Models.Select(x => x.Name).ToList();

        public LlamaCpp? Model => _model;

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
                    configure.IgnoreEndOfStream = modelOptions.IgnoreEndOfStream ?? _configuration.IgnoreEndOfStream;
                    configure.InstructionPrompt = modelOptions.InstructionPrompt ?? _configuration.InstructionPrompt;
                    configure.StopOnInstructionPrompt = modelOptions.StopOnInstructionPrompt ?? _configuration.StopOnInstructionPrompt;
                });

                initialContext = modelOptions.InitialContext ?? _configuration.InitialContext;
            }
        }

        public void Unload()
        {
            _model?.Dispose();
            _model = null;
        }
    }
}
