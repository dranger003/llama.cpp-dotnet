namespace LlamaCppLib
{
    public class LlamaCppLoader
    {
        private IConfiguration _configuration;
        private Dictionary<string, string> _modelPathsByName = new();
        private LlamaCpp? _model;

        public LlamaCppLoader(IConfiguration configuration)
        {
            _configuration = configuration;
            _ReadModels();
        }

        private void _ReadModels()
        {
            _modelPathsByName = _configuration
                .GetSection("LlamaCpp:Models")
                .GetChildren()
                .ToDictionary(
                    k => k.GetValue<string>("Name") ?? String.Empty,
                    v => v.GetValue<string>("Path") ?? String.Empty
                );
        }

        public IEnumerable<string> Models
        {
            get
            {
                _ReadModels();
                return _modelPathsByName
                    .Select(model => model.Key)
                    .ToList();
            }
        }

        public LlamaCpp Model
        {
            get
            {
                if (_model == null)
                    throw new InvalidOperationException("No model loaded.");

                return _model;
            }
        }

        public bool IsModelLoaded => _model != null;

        public void Load(string modelName)
        {
            var modelPath = _modelPathsByName[modelName];

            // Different model requested? If so, dispose current
            if (_model != null && _model.ModelPath != modelPath)
                Unload();

            // No model loaded, load it
            if (_model == null)
            {
                _model = new LlamaCpp(modelName);
                _model.Load(modelPath);
            }
        }

        public void Unload()
        {
            _model?.Dispose();
            _model = null;
        }
    }
}
