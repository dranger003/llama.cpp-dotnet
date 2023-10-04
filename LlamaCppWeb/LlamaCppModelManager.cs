using LlamaCppWeb;

namespace LlamaCppLib
{
    //public enum LlamaCppModelStatus { Unloaded, Loaded }

    //public class LlamaCppModelManager
    //{
    //    private LlamaCppConfiguration _configuration;
    //    private LlamaCppModel? _model;
    //    private string? _modelName;

    //    public LlamaCppModelManager(IConfiguration configuration)
    //    {
    //        _configuration = new LlamaCppConfiguration(configuration);
    //        _configuration.Load();
    //    }

    //    public IEnumerable<string> Models { get => _configuration.Models.Select(x => x.Name ?? String.Empty); }

    //    public LlamaCppModelStatus Status { get => _model == null ? LlamaCppModelStatus.Unloaded : LlamaCppModelStatus.Loaded; }

    //    public LlamaCppModel? Model => _model;
    //    public string? ModelName => _modelName;

    //    public void LoadModel(string modelName, LlamaCppModelOptions options)
    //    {
    //        var modelIndex = _configuration.Models
    //            .Select((model, index) => (Model: model, Index: index))
    //            .Where(x => x.Model.Name == modelName)
    //            .Single()
    //            .Index;

    //        if (_model != null)
    //            UnloadModel();

    //        if (_model == null)
    //        {
    //            var modelPath = _configuration.Models[modelIndex].Path ?? String.Empty;

    //            if (!Path.Exists(modelPath))
    //                throw new FileNotFoundException(modelPath);

    //            _model = new();
    //            _model.Load(modelPath, options);
    //            _modelName = modelName;
    //        }
    //    }

    //    public void UnloadModel()
    //    {
    //        _model?.Dispose();
    //        _model = null;
    //    }
    //}
}
