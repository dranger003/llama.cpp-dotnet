using LlamaCppLib;

namespace LlamaCppWeb
{
    public class LlamaCppConfiguration
    {
        public class Model
        {
            public string? Name { get; set; }
            public string? Path { get; set; }
        }

        public List<Model> Models { get; set; } = new();

        public IConfiguration Configuration;
        public LlamaCppConfiguration(IConfiguration configuration) => Configuration = configuration;

        public void Load() => Configuration.GetSection(nameof(LlamaCpp)).Bind(this);
        public void Reload() => Load();
    }
}
