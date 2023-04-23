namespace LlamaCppWeb
{
    public class LlamaCppConfiguration
    {
        public class ModelOptions
        {
            public int? ThreadCount { get; set; }
            public int? TopK { get; set; }
            public float? TopP { get; set; }
            public float? Temperature { get; set; }
            public float? RepeatPenalty { get; set; }
        }

        public class Model
        {
            public string? Name { get; set; }
            public string? Path { get; set; }
            public ModelOptions Options { get; set; } = new();
        }

        public int? ThreadCount { get; set; }
        public int? TopK { get; set; }
        public float? TopP { get; set; }
        public float? Temperature { get; set; }
        public float? RepeatPenalty { get; set; }

        public List<Model> Models { get; set; } = new();

        public IConfiguration Configuration;
        public LlamaCppConfiguration(IConfiguration configuration) => Configuration = configuration;

        public void Load() => Configuration.GetSection(nameof(LlamaCppConfiguration)).Bind(this);
        public void Reload() => Load();
    }
}
