namespace LlamaCppWeb
{
    public class LlamaCppConfiguration
    {
        // Model options override global options
        public class ModelOptions
        {
            public int? Seed { get; set; }
            public int? PredictCount { get; set; }
            public int? ContextSize { get; set; }
            public int? LastTokenCountPenalty { get; set; }
            public bool? UseHalf { get; set; }
            public bool? NewLinePenalty { get; set; }
            public bool? UseMemoryMapping { get; set; }
            public bool? UseMemoryLocking { get; set; }
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

        // Global options
        public int? Seed { get; set; }
        public int? PredictCount { get; set; }
        public int? ContextSize { get; set; }
        public int? LastTokenCountPenalty { get; set; }
        public bool? UseHalf { get; set; }
        public bool? NewLinePenalty { get; set; }
        public bool? UseMemoryMapping { get; set; }
        public bool? UseMemoryLocking { get; set; }
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
