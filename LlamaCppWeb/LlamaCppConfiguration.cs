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
            public bool? IgnoreEndOfStream { get; set; }
            public string? InstructionPrompt { get; set; }
            public bool? StopOnInstructionPrompt { get; set; }
            public string? InitialContext { get; set; }
        }

        public class Model
        {
            public string Name { get; set; } = String.Empty;
            public string Path { get; set; } = String.Empty;
            public ModelOptions Options { get; set; } = new();
        }

        public int ThreadCount { get; set; } = 4;
        public int TopK { get; set; } = 40;
        public float TopP { get; set; } = 0.95f;
        public float Temperature { get; set; } = 0.0f;
        public float RepeatPenalty { get; set; } = 1.5f;
        public bool IgnoreEndOfStream { get; set; } = false;
        public string InstructionPrompt { get; set; } = String.Empty;
        public bool StopOnInstructionPrompt { get; set; } = false;
        public string InitialContext { get; set; } = String.Empty;

        public List<Model> Models { get; set; } = new();

        public IConfiguration Configuration;
        public LlamaCppConfiguration(IConfiguration configuration) => Configuration = configuration;

        public void Load() => Configuration.GetSection(nameof(LlamaCppConfiguration)).Bind(this);
        public void Reload() => Load();
    }
}
