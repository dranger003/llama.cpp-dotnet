namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public enum Mirostat { Disabled, Mirostat, Mirostat2 }

    public class LlamaCppOptions
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

        // New sampling options
        public Dictionary<LlamaToken, float> LogitBias { get; set; } = new();
        public float? TfsZ { get; set; }
        public float? TypicalP { get; set; }
        public float? FrequencyPenalty { get; set; }
        public float? PresencePenalty { get; set; }
        public Mirostat? Mirostat { get; set; }
        public float? MirostatTAU { get; set; }
        public float? MirostatETA { get; set; }
        public bool? PenalizeNewLine { get; set; }

        public bool IsConfigured => ThreadCount != null && TopK != null && TopP != null && Temperature != null && RepeatPenalty != null;
    }
}
