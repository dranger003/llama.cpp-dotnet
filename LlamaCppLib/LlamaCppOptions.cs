using System.Text.Json;

namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public enum Mirostat { Disabled, Mirostat, Mirostat2 }

    public class LlamaCppModelOptions
    {
        public uint Seed { get; set; } = unchecked((uint)-1);
        public int ContextSize { get; set; } = 512;
        public int BatchSize { get; set; } = 512;
        public bool UseMemoryMapping { get; set; } = true;
        public bool UseMemoryLocking { get; set; } = false;
        public int GpuLayers { get; set; } = 0;
        public float RopeFrequencyBase { get; set; } = 10000.0f;
        public float RopeFrequencyScale { get; set; } = 1.0f;

        // LLAMA-2 (TEMPORARY)
        public int GroupedQueryAttentionCount { get; set; } = 1;    // Grouped-query attention (TEMPORARY)
        public float RmsNormEpsilon { get; set; } = 5e-6f;          // RMS norm epsilon (TEMPORARY)

        public static bool TryParse(string input, out LlamaCppModelOptions options)
        {
            options = JsonSerializer.Deserialize<LlamaCppModelOptions>(input) ?? new();
            return true;
        }
    }

    public class LlamaCppGenerateOptions
    {
        public int ThreadCount { get; set; } = 4;
        public int TopK { get; set; } = 40;
        public float TopP { get; set; } = 0.95f;
        public float Temperature { get; set; } = 0.8f;
        public float RepeatPenalty { get; set; } = 1.1f;
        public int LastTokenCountPenalty { get; set; } = 64;
        public bool PenalizeNewLine { get; set; } = false;

        public Dictionary<LlamaToken, float> LogitBias { get; set; } = new();
        public float TfsZ { get; set; } = 1.0f;
        public float TypicalP { get; set; } = 1.0f;
        public float FrequencyPenalty { get; set; } = 0.0f;
        public float PresencePenalty { get; set; } = 0.0f;
        public Mirostat Mirostat { get; set; } = Mirostat.Disabled;
        public float MirostatTAU { get; set; } = 5.0f;
        public float MirostatETA { get; set; } = 0.1f;

        public static bool TryParse(string input, out LlamaCppGenerateOptions options)
        {
            options = JsonSerializer.Deserialize<LlamaCppGenerateOptions>(input) ?? new();
            return true;
        }
    }
}
