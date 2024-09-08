namespace LlamaCppLib
{
    public class LlmEngineOptions
    {
        public bool NumaOptimizations { get; set; } = false;
        public int MaxParallel { get; set; } = 1;
    }

    public class LlmModelOptions
    {
        public int GpuLayers { get; set; } = 0;
        public int MainGpu { get; set; } = 0;
        public float[]? TensorSplit { get; set; } = null;
        public bool UseMemoryMap { get; set; } = true;
        public bool UseMemoryLock { get; set; } = false;

        public int ContextLength { get; set; } = 0;
        public int BatchSize { get; set; } = 512;
        public int ThreadCount { get; set; } = 4;
        public int BatchThreadCount { get; set; } = 4;
        public bool UseFlashAttention { get; set; } = false;

        public float RopeFrequeceBase { get; set; } = 0.0f;
        public float RopeFrequenceScale { get; set; } = 0.0f;
    }

    public enum Mirostat : int { Disabled, MirostatV1, MirostatV2 }

    public class SamplingOptions
    {
        public int Seed { get; set; } = -1;
        public int TopK { get; set; } = 40;
        public float TopP { get; set; } = 0.95f;
        public float MinP { get; set; } = 0.05f;
        public float TfsZ { get; set; } = 1.0f;
        public float TypicalP { get; set; } = 1.0f;
        public float Temperature { get; set; } = 0.8f;

        public Mirostat Mirostat { get; set; } = Mirostat.Disabled;
        public float MirostatTau { get; set; } = 5.0f;
        public float MirostatEta { get; set; } = 0.1f;

        public int PenaltyLastN { get; set; } = 64;
        public float PenaltyRepeat { get; set; } = 1.0f;
        public float PenaltyFreq { get; set; } = 0.0f;
        public float PenaltyPresent { get; set; } = 0.0f;

        public int? ResponseMaxTokenCount { get; set; } = default;
        public string[]? ExtraStopTokens { get; set; } = default;
    }
}
