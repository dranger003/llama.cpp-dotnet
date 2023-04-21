namespace LlamaCppLib
{
    public class LlamaCppOptions
    {
        public int ThreadCount { get; set; } = 4;
        public int TopK { get; set; } = 40;
        public float TopP { get; set; } = 0.95f;
        public float Temperature { get; set; } = 0.0f;
        public float RepeatPenalty { get; set; } = 1.5f;
    }
}
