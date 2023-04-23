namespace LlamaCppLib
{
    public class LlamaCppOptions
    {
        public int? ThreadCount { get; set; }
        public int? TopK { get; set; }
        public float? TopP { get; set; }
        public float? Temperature { get; set; }
        public float? RepeatPenalty { get; set; }

        public bool IsConfigured  => ThreadCount != null && TopK != null && TopP != null && Temperature != null && RepeatPenalty != null;
    }
}
