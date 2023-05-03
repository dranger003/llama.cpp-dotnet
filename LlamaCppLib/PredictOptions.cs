namespace LlamaCppLib
{
    using LlamaToken = System.Int32;

    public class PredictOptions
    {
        public List<LlamaToken> ContextVocabIds { get; set; } = new();
        public List<LlamaToken> PromptVocabIds { get; set; } = new();
        public float MirostatMU { get; set; }
    }
}
