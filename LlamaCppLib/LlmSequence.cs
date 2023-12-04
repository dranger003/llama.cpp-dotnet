using System.Diagnostics.CodeAnalysis;

namespace LlamaCppLib
{
    internal class LlmSequence : IEquatable<LlmSequence>
    {
        public int Id { get; set; }

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosTokens { get; set; }
        public int PosResponse { get; set; }
        public int[] Tokens { get; set; }
        public int[] StopTokens { get; set; }

        public SamplingOptions SamplingOptions { get; set; } = new();
        public int MirostatM { get; private set; }
        public float MirostatMu = 0.0f;

        public DateTime? T1 { get; set; }
        public DateTime? T2 { get; set; }
        public DateTime? T3 { get; set; }

        public LlmPrompt Prompt { get; private set; }

        public LlmSequence(LlmPrompt prompt, int tokenCount, ReadOnlySpan<int> tokens, ReadOnlySpan<int> stopTokens, int mirostatM = 100)
        {
            this.Tokens = new int[tokenCount];
            tokens.CopyTo(Tokens);

            this.StopTokens = new int[stopTokens.Length];
            stopTokens.CopyTo(StopTokens);

            this.MirostatM = mirostatM;
            this.Prompt = prompt;
            this.SamplingOptions = prompt.SamplingOptions;

            PosTokens += tokens.Length;
            PosResponse = PosTokens;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is LlmSequence request && Equals(request);
        public override int GetHashCode() => Id.GetHashCode();

        // IEquatable<T>
        public bool Equals(LlmSequence? other) => other?.Id == this.Id;
    }
}
