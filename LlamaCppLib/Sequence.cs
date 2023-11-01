using System.Diagnostics.CodeAnalysis;

namespace LlamaCppLib
{
    internal class LlmSequence : IEquatable<LlmSequence>
    {
        public int Id { get; set; }

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosTokens { get; set; }
        public int[] Tokens { get; set; }

        public SamplingOptions SamplingOptions { get; set; } = new();
        public int MirostatM { get; private set; }
        public float MirostatMu = 0.0f;

        public LlmRequest Request { get; private set; }

        public LlmSequence(LlmRequest request, int tokenCount, ReadOnlySpan<int> tokens, int mirostatM = 100)
        {
            this.Tokens = new int[tokenCount];
            this.MirostatM = mirostatM;
            this.Request = request;
            this.SamplingOptions = request.SamplingOptions;

            tokens.CopyTo(Tokens);
            PosTokens += tokens.Length;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is LlmSequence request && Equals(request);
        public override int GetHashCode() => Id.GetHashCode();

        // IEquatable<T>
        public bool Equals(LlmSequence? other) => other?.Id == this.Id;
    }
}
