using System.Threading.Channels;

namespace LlamaCppLib
{
    public class LlmRequest
    {
        public LlmRequest(string prompt, bool preprendBosToken = false, bool processSpecialTokens = false)
        {
            this.Prompt = prompt;
            this.PrependBosToken = preprendBosToken;
            this.ProcessSpecialTokens = processSpecialTokens;

            this.Tokens = Channel.CreateUnbounded<byte[]>();
        }

        public LlmRequest(string prompt, SamplingOptions samplingOptions, bool preprendBosToken = false, bool processSpecialTokens = false) :
            this(prompt, preprendBosToken, processSpecialTokens)
        {
            this.SamplingOptions = samplingOptions;
        }

        public SamplingOptions SamplingOptions { get; private set; } = new();
        public string Prompt { get; private set; }
        public bool PrependBosToken { get; private set; }
        public bool ProcessSpecialTokens { get; private set; }

        public Channel<byte[]> Tokens { get; private set; }
    }
}
