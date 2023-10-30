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

        public string Prompt { get; private set; }
        public bool PrependBosToken { get; private set; }
        public bool ProcessSpecialTokens { get; private set; }

        public Channel<byte[]> Tokens { get; private set; }
    }
}
