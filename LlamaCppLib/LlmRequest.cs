using System.Runtime.CompilerServices;
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

        public bool Cancelled { get; private set; }

        public SamplingOptions SamplingOptions { get; private set; } = new();
        public string Prompt { get; private set; }
        public bool PrependBosToken { get; private set; }
        public bool ProcessSpecialTokens { get; private set; }
        public int[]? ExtraStopTokens { get; set; }

        public Channel<byte[]> Tokens { get; private set; }

        public async IAsyncEnumerable<byte[]> NextToken([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var result = default(byte[]?);

            while (true)
            {
                try
                {
                    if ((result = await this.Tokens.Reader.ReadAsync(cancellationToken)).Length == 0)
                        break;
                }
                catch (OperationCanceledException)
                {
                    this.Cancelled = true;
                    break;
                }

                yield return result;
            }
        }

        public TimeSpan PromptingTime { get; set; }
        public TimeSpan SamplingTime { get; set; }
    }
}
