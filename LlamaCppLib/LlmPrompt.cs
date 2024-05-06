using System.Runtime.CompilerServices;
using System.Threading.Channels;

namespace LlamaCppLib
{
    public class LlmMessage
    {
        public string? Role { get; set; }
        public string? Content { get; set; }
    }

    public class LlmPrompt
    {
        public LlmPrompt(List<LlmMessage> messages)
        {
            this.Messages = messages;
            this.TokenChannel = Channel.CreateUnbounded<byte[]>();
        }

        public LlmPrompt(List<LlmMessage> messages, SamplingOptions samplingOptions) : this(messages)
        {
            this.SamplingOptions = samplingOptions;
        }

        public bool Cancelled { get; private set; }

        public SamplingOptions SamplingOptions { get; private set; } = new();
        public List<LlmMessage> Messages { get; private set; }

        public Channel<byte[]> TokenChannel { get; private set; }

        public async IAsyncEnumerable<byte[]> NextToken([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var result = default(byte[]?);

            while (true)
            {
                try
                {
                    if ((result = await this.TokenChannel.Reader.ReadAsync(cancellationToken)).Length == 0)
                        break;
                }
                catch (OperationCanceledException)
                {
                    this.Cancelled = true;
                    break;
                }
                catch (ChannelClosedException)
                {
                    break;
                }

                yield return result;
            }
        }

        public double PromptingSpeed { get; set; }
        public double SamplingSpeed { get; set; }
    }

    public class TokenEnumerator : IAsyncEnumerable<string>
    {
        private MultibyteCharAssembler _assembler = new();
        private LlmPrompt _prompt;
        private CancellationToken? _cancellationToken;

        public TokenEnumerator(LlmPrompt prompt, CancellationToken? cancellationToken = default)
        {
            _prompt = prompt;
            _cancellationToken = cancellationToken;
        }

        public async IAsyncEnumerator<string> GetAsyncEnumerator(CancellationToken cancellationToken = default)
        {
            var ct = _cancellationToken != null
                ? CancellationTokenSource.CreateLinkedTokenSource(_cancellationToken.Value, cancellationToken).Token
                : cancellationToken;

            await foreach (var token in _prompt.NextToken(ct))
                yield return _assembler.Consume(token);

            yield return _assembler.Consume();
        }
    }
}
