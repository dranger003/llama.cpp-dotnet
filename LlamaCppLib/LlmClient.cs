using System.Net.Http.Json;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace LlamaCppLib
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum LlmModelStatus { Unknown, Unloaded, Loaded };

    public class LlmStateResponse
    {
        public string? ModelName { get; set; }
        public LlmModelStatus? ModelStatus { get; set; }
    }

    public class LlmLoadRequest
    {
        public string? ModelName { get; set; }
        public LlmModelOptions? ModelOptions { get; set; }
    }

    public class LlmPromptRequest
    {
        public List<LlmMessage>? Messages { get; set; }
        public SamplingOptions? SamplingOptions { get; set; }
    }

    public class LlmClient : IDisposable
    {
        private HttpClient _httpClient = new();
        private readonly Uri _baseUri;

        public LlmClient(string uri) : this(new Uri(uri))
        {
            _httpClient.Timeout = TimeSpan.FromHours(1);
        }

        public LlmClient(Uri uri) => _baseUri = uri;

        public void Dispose() => _httpClient.Dispose();

        public async Task<List<string>> ListAsync()
        {
            using var response = await _httpClient.GetAsync(new Uri(_baseUri, $"/list"));
            return await response.Content.ReadFromJsonAsync<List<string>>() ?? new();
        }

        public async Task<LlmStateResponse> StateAsync()
        {
            using var response = await _httpClient.GetAsync(new Uri(_baseUri, $"/state"));
            return (await response.Content.ReadFromJsonAsync<LlmStateResponse>()) ?? new();
        }

        public async Task<LlmStateResponse> LoadAsync(string modelName, LlmModelOptions? options = default)
        {
            using var response = await _httpClient.PostAsync(new Uri(_baseUri, $"/load"), JsonContent.Create(new { ModelName = modelName, ModelOptions = options ?? new() }));
            return (await response.Content.ReadFromJsonAsync<LlmStateResponse>()) ?? new();
        }

        public async Task<LlmStateResponse> UnloadAsync()
        {
            using var response = await _httpClient.GetAsync(new Uri(_baseUri, $"/unload"));
            return (await response.Content.ReadFromJsonAsync<LlmStateResponse>()) ?? new();
        }

        public async IAsyncEnumerable<string> PromptAsync(List<LlmMessage> messages, SamplingOptions? samplingOptions = default, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            using var response = await _httpClient.PostAsync(
                new Uri(_baseUri, $"/prompt"),
                JsonContent.Create(new { Messages = messages, SamplingOptions = samplingOptions }),
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken
            );

            await using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var reader = new StreamReader(stream);

            while (!reader.EndOfStream && !cancellationToken.IsCancellationRequested)
            {
                var data = await reader.ReadLineAsync(cancellationToken) ?? String.Empty;
                yield return Encoding.UTF8.GetString(Convert.FromBase64String(Regex.Replace(data, @"^data: |\n\n$", String.Empty)));
            }
        }
    }
}
