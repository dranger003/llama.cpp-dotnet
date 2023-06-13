using System.Text.Json.Serialization;

namespace LlamaCppWeb
{
    public class CreateChatCompletionMessage
    {
        [JsonPropertyName("role")]
        public string? Role { get; set; }

        [JsonPropertyName("content")]
        public string? Content { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }
    }

    public class CreateChatCompletion
    {
        [JsonPropertyName("model")]
        public string? Model { get; set; }

        [JsonPropertyName("messages")]
        public List<CreateChatCompletionMessage> Messages { get; set; } = new();

        [JsonPropertyName("temperature")]
        public float Temperature { get; set; } = 0.8f;

        [JsonPropertyName("top_p")]
        public float TopP { get; set; } = 0.95f;

        [JsonPropertyName("n")]
        public int N { get; set; } = 1;

        [JsonPropertyName("stream")]
        public bool Stream { get; set; } = false;

        [JsonPropertyName("stop")]
        public string? Stop { get; set; }

        [JsonPropertyName("max_tokens")]
        public int? MaxTokens { get; set; }

        [JsonPropertyName("presence_penalty")]
        public float PresencePenalty { get; set; } = 1.1f;

        [JsonPropertyName("frequency_penalty")]
        public float FrequencyPenalty { get; set; } = 0;

        [JsonPropertyName("logit_bias")]
        public Dictionary<int, int> LogitBias { get; set; } = new();

        [JsonPropertyName("user")]
        public string? User { get; set; }
    }
}
