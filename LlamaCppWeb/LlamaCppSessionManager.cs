using System.Text.Json.Serialization;

namespace LlamaCppLib
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public enum LlamaCppSessionAction { Create }

    public class LlamaCppSessionManager
    {
        public IDictionary<Guid, LlamaCppSession> Sessions { get; private set; } = new Dictionary<Guid, LlamaCppSession>();
    }
}
