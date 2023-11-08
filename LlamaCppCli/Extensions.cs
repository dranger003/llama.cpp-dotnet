public static class Extensions
{
    public static Task<HttpResponseMessage> PostAsync(
        this HttpClient client,
        string? requestUri,
        HttpContent? content,
        HttpCompletionOption? completionOption = default,
        CancellationToken? cancellationToken = default)
    {
        return client.SendAsync(
            new HttpRequestMessage(HttpMethod.Post, requestUri) { Content = content },
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken ?? default
        );
    }
}
