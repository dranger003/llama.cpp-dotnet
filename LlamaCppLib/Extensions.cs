using System.Text;

namespace LlamaCppLib
{
    public static class Extensions
    {
        private static Encoding? _utf8;

        public static bool TryGetUtf8String(this byte[] bytes, out string? str)
        {
            if (_utf8 == null)
            {
                _utf8 = (Encoding)Encoding.UTF8.Clone();
                _utf8.DecoderFallback = new DecoderExceptionFallback();
            }

            try
            {
                _utf8.DecoderFallback = new DecoderExceptionFallback();
                str = _utf8.GetString(bytes);
                return true;
            }
            catch (DecoderFallbackException)
            {
                str = null;
                return false;
            }
        }

        public static Task<HttpResponseMessage> PostAsync(
            this HttpClient client,
            string? requestUri,
            HttpContent? content,
            HttpCompletionOption? completionOption = default,
            CancellationToken? cancellationToken = default)
        {
            return client.SendAsync(
                new HttpRequestMessage(HttpMethod.Post, requestUri) { Content = content },
                completionOption ?? HttpCompletionOption.ResponseContentRead,
                cancellationToken ?? default
            );
        }

        public static Task<HttpResponseMessage> PostAsync(
            this HttpClient client,
            Uri? requestUri,
            HttpContent? content,
            HttpCompletionOption? completionOption = default,
            CancellationToken? cancellationToken = default)
        {
            return client.SendAsync(
                new HttpRequestMessage(HttpMethod.Post, requestUri) { Content = content },
                completionOption ?? HttpCompletionOption.ResponseContentRead,
                cancellationToken ?? default
            );
        }
    }
}
