namespace LlamaCppWeb
{
    internal class QueryPredict
    {
        public string? Context { get; set; }

        public string? Prompt { get; set; }

        public static ValueTask<QueryPredict> BindAsync(HttpContext context)
        {
            var query = context.Request.Query;
            return ValueTask.FromResult(new QueryPredict { Context = query["context"], Prompt = query["prompt"] });
        }
    }
}
