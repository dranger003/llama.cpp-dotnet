namespace LlamaCppWeb
{
    internal class QueryConfigure
    {
        public int ThreadCount { get; set; }

        public string? InstructionPrompt { get; set; }

        public bool StopOnInstructionPrompt { get; set; }

        public static ValueTask<QueryConfigure> BindAsync(HttpContext context)
        {
            var query = context.Request.Query;
            return ValueTask.FromResult(new QueryConfigure
            {
                ThreadCount = $"{query["threadCount"]}".ToInt32(),
                InstructionPrompt = $"{query["instructionPrompt"]}",
                StopOnInstructionPrompt = $"{query["StopOnInstructionPrompt"]}".ToBool(),
            });
        }
    }
}
