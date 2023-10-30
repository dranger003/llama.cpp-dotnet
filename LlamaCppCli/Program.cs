using LlamaCppLib;

namespace LlamaCppCli
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            args = new[]
            {
                "D:\\LLM_MODELS\\teknium\\ggml-openhermes-2-mistral-7b-q8_0.gguf",
                //"D:\\LLM_MODELS\\codellama\\ggml-codellama-34b-instruct-q4_k.gguf",
            };

            await RunSampleAsync(args);
        }

        static async Task RunSampleAsync(string[] args)
        {
            var modelOptions = new ModelOptions { GpuLayers = 64 };
            using var model = new LlmModel();

            model.Load(args[0], modelOptions, loadProgress => { Console.Write($"\r{new String(' ', 32)}\r{loadProgress:F2}{(loadProgress == 100.0f ? "\n" : "")}"); });
            _ = model.RunAsync();

            Console.WriteLine("Press <any key> to quit.");
            Console.ReadKey(true);

            await model.StopAsync();
        }

        //static void RunSample(string[] args)
        //{
        //    //_requests
        //    //    .Where(r => r.Tokens[r.PosTokens - 1] == PInvoke.llama_token_eos(mdl))
        //    //    .ToList()
        //    //    .ForEach(r => Console.WriteLine($"\n{r.PosTokens / (double)(((r.T2 - r.T1)?.TotalSeconds) ?? 1):F2} t/s"));

        //    //if (!_requests.Any())
        //    //{
        //    //    Console.Write("\n> ");
        //    //    var prompt = Console.ReadLine() ?? String.Empty;

        //    //    prompt = Regex.Replace(prompt, "\\\\n", "\n");

        //    //    if (String.IsNullOrWhiteSpace(prompt))
        //    //        break;

        //    //    var match = Regex.Match(prompt, @"^/load (.+)$");
        //    //    if (match.Success)
        //    //        prompt = File.ReadAllText(match.Groups[1].Value);

        //    //    var tokens = Interop.llama_tokenize(_model.Handle, prompt, true, true);
        //    //    Console.WriteLine($"{tokens.Length} token(s)");

        //    //    _requests.Add(new LlmRequest(PInvoke.llama_n_ctx(_context.Handle), tokens));

        //    //    PInvoke.llama_kv_cache_seq_rm(_context.Handle, 0, 0, -1);
        //    //}
        //}
    }
}
