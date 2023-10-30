using LlamaCppLib;
using System.Text;

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
            using var llm = new LlmEngine();

            llm.LoadModel(
                args[0],
                new ModelOptions { GpuLayers = 64 },
                loadProgress => Console.Write($"\r{new String(' ', 64)}\rLoading... {loadProgress:F2}%{(loadProgress == 100.0f ? "\n" : "")}")
            );

            var task = Task.Run(
                async () =>
                {
                    await llm.WaitForRunningAsync();
                    Console.WriteLine("\nEngine ready and model loaded.");

                    while (true)
                    {
                        Console.Write("> ");
                        var line = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                        if (String.IsNullOrWhiteSpace(line))
                            break;

                        var request = llm.NewRequest(line, true, true);
                        await foreach (var token in request.Tokens.Reader.ReadAllAsync())
                            Console.Write(Encoding.ASCII.GetString(token));

                        Console.WriteLine();
                    }

                    Console.WriteLine($"Stopping...");
                    await llm.StopAsync();
                }
            );

            await llm.RunAsync();

            await task;
            Console.WriteLine("Done.");
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
