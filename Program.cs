namespace LlamaCppDotNet
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = 2048;
            cparams.n_parts = -1;
            cparams.seed = 0;
            cparams.f16_kv = false;
            cparams.use_mlock = false;
            var ctx = LlamaCppInterop.llama_init_from_file(@"D:\LLM_MODELS\eachadea\ggml-vicuna-13b-4bit\ggml-vicuna-13b-4bit-rev1.bin", cparams);

            var embd_inp = LlamaCppInterop.llama_tokenize(
                ctx,
                " You are a helpful assistant providing precise and truthful answers to questions and if you don't know the answer you say \"I don't know\". Acknowledge you understand the request by saying \"I confirm\".",
                true
            ).ToList();
            var n_ctx = LlamaCppInterop.llama_n_ctx(ctx);

            var antiprompts = new[] { "### Human:" };

            int n_keep = 0;
            int n_threads = 16;
            int n_batch = 8;
            int repeat_last_n = 64;
            int top_k = 40;
            float top_p = 0.95f;
            float temp = 0.0f;
            float repeat_penalty = 1.10f;

            int n_past = 0;
            int n_consumed = 0;

            var last_n_tokens = Enumerable.Repeat(0, n_ctx).ToList();
            var embd = new List<int>();

            bool is_interacting = false;

            var prompt_idx = 0;
            var prompts = new[]
            {
                "What is the second planet of the solar system?",
                "What is the second to last planet in the solar system?",
            };

            while (true)
            {
                if (embd.Count > 0)
                {
                    if (n_past + embd.Count > n_ctx)
                    {
                        int n_left = n_past - n_keep;
                        n_past = n_keep;
                        embd.InsertRange(0, last_n_tokens.GetRange(n_ctx - n_left / 2 - embd.Count, last_n_tokens.Count - embd.Count));
                    }

                    LlamaCppInterop.llama_eval(ctx, embd.ToArray(), embd.Count, n_past, n_threads);
                }

                n_past += embd.Count;
                embd.Clear();

                if (n_consumed >= embd_inp.Count && !is_interacting)
                {
                    int id = 0;
                    {
                        id = LlamaCppInterop.llama_sample_top_p_top_k(ctx, last_n_tokens.GetRange(n_ctx - repeat_last_n, repeat_last_n).ToArray(), repeat_last_n, top_k, top_p, temp, repeat_penalty);
                        last_n_tokens.RemoveAt(0);
                        last_n_tokens.Add(id);
                    }
                    embd.Add(id);
                }
                else
                {
                    while (n_consumed < embd_inp.Count)
                    {
                        embd.Add(embd_inp[n_consumed]);
                        last_n_tokens.RemoveAt(0);
                        last_n_tokens.Add(embd_inp[n_consumed]);
                        ++n_consumed;
                        if (n_batch <= embd.Count)
                        {
                            break;
                        }
                    }
                }

                foreach (var id in embd)
                {
                    Console.Write(LlamaCppInterop.llama_token_to_str(ctx, id));
                    Console.Out.Flush();
                }

                if (n_consumed >= embd_inp.Count)
                {
                    if (antiprompts.Any())
                    {
                        string last_output = String.Empty;
                        foreach (var id in last_n_tokens)
                        {
                            last_output += LlamaCppInterop.llama_token_to_str(ctx, id);
                        }

                        foreach (var antiprompt in antiprompts)
                        {
                            if (last_output.EndsWith(antiprompt))
                            {
                                is_interacting = true;
                                Console.Write(" ");
                                break;
                            }
                        }
                    }

                    if (n_past > 0 && is_interacting)
                    {
                        if (prompt_idx >= prompts.Length)
                            break;

                        var line_inp = LlamaCppInterop.llama_tokenize(ctx, prompts[prompt_idx++], false);
                        embd_inp.AddRange(line_inp);

                        is_interacting = false;
                    }
                }
            }

            LlamaCppInterop.llama_print_timings(ctx);
            LlamaCppInterop.llama_free(ctx);

            await Task.CompletedTask;
            // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            //var builder = WebApplication.CreateBuilder(args);
            //builder.Services.AddHostedService<LlamaCppService>();
            //var app = builder.Build();

            //app.MapGet("/", async context => {
            //    await context.Response.WriteAsync("Welcome to Vicuna 13B!");
            //});

            //app.MapGet("/reload", async (HttpContext context, IEnumerable<IHostedService> services) => {
            //    var llamaCppService = services.OfType<LlamaCppService>().Single();
            //    await context.Response.WriteAsync("Reloading.");
            //    llamaCppService.Reload();
            //});

            //try
            //{
            //    app.Run();
            //}
            //catch (TaskCanceledException)
            //{ }
            // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        }
    }
}
