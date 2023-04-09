using System.Collections.Concurrent;
using System.Text;

namespace LlamaCppDotNet
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            var cts = new CancellationTokenSource();

            var input = new BlockingCollection<string>();
            var output = new BlockingCollection<string>();

            var context = "";
            //var context = """
            //    ### Human: What is the role of the Oort Cloud in our solar system?
            //    ### Assistant: The Oort Cloud is a hypothetical, vast, spherical region surrounding our solar system, containing billions of icy objects and comets. Its role is primarily as a reservoir for long-period comets, which occasionally enter the inner solar system due to gravitational perturbations from nearby stars or other celestial objects.
            //    """;

            input.Add("Hi! I have a question, can you help me?");
            input.Add("What on Earth causes the distinct features of the moon?");

            // -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            var cparams = LlamaCppInterop.llama_context_default_params();
            cparams.n_ctx = 2048;
            cparams.n_parts = -1;
            cparams.seed = 0;
            cparams.f16_kv = false;
            cparams.use_mlock = false;
            var ctx = LlamaCppInterop.llama_init_from_file(@"D:\LLM_MODELS\eachadea\ggml-vicuna-13b-4bit\ggml-vicuna-13b-4bit-rev1.bin", cparams);

            Console.Error.WriteLine($"llama_print_system_info: {LlamaCppInterop.llama_print_system_info()}");

            var prompt = new StringBuilder();
            prompt.Append(File.ReadAllText("personality.txt"));
            prompt.Append(context);

            var embd_inp = LlamaCppInterop.llama_tokenize(ctx, prompt.ToString(), false).ToList();
            var n_ctx = LlamaCppInterop.llama_n_ctx(ctx);

            var antiprompts = new[] { "### Human:" };

            int n_keep = embd_inp.Count;
            int n_threads = 16;
            int n_batch = n_ctx;
            int repeat_last_n = n_ctx;
            int top_k = 40;
            float top_p = 0.95f;
            float temp = 0.0f;
            float repeat_penalty = 1.5f;

            int n_past = 0;
            int n_consumed = 0;

            var last_n_tokens = Enumerable.Repeat(0, n_ctx).ToList();
            var embd = new List<int>();

            bool is_infering = true;

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

                if (n_consumed >= embd_inp.Count && is_infering)
                {
                    int id = 0;
                    {
                        id = LlamaCppInterop.llama_sample_top_p_top_k(
                            ctx,
                            last_n_tokens.GetRange(n_ctx - repeat_last_n, repeat_last_n).ToArray(),
                            repeat_last_n,
                            top_k,
                            top_p,
                            temp,
                            repeat_penalty
                        );

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
                        var last_output = new StringBuilder();
                        var n = last_n_tokens.Count;
                        while (--n > 0)
                        {
                            last_output.Insert(0, LlamaCppInterop.llama_token_to_str(ctx, last_n_tokens[n]));

                            foreach (var antiprompt in antiprompts)
                            {
                                if (last_output.ToString().EndsWith(antiprompt))
                                {
                                    is_infering = false;
                                    n = 0;
                                    Console.Write(" ");
                                    break;
                                }
                            }
                        }
                    }

                    if (n_past > 0 && !is_infering)
                    {
                        if (input.TryTake(out var next_prompt, (int)TimeSpan.FromSeconds(2).TotalMilliseconds, cts.Token))
                        {
                            embd_inp.AddRange(LlamaCppInterop.llama_tokenize(ctx, next_prompt, false));
                            is_infering = true;
                        }
                        else
                        {
                            break;
                        }
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
