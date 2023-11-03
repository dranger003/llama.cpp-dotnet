﻿using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

using LlamaCppLib;

using static LlamaCppLib.Native;
using static LlamaCppLib.Interop;

namespace LlamaCppCli
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            args = new[]
            {
                @"D:\LLM_MODELS\openchat\ggml-openchat_3.5-q8_0.gguf",
                @"D:\LLM_MODELS\teknium\ggml-openhermes-2-mistral-7b-q8_0.gguf",
                @"D:\LLM_MODELS\codellama\ggml-codellama-34b-instruct-q4_k.gguf",
            };

            // Multi-byte character encoding support (e.g. emojis)
            Console.OutputEncoding = Encoding.UTF8;

            //await RunSampleRawAsync(args);
            await RunSampleAsync(args);
        }

        /// <summary>
        /// This barebone sample serves for testing the native API using raw function calls.
        /// </summary>
        /// <param name="args"></param>
        static async Task RunSampleRawAsync(string[] args)
        {
            RunSampleRaw(args);
            await Task.CompletedTask;
        }

        [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
        static unsafe void ProgressCallback(float progress, void* state) => Console.Write($"{new string(' ', 32)}\rLoading model... {(byte)(progress * 100)}%\r");

        static unsafe void RunSampleRaw(string[] args)
        {
            var top_k = 40;
            var top_p = 0.95f;
            var tfs_z = 1.0f;
            var typical_p = 1.0f;
            var temp = 0.0f;

            var mirostat = 0;
            var mirostat_tau = 5.0f;
            var mirostat_eta = 0.1f;

            var penalty_last_n = 64;
            var penalty_repeat = 1.1f;
            var penalty_freq = 0.0f;
            var penalty_present = 0.0f;

            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = 64;
            mparams.progress_callback = &ProgressCallback;

            var cparams = llama_context_default_params();
            cparams.seed = 0;
            cparams.n_ctx = 0;
            cparams.n_batch = 64;
            cparams.n_threads = 1;
            cparams.n_threads_batch = 1;
            cparams.logits_all = true ? 1 : 0;

            llama_backend_init(false);

            var mdl = llama_load_model_from_file(args[0], mparams);
            var ctx = llama_new_context_with_model(mdl, cparams);
            var bat = llama_batch_init(llama_n_ctx(ctx), 0, 1);

            var requests = new List<Request>();

            const string prompt = "GPT4 User: Write a table listing the planets of the solar system in reverse order from the Sun. Then write another table listing the planets in order from the Sun.<|end_of_turn|>GPT4 Assistant:";
            //const string prompt = "<|im_start|>system\nYou are an emoji expert.<|im_end|>\n<|im_start|>user\nWhat are the top five emojis on the net?<|im_end|>\n<|im_start|>assistant\n";
            //const string prompt = "<|im_start|>system\nYou are an astrophysicist.<|im_end|>\n<|im_start|>user\nWrite a table listing the planets of the solar system in reverse order from the Sun. Then write another table listing the same planets but this time in order from the Sun. Lastly, provide an expert comment about the solar system and how it differs or not from the other planetary systems in the universe.<|im_end|>\n<|im_start|>assistant\n";

            var tokens = (ReadOnlySpan<int>)llama_tokenize(mdl, prompt, true, true);
            Console.WriteLine($"{tokens.Length} token(s)");

            for (var id = 0; id < 1; id++)
                requests.Add(new Request(llama_n_ctx(ctx), tokens) { Id = id });

            var stream = true;
            var assembler = new MultibyteCharAssembler();

            var stopTokens = new[]
            {
                llama_token_eos(mdl),
                llama_tokenize(mdl, "<|end_of_turn|>", false, true)[0], // openchat-3.5
            };

            while (true)
            {
                bat.n_tokens = 0;

                foreach (var request in requests)
                {
                    for (; request.PosBatch < request.PosToken; request.PosBatch++)
                        llama_batch_add(ref bat, request.Tokens[request.PosBatch], request.PosBatch, new[] { request.Id }, false);

                    request.PosLogit = bat.n_tokens - 1;
                    bat.logits[request.PosLogit] = true ? 1 : 0;
                }

                if (bat.n_tokens == 0)
                    break;

                var n_batch = (int)cparams.n_batch;
                for (var i = 0; i < bat.n_tokens; i += n_batch)
                {
                    var n_tokens = Math.Min(n_batch, bat.n_tokens - i);

                    llama_decode(
                        ctx,
                        new llama_batch
                        {
                            n_tokens = n_tokens,
                            token = bat.token + i,
                            embd = null,
                            pos = bat.pos + i,
                            n_seq_id = bat.n_seq_id + i,
                            seq_id = bat.seq_id + i,
                            logits = bat.logits + i,
                            all_pos_0 = 0,
                            all_pos_1 = 0,
                            all_seq_id = 0,
                        }
                    );

                    foreach (var request in requests)
                    {
                        if (request.PosLogit < i || request.PosLogit >= i + n_tokens)
                            continue;

                        var logits = llama_get_logits_ith(ctx, request.PosLogit - i);

                        var candidates = new llama_token_data[llama_n_vocab(mdl)];
                        for (var token = 0; token < candidates.Length; token++)
                        {
                            candidates[token].id = token;
                            candidates[token].logit = logits[token];
                            candidates[token].p = 0.0f;
                        }

                        fixed (llama_token_data* ptr1 = &candidates[0])
                        {
                            var candidates_p = new llama_token_data_array
                            {
                                data = ptr1,
                                size = (nuint)candidates.Length,
                                sorted = false ? 1 : 0,
                            };

                            fixed (int* ptr2 = &request.Tokens[Math.Max(0, request.PosToken - penalty_last_n)])
                            {
                                llama_sample_repetition_penalties(
                                    ctx,
                                    ref candidates_p,
                                    ptr2,
                                    (nuint)penalty_last_n,
                                    penalty_repeat,
                                    penalty_freq,
                                    penalty_present
                                );
                            }

                            var token = llama_token_eos(mdl);

                            if (temp < 0.0f)
                            {
                                llama_sample_softmax(ctx, ref candidates_p);
                                token = candidates_p.data[0].id;
                            }
                            else if (temp == 0.0f)
                            {
                                token = llama_sample_token_greedy(ctx, ref candidates_p);
                            }
                            else if (mirostat == 1)
                            {
                                llama_sample_temp(ctx, ref candidates_p, temp);
                                token = llama_sample_token_mirostat(ctx, ref candidates_p, mirostat_tau, mirostat_eta, 100, ref request.MirostatMU);
                            }
                            else if (mirostat == 2)
                            {
                                llama_sample_temp(ctx, ref candidates_p, temp);
                                token = llama_sample_token_mirostat_v2(ctx, ref candidates_p, mirostat_tau, mirostat_eta, ref request.MirostatMU);
                            }
                            else
                            {
                                llama_sample_top_k(ctx, ref candidates_p, top_k, 1);
                                llama_sample_tail_free(ctx, ref candidates_p, tfs_z, 1);
                                llama_sample_typical(ctx, ref candidates_p, typical_p, 1);
                                llama_sample_top_p(ctx, ref candidates_p, top_p, 1);
                                llama_sample_temp(ctx, ref candidates_p, temp);
                                token = llama_sample_token(ctx, ref candidates_p);
                            }

                            request.Tokens[request.PosToken++] = token;

                            if (stream)
                                Console.Write(assembler.Consume(llama_token_to_piece(mdl, token)));

                            if (request.T1 == default)
                                request.T1 = DateTime.Now;

                            if (stopTokens.Contains(token))
                                request.T2 = DateTime.Now;
                        }
                    }
                }

                foreach (var r in requests.Where(r => stopTokens.Contains(r.Tokens[r.PosToken - 1])))
                {
                    llama_kv_cache_seq_rm(ctx, r.Id, 0, -1);

                    if (!stream)
                    {
                        Console.WriteLine(new String('=', 128));
                        Console.WriteLine($"request id {r.Id} [{r.PosToken / r.Elapsed.TotalMilliseconds * 1000:F2} t/s]");
                        Console.WriteLine(new String('-', 128));
                        Console.WriteLine(Encoding.UTF8.GetString(r.Tokens.Take(r.PosToken).SelectMany(token => llama_token_to_piece(mdl, token).ToArray()).ToArray()));
                        Console.WriteLine(new String('=', 128));
                    }
                    else
                    {
                        Console.WriteLine();
                    }
                }

                requests.RemoveAll(r => stopTokens.Contains(r.Tokens[r.PosToken - 1]));
            }

            llama_batch_free(bat);
            llama_free(ctx);
            llama_free_model(mdl);

            llama_backend_free();
        }

        /// <summary>
        /// This sample serves for testing the library wrapped native core functionality.
        /// </summary>
        /// <param name="args"></param>
        static async Task RunSampleAsync(string[] args)
        {
            using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });
            llm.LoadModel(args[0], new ModelOptions { Seed = 0, GpuLayers = 64 });

            var extraStopTokens = new[] { llm.Tokenize("<|end_of_turn|>", false, true)[0] };

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { cancellationTokenSource.Cancel(); e.Cancel = true; };

            var inputTask = Task.Run(
                async () =>
                {
                    await llm.WaitForRunningAsync();
                    Console.WriteLine("\nEngine ready and model loaded.");

                    while (true)
                    {
                        if (cancellationTokenSource.IsCancellationRequested)
                            cancellationTokenSource = new();

                        Console.Write("> ");
                        var prompt = (Console.ReadLine() ?? String.Empty).Replace("\\n", "\n");
                        if (String.IsNullOrWhiteSpace(prompt))
                            break;

                        // Bulk parallel requests without streaming
                        // i.e. `/load "prompt_file1.txt" "prompt_file2.txt" ...`
                        var match = Regex.Match(prompt, @"\/load\s+("".*?""(?:\s+|$))+");
                        if (match.Success)
                        {
                            var fileNames = Regex.Matches(prompt, "\"(.*?)\"").Select(x => x.Groups[1].Value).ToList();

                            fileNames
                                .Where(fileName => !File.Exists(fileName))
                                .ToList()
                                .ForEach(fileName => Console.WriteLine($"File \"{fileName}\" not found."));

                            var requests = fileNames
                                .Where(fileName => File.Exists(fileName))
                                .Select(fileName => llm.NewRequest(File.ReadAllText(fileName), new SamplingOptions { Temperature = 0.0f }, true, true, extraStopTokens))
                                .Select(
                                    async request =>
                                    {
                                        var response = new StringBuilder();
                                        var assembler = new MultibyteCharAssembler();

                                        await foreach (var token in request.NextToken(cancellationTokenSource.Token))
                                            response.Append(assembler.Consume(token));
                                        response.Append(assembler.Consume());

                                        return (Request: request, Response: response.ToString(), request.Cancelled);
                                    }
                                )
                                .ToList();

                            var results = await Task.WhenAll(requests);

                            Console.WriteLine(new String('=', 196));
                            foreach (var result in results)
                            {
                                Console.WriteLine($"Request {result.Request.GetHashCode()} | Prompting {result.Request.PromptingTime.TotalSeconds} | Sampling {result.Request.SamplingTime.TotalSeconds}");
                                Console.WriteLine(new String('-', 196));
                                Console.WriteLine(result.Request.Prompt);
                                Console.WriteLine(new String('-', 196));
                                Console.WriteLine($"{result.Response}{(result.Cancelled ? " [Cancelled]" : "")}");
                                Console.WriteLine(new String('=', 196));
                            }

                            continue;
                        }

                        // Single request with streaming
                        // i.e. `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello there! How are you today?<|im_end|>\n<|im_start|>assistant\n`
                        var request = llm.NewRequest(prompt, new SamplingOptions { Temperature = 0.0f }, true, true, extraStopTokens);
                        var assembler = new MultibyteCharAssembler();

                        await foreach (var token in request.NextToken(cancellationTokenSource.Token))
                            Console.Write(assembler.Consume(token));
                        Console.WriteLine(assembler.Consume());

                        if (request.Cancelled)
                            Console.WriteLine($" [Cancelled]");
                    }

                    Console.WriteLine($"Shutting down...");
                    await llm.StopAsync();
                }
            );

            await llm.RunAsync();

            await inputTask;
            Console.WriteLine("Bye.");
        }
    }

    file class Request : IEquatable<Request>
    {
        public int Id { get; set; }

        public int PosBatch { get; set; }
        public int PosLogit { get; set; }

        public int PosToken { get; set; }
        public int[] Tokens { get; set; }

        public float MirostatMU = 0.0f;

        public DateTime? T1 { get; set; }
        public DateTime? T2 { get; set; }
        public TimeSpan Elapsed => (T2 - T1) ?? TimeSpan.FromSeconds(0);

        public Request(int n_ctx, ReadOnlySpan<int> tokens)
        {
            this.Tokens = new int[n_ctx];
            tokens.CopyTo(Tokens);
            this.PosToken += tokens.Length;
        }

        public override bool Equals([NotNullWhen(true)] object? obj) => obj is Request request && Equals(request);
        public override int GetHashCode() => Id.GetHashCode();

        // IEquatable<T>
        public bool Equals(Request? other) => other?.Id == this.Id;
    }
}
