using System.Text;

using static LlamaCppLib.Native;
using static LlamaCppLib.Interop;

namespace LlamaCppCli
{
    using llama_context = System.IntPtr;
    using llama_token = System.Int32;
    using llama_seq_id = System.Int32;

    internal partial class Program
    {
        static async Task RunSampleEmbeddingAsync(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine($"Usage: RunSampleEmbeddingAsync <ModelPath> [GpuLayers]");
                return;
            }

            RunSampleEmbedding(args);
            await Task.CompletedTask;
        }

        // Tested using https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF
        static unsafe void RunSampleEmbedding(string[] args)
        {
            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = args.Length > 1 ? Int32.Parse(args[1]) : 0;

            var cparams = llama_context_default_params();
            cparams.n_ctx = 2048;
            cparams.embeddings = true ? 1 : 0;
            cparams.pooling_type = _llama_pooling_type.LLAMA_POOLING_TYPE_MEAN;

            // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
            var embd_normalize = 2;

            var mdl = llama_load_model_from_file(args[0], mparams);
            var ctx = llama_new_context_with_model(mdl, cparams);

            var n_ctx_train = llama_n_ctx_train(mdl);
            var n_ctx = llama_n_ctx(ctx);

            var pooling_type = llama_pooling_type(ctx);

            if (llama_model_has_encoder(mdl) && llama_model_has_decoder(mdl))
            {
                Console.WriteLine("computing embeddings in encoder-decoder models is not supported.");
                return;
            }

            if (n_ctx > n_ctx_train)
            {
                Console.WriteLine($"warning: model was trained on only {n_ctx_train} context tokens ({n_ctx} specified)");
            }

            var n_batch = (int)cparams.n_batch;
            if (n_batch < n_ctx)
            {
                Console.WriteLine($"error: cparams.n_batch < n_ctx ({cparams.n_batch} < {n_ctx})");
                return;
            }

            var prompts = new[]
            {
                "Hello world!",
            };

            var inputs = new List<llama_token[]>();
            foreach (var prompt in prompts)
            {
                var inp = llama_tokenize(mdl, Encoding.UTF8.GetBytes(prompt), true, true);
                if (inp.Length > cparams.n_batch)
                {
                    Console.WriteLine($"number of tokens in input line ({inp.Length}) exceeds batch size ({cparams.n_batch}), increase batch size and re-run.");
                    return;
                }
                inputs.Add(inp);
            }

            var n_prompts = prompts.Length;
            var batch = llama_batch_init(n_batch, 0, 1);

            var n_embd_count = 0;
            if (pooling_type == _llama_pooling_type.LLAMA_POOLING_TYPE_NONE)
            {
                for (var k = 0; k < n_prompts; k++)
                {
                    n_embd_count += inputs[k].Length;
                }
            }
            else
            {
                n_embd_count = n_prompts;
            }

            var n_embd = llama_n_embd(mdl);
            var embeddings = new float[n_embd_count * n_embd];

            fixed (float* emb = &embeddings[0])
            {
                float* @out = null;
                var e = 0;
                var s = 0;

                for (var k = 0; k < n_prompts; k++)
                {
                    var inp = inputs[k];
                    var n_toks = inp.Length;

                    if (batch.n_tokens + n_toks > n_batch)
                    {
                        @out = emb + e * n_embd;
                        batch_decode(ctx, ref batch, @out, s, n_embd, embd_normalize);
                        e += pooling_type == _llama_pooling_type.LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
                        s = 0;
                        llama_batch_clear(ref batch);
                    }

                    batch_add_seq(ref batch, inp, s);
                    s += 1;
                }

                @out = emb + e * n_embd;
                batch_decode(ctx, ref batch, @out, s, n_embd, embd_normalize);

                if (pooling_type == _llama_pooling_type.LLAMA_POOLING_TYPE_NONE)
                {
                    for (var j = 0; j < n_embd_count; j++)
                    {
                        Console.Write($"embedding {j}: ");
                        for (var i = 0; i < Math.Min(3, n_embd); i++)
                        {
                            if (embd_normalize == 0)
                            {
                                Console.Write($"{emb[j * n_embd + i],6:0} ");
                            }
                            else
                            {
                                Console.Write($"{emb[j * n_embd + i],9:F6} ");
                            }
                        }
                        Console.Write(" ... ");
                        for (var i = n_embd - 3; i < n_embd; i++)
                        {
                            if (embd_normalize == 0)
                            {
                                Console.Write($"{emb[j * n_embd + i],6:0} ");
                            }
                            else
                            {
                                Console.Write($"{emb[j * n_embd + i],9:F6} ");
                            }
                        }
                        Console.WriteLine();
                    }
                }
                else if (pooling_type == _llama_pooling_type.LLAMA_POOLING_TYPE_RANK)
                {
                    for (var j = 0; j < n_embd_count; j++)
                    {
                        Console.WriteLine($"rerank score {j}: {emb[j * n_embd],8:F3}");
                    }
                }
                else
                {
                    // print the first part of the embeddings or for a single prompt, the full embedding
                    for (var j = 0; j < n_prompts; j++)
                    {
                        Console.Write($"embedding {j}: ");
                        for (var i = 0; i < (n_prompts > 1 ? Math.Min(16, n_embd) : n_embd); i++)
                        {
                            if (embd_normalize == 0)
                            {
                                Console.Write($"{emb[j * n_embd + i],6:0} ");
                            }
                            else
                            {
                                Console.Write($"{emb[j * n_embd + i],9:F6} ");
                            }
                        }
                        Console.WriteLine();
                    }

                    // print cosine similarity matrix
                    if (n_prompts > 1)
                    {
                        Console.Write("\n");
                        Console.Write("cosine similarity matrix:\n\n");
                        for (var i = 0; i < n_prompts; i++)
                        {
                            Console.Write($"{prompts[i][..6]} ");
                        }
                        Console.WriteLine();
                        for (var i = 0; i < n_prompts; i++)
                        {
                            for (var j = 0; j < n_prompts; j++)
                            {
                                var sim = common_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                                Console.Write($"{sim,6:F2} ");
                            }
                            Console.WriteLine($"{prompts[i][..10]}");
                        }
                    }
                }
            }

            //var documents = new[]
            //{
            //    "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
            //    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
            //    "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
            //    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
            //    "Proteins are the building blocks of muscle tissue and other important structures in chickens, helping them grow strong and healthy!",
            //    "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
            //    "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
            //    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            //    "The World Summit on Climate Change is an international conference aimed at addressing global warming and promoting sustainable development efforts around the globe.",
            //    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
            //};

            //var queries = new[]
            //{
            //    "how much protein should a female eat",
            //    "summit define",
            //    "What is the capital of the United States?",
            //}.ToList();

            //var documentsEmbeddings = documents
            //    .Select(x => GetEmbeddings(x).ToArray())
            //    .ToList();

            //foreach (var query in queries)
            //{
            //    var queryEmbeddings = GetEmbeddings(query).ToArray();

            //    var cosineSimilarities = documentsEmbeddings
            //        .Select(documentEmbeddings => TensorPrimitives.CosineSimilarity(queryEmbeddings, documentEmbeddings))
            //        .ToList();

            //    var topResults = documents
            //        .Zip(cosineSimilarities, (x, similarity) => new { Document = x, CosineSimilarity = similarity })
            //        .OrderByDescending(x => x.CosineSimilarity)
            //        .Take(3)
            //        .ToList();

            //    Console.WriteLine($"\n[{query}]");
            //    topResults.ForEach(result => Console.WriteLine($"    [{result.CosineSimilarity * 100:0.00}%][{result.Document.TruncateWithEllipsis()}]"));
            //}

            llama_batch_free(batch);

            llama_free(ctx);
            llama_free_model(mdl);
        }

        static unsafe void batch_decode(llama_context ctx, ref llama_batch batch, float* output, int n_seq, int n_embd, int embd_norm)
        {
            var pooling_type = llama_pooling_type(ctx);
            var model = llama_get_model(ctx);

            // clear previous kv_cache values (irrelevant for embeddings)
            llama_kv_cache_clear(ctx);

            // run model
            Console.WriteLine($"n_tokens = {batch.n_tokens}, n_seq = {n_seq}");
            if (llama_model_has_encoder(model) && !llama_model_has_decoder(model))
            {
                // encoder-only model
                if (llama_encode(ctx, batch) < 0)
                {
                    Console.WriteLine("failed to encode");
                }
            }
            else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model))
            {
                // decoder-only model
                if (llama_decode(ctx, batch) < 0)
                {
                    Console.WriteLine("failed to decode");
                }
            }

            for (var i = 0; i < batch.n_tokens; i++)
            {
                if (batch.logits[i] == 0)
                    continue;

                float* embd = null;
                var embd_pos = 0;

                if (pooling_type == _llama_pooling_type.LLAMA_POOLING_TYPE_NONE)
                {
                    // try to get token embeddings
                    embd = llama_get_embeddings_ith(ctx, i);
                    embd_pos = i;
                    //GGML_ASSERT(embd != NULL && "failed to get token embeddings");
                }
                else
                {
                    // try to get sequence embeddings - supported only when pooling_type is not NONE
                    embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
                    embd_pos = batch.seq_id[i][0];
                    //GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
                }

                float* @out = output + embd_pos * n_embd;
                common_embd_normalize(embd, @out, n_embd, embd_norm);
            }
        }

        static void batch_add_seq(ref llama_batch batch, llama_token[] tokens, llama_seq_id seq_id)
        {
            var n_tokens = tokens.Length;
            for (var i = 0; i < n_tokens; i++)
            {
                llama_batch_add(ref batch, tokens[i], i, [seq_id], true);
            }
        }

        static unsafe void common_embd_normalize(float* inp, float* @out, int n, int embd_norm)
        {
            var sum = 0.0;

            switch (embd_norm)
            {
                case -1: // no normalisation
                    sum = 1.0;
                    break;
                case 0: // max absolute
                    for (var i = 0; i < n; i++)
                    {
                        if (sum < Math.Abs(inp[i])) sum = Math.Abs(inp[i]);
                    }
                    sum /= 32760.0; // make an int16 range
                    break;
                case 2: // euclidean
                    for (var i = 0; i < n; i++)
                    {
                        sum += inp[i] * inp[i];
                    }
                    sum = Math.Sqrt(sum);
                    break;
                default: // p-norm (euclidean is p-norm p=2)
                    for (var i = 0; i < n; i++)
                    {
                        sum += Math.Pow(Math.Abs(inp[i]), embd_norm);
                    }
                    sum = Math.Pow(sum, 1.0 / embd_norm);
                    break;
            }

            var norm = (float)(sum > 0.0 ? 1.0 / sum : 0.0f);

            for (var i = 0; i < n; i++)
            {
                @out[i] = inp[i] * norm;
            }
        }

        static unsafe float common_embd_similarity_cos(float* embd1, float* embd2, int n)
        {
            var sum = 0.0;
            var sum1 = 0.0;
            var sum2 = 0.0;

            for (var i = 0; i < n; i++)
            {
                sum += embd1[i] * embd2[i];
                sum1 += embd1[i] * embd1[i];
                sum2 += embd2[i] * embd2[i];
            }

            // Handle the case where one or both vectors are zero vectors
            if (sum1 == 0.0 || sum2 == 0.0)
            {
                if (sum1 == 0.0 && sum2 == 0.0)
                {
                    return 1.0f; // two zero vectors are similar
                }
                return 0.0f;
            }

            return (float)(sum / (Math.Sqrt(sum1) * Math.Sqrt(sum2)));
        }
    }
}
