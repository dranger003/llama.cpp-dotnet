using static LlamaCppLib.Native;
using static LlamaCppLib.Interop;

namespace LlamaCppCli
{
    using llama_token = System.Int32;

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

        static unsafe void RunSampleEmbedding(string[] args)
        {
            var mparams = llama_model_default_params();
            mparams.n_gpu_layers = args.Length > 1 ? Int32.Parse(args[1]) : 0;
            mparams.progress_callback = &ProgressCallback;

            var cparams = llama_context_default_params();
            cparams.seed = unchecked((uint)-1);
            cparams.n_ctx = 4096;
            cparams.n_batch = 4096;
            cparams.n_threads = 8;
            cparams.n_threads_batch = 8;
            cparams.embeddings = true ? 1 : 0;

            llama_backend_init();
            llama_numa_init(ggml_numa_strategy.GGML_NUMA_STRATEGY_DISABLED);

            var mdl = llama_load_model_from_file(args[0], mparams);
            var ctx = llama_new_context_with_model(mdl, cparams);

            var GetEmbeddings = (string text) =>
            {
                var batch = llama_batch_init((int)cparams.n_batch, 0, 1);

                var embd_inp = llama_tokenize(mdl, text, true, false, true);
                for (var i = 0; i < embd_inp.Length; i++)
                    llama_batch_add(ref batch, embd_inp[i], i, [0], i == embd_inp.Length - 1);

                llama_kv_cache_clear(ctx);
                var res = llama_decode(ctx, batch);

                float* embd = null;
                for (var i = 0; i < batch.n_tokens; i++)
                {
                    if (batch.logits[i] == 0)
                        continue;

                    embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
                    if (embd == null)
                        embd = llama_get_embeddings_ith(ctx, i);
                }

                return new Span<float>(embd, llama_n_embd(mdl));
            };

            var CosineSimilarity = (float[] vec1, float[] vec2) =>
            {
                if (vec1.Length != vec2.Length)
                    throw new ArgumentException("Vectors must be of the same size.");

                var dotProduct = vec1.Zip(vec2, (a, b) => a * b).Sum();
                var normA = Math.Sqrt(vec1.Sum(a => Math.Pow(a, 2)));
                var normB = Math.Sqrt(vec2.Sum(b => Math.Pow(b, 2)));

                if (normA == 0.0 || normB == 0.0)
                    throw new ArgumentException("Vectors must not be zero vectors.");

                return (float)(dotProduct / (normA * normB));
            };

            var documents = new[]
            {
                "Carson City is the capital city of the American state of Nevada. At the  2010 United States Census, Carson City had a population of 55,274.",
                "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
                "Charlotte Amalie is the capital and largest city of the United States Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
                "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
                "Proteins are the building blocks of muscle tissue and other important structures in chickens, helping them grow strong and healthy!",
                "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
                "North Dakota is a state in the United States. 672,591 people lived in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
                "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
                "The World Summit on Climate Change is an international conference aimed at addressing global warming and promoting sustainable development efforts around the globe.",
                "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
            };

            // https://huggingface.co/intfloat/e5-mistral-7b-instruct#usage
            var template = "Instruct: {0}\nQuery: {1}";
            var task = "Given a web search query, retrieve relevant passages that answer the query";

            var queries = new[]
            {
                "how much protein should a female eat",
                "summit define",
                "What is the capital of the United States?",
            }.Select(x => String.Format(template, task, x)).ToList();

            var documentsEmbeddings = documents
                .Select(x => GetEmbeddings(x).ToArray())
                .ToList();

            foreach (var query in queries)
            {
                var queryEmbeddings = GetEmbeddings(query).ToArray();

                var cosineSimilarities = documentsEmbeddings
                    .Select(documentEmbeddings => CosineSimilarity(queryEmbeddings, documentEmbeddings))
                    .ToList();

                var topResults = documents
                    .Zip(cosineSimilarities, (x, similarity) => new { Document = x, CosineSimilarity = similarity })
                    .OrderByDescending(x => x.CosineSimilarity)
                    .Take(3)
                    .ToList();

                Console.WriteLine($"\n[{query}]");
                topResults.ForEach(result => Console.WriteLine($"    [{result.CosineSimilarity * 100:0.00}%][{result.Document.TruncateWithEllipsis()}]"));
            }

            llama_free(ctx);
            llama_free_model(mdl);

            llama_backend_free();
        }
    }
}
