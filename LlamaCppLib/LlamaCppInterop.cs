using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    using LlamaContext = System.IntPtr;
    using LlamaToken = System.Int32;

    public static class LlamaCppInterop
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaTokenData
        {
            public LlamaToken id;
            public float logit;
            public float p;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaTokenDataArray
        {
            public nint data;
            public int size;
            public bool sorted;
        }

        public struct LlamaTokenDataArrayManaged
        {
            public List<LlamaTokenData> data;
            public bool sorted;
        }

        public delegate void LlamaProgressCallback(float progress, LlamaContext ctx);

        [StructLayout(LayoutKind.Sequential)]
        public struct LlamaContextParams
        {
            public int n_ctx;
            public int n_parts;
            public int seed;
            [MarshalAs(UnmanagedType.I1)]
            public bool f16_kv;
            [MarshalAs(UnmanagedType.I1)]
            public bool logits_all;
            [MarshalAs(UnmanagedType.I1)]
            public bool vocab_only;
            [MarshalAs(UnmanagedType.I1)]
            public bool use_mmap;
            [MarshalAs(UnmanagedType.I1)]
            public bool use_mlock;
            [MarshalAs(UnmanagedType.I1)]
            public bool embedding;
            public LlamaProgressCallback progress_callback;
            public nint progress_callback_user_data;
        }

        public enum LlamaFType
        {
            LLAMA_FTYPE_ALL_F32 = 0,
            LLAMA_FTYPE_MOSTLY_F16 = 1,
            LLAMA_FTYPE_MOSTLY_Q4_0 = 2,
            LLAMA_FTYPE_MOSTLY_Q4_1 = 3,
            LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,
            LLAMA_FTYPE_MOSTLY_Q4_2 = 5,
            LLAMA_FTYPE_MOSTLY_Q4_3 = 6,
        }

        [DllImport("llama")]
        public static extern LlamaContextParams llama_context_default_params();

        [DllImport("llama")]
        public static extern bool llama_mmap_supported();

        [DllImport("llama")]
        public static extern bool llama_mlock_supported();

        /// <summary>
        /// Various functions for loading a ggml llama model.
        /// Allocate (almost) all memory needed for the model.
        /// </summary>
        /// <param name="path_model">Model file path</param>
        /// <param name="cparams">Parameters to use for loading the model</param>
        /// <returns>LlamaContext on success or null on failure</returns>
        [DllImport("llama")]
        public static extern LlamaContext llama_init_from_file(string path_model, LlamaContextParams cparams);

        /// <summary>
        /// Frees all allocated memory
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        [DllImport("llama")]
        public static extern void llama_free(LlamaContext ctx);

        /// <summary>
        /// TODO: not great API - very likely to change (from llama.cpp)
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="path_lora">Lora adapter file path</param>
        /// <param name="path_base_model">Model file path</param>
        /// <param name="n_threads">nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given</param>
        /// <returns>Returns 0 on success</returns>
        [DllImport("llama", EntryPoint = "llama_model_quantize")]
        public static extern int llama_model_quantize(string fname_inp, string fname_out, LlamaFType ftype, int nthread);

        /// <summary>
        /// Apply a LoRA adapter to a loaded model
        /// The model needs to be reloaded before applying a new adapter, otherwise the adapter
        /// will be applied on top of the previous one
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="path_lora">path_base_model is the path to a higher quality model to use as a base for the layers modified by the adapter. Can be NULL to use the current loaded model.</param>
        /// <param name="path_base_model">Model file path</param>
        /// <param name="n_threads">nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given</param>
        /// <returns>Returns 0 on success</returns>
        [DllImport("llama")]
        public static extern int llama_apply_lora_from_file(LlamaContext ctx, string path_lora, string? path_base_model, int n_threads);

        /// <summary>
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <returns>Returns the number of tokens in the KV cache</returns>
        [DllImport("llama")]
        public static extern int llama_get_kv_cache_token_count(LlamaContext ctx);

        /// <summary>
        /// Sets the current rng seed.
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="seed">Seed</param>
        [DllImport("llama")]
        public static extern void llama_set_rng_seed(LlamaContext ctx, int seed);

        /// <summary>
        /// Returns the size in bytes of the state (rng, logits, embedding and kv_cache)
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <returns>Returns the size in bytes of the state (rng, logits, embedding and kv_cache)</returns>
        [DllImport("llama")]
        public static extern int llama_get_state_size(LlamaContext ctx);

        /// <summary>
        /// Copies the state to the specified destination address.
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="dest">Destination needs to have allocated enough memory.</param>
        /// <returns>Returns the number of bytes copied</returns>
        [DllImport("llama")]
        public static extern int llama_copy_state_data(LlamaContext ctx, nint dest);

        /// <summary>
        /// Set the state reading from the specified address
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="src">State source</param>
        /// <returns>Returns the number of bytes read</returns>
        [DllImport("llama")]
        public static extern int llama_set_state_data(LlamaContext ctx, nint src);

        /// <summary>
        /// Load session file
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="path_session"></param>
        /// <param name="tokens_out"></param>
        /// <param name="n_token_capacity"></param>
        /// <param name="n_token_count_out"></param>
        /// <returns></returns>
        [DllImport("llama", EntryPoint = "llama_load_session_file")]
        private static extern bool _llama_load_session_file(LlamaContext ctx, string path_session, nint tokens_out, int n_token_capacity, out int n_token_count_out);

        public static bool llama_load_session_file(LlamaContext ctx, string path_session, out List<LlamaToken> tokens_out, int n_token_count)
        {
            using var nmem = new NativeHGlobal(n_token_count * sizeof(LlamaToken));
            var result = _llama_load_session_file(ctx, path_session, nmem.Ptr, n_token_count, out var n_token_count_out);

            var res = new LlamaToken[n_token_count_out];
            Marshal.Copy(nmem.Ptr, res, 0, res.Length);
            tokens_out = new(res);

            return result;
        }

        /// <summary>
        /// Save session file
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="path_session"></param>
        /// <param name="tokens"></param>
        /// <param name="n_token_count"></param>
        /// <returns></returns>
        [DllImport("llama", EntryPoint = "llama_save_session_file")]
        private static extern bool _llama_save_session_file(LlamaContext ctx, string path_session, nint tokens, int n_token_count);

        public static bool llama_save_session_file(LlamaContext ctx, string path_session, List<LlamaToken> tokens)
        {
            using var nmem = new NativeHGlobal(tokens.Count * sizeof(LlamaToken));
            Marshal.Copy(tokens.ToArray(), 0, nmem.Ptr, tokens.Count);
            var result = _llama_save_session_file(ctx, path_session, nmem.Ptr, tokens.Count);
            return result;
        }

        /// <summary>
        /// Run the llama inference to obtain the logits and probabilities for the next token.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="tokens">tokens + n_tokens is the provided batch of new tokens to process</param>
        /// <param name="n_tokens"></param>
        /// <param name="n_past">n_past is the number of tokens to use from previous eval calls</param>
        /// <param name="n_threads">nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given</param>
        /// <returns>Returns 0 on success</returns>
        [DllImport("llama", EntryPoint = "llama_eval")]
        private static extern int _llama_eval(LlamaContext ctx, nint tokens, int n_tokens, int n_past, int n_threads);

        public static int llama_eval(LlamaContext ctx, List<LlamaToken> tokens, int n_past, int n_threads)
        {
            var len = tokens.Count;
            using var nmem = new NativeHGlobal(len * sizeof(LlamaToken));
            Marshal.Copy(tokens.ToArray(), 0, nmem.Ptr, len);
            var res = _llama_eval(ctx, nmem.Ptr, len, n_past, n_threads);
            return res;
        }

        /// <summary>
        /// Convert the provided text into tokens.
        /// The tokens pointer must be large enough to hold the resulting tokens.
        /// TODO: not sure if correct (from llama.cpp)
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="text"></param>
        /// <param name="tokens"></param>
        /// <param name="n_max_tokens"></param>
        /// <param name="add_bos"></param>
        /// <returns>Returns the number of tokens on success, no more than n_max_tokens and returns a negative number on failure - the number of tokens that would have been returned</returns>
        [DllImport("llama", EntryPoint = "llama_tokenize")]
        private static extern int _llama_tokenize(LlamaContext ctx, string text, nint tokens, int n_max_tokens, bool add_bos);

        public static List<LlamaToken> llama_tokenize(LlamaContext ctx, string text, bool add_bos = false)
        {
            var len = text.Length + (add_bos ? 1 : 0);
            using var nmem = new NativeHGlobal(len * sizeof(LlamaToken));
            var cnt = _llama_tokenize(ctx, text, nmem.Ptr, len, add_bos);

            if (cnt == 0)
                return new();

            var res = new LlamaToken[cnt];
            Marshal.Copy(nmem.Ptr, res, 0, res.Length);
            return new(res);
        }

        [DllImport("llama")]
        public static extern int llama_n_vocab(LlamaContext ctx);

        [DllImport("llama")]
        public static extern int llama_n_ctx(LlamaContext ctx);

        [DllImport("llama")]
        public static extern int llama_n_embd(LlamaContext ctx);

        /// <summary>
        /// Token logits obtained from the last call to llama_eval()
        /// The logits for the last token are stored in the last row
        /// Can be mutated in order to change the probabilities of the next token
        /// Rows: n_tokens
        /// Cols: n_vocab
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <returns>List of floats (logits)</returns>
        [DllImport("llama", EntryPoint = "llama_get_logits")]
        private static extern nint _llama_get_logits(LlamaContext ctx);

        public static List<float> llama_get_logits(LlamaContext ctx)
        {
            var len = llama_n_vocab(ctx);
            var ptr = _llama_get_logits(ctx);
            var res = new float[len];
            Marshal.Copy(ptr, res, 0, len);
            return new(res);
        }

        /// <summary>
        /// Get the embeddings for the input
        /// shape: [n_embd] (1-dimensional)
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <returns>List of floats (embeddings)</returns>
        [DllImport("llama", EntryPoint = "llama_get_embeddings")]
        private static extern nint _llama_get_embeddings(LlamaContext ctx);

        public static List<float> llama_get_embeddings(LlamaContext ctx)
        {
            var len = llama_n_embd(ctx);
            var ptr = _llama_get_embeddings(ctx);

            if (ptr == nint.Zero)
                return new();

            var res = new float[len];
            Marshal.Copy(ptr, res, 0, len);
            return new(res);
        }

        /// <summary>
        /// Token Id -> String. Uses the vocabulary in the provided context
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="token">Token ID to convert</param>
        /// <returns>Text token</returns>
        [DllImport("llama", EntryPoint = "llama_token_to_str")]
        private static extern nint _llama_token_to_str(LlamaContext ctx, LlamaToken token);

        public static string llama_token_to_str(LlamaContext ctx, LlamaToken token) => Marshal.PtrToStringUTF8(_llama_token_to_str(ctx, token)) ?? String.Empty;

        /// <summary>
        /// Special tokens
        /// </summary>
        /// <returns>Beginning of stream token</returns>
        [DllImport("llama")]
        public static extern LlamaToken llama_token_bos();

        /// <summary>
        /// Special tokens
        /// </summary>
        /// <returns>End of stream token</returns>
        [DllImport("llama")]
        public static extern LlamaToken llama_token_eos();

        /// <summary>
        /// Special tokens
        /// </summary>
        /// <returns>End of stream token</returns>
        [DllImport("llama")]
        public static extern LlamaToken llama_token_nl();

        /// <summary>
        /// Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <param name="last_tokens"></param>
        /// <param name="last_tokens_size"></param>
        /// <param name="penalty"></param>
        [DllImport("llama", EntryPoint = "llama_sample_repetition_penalty")]
        private static extern void _llama_sample_repetition_penalty(LlamaContext ctx, ref LlamaTokenDataArray candidates, nint last_tokens, int last_tokens_size, float penalty);

        public static void llama_sample_repetition_penalty(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, List<LlamaToken> last_tokens, float penalty)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem1 = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem1.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem1.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            using var nmem2 = new NativeHGlobal(last_tokens.Count * sizeof(LlamaToken));
            Marshal.Copy(last_tokens.ToArray(), 0, nmem2.Ptr, last_tokens.Count);

            _llama_sample_repetition_penalty(ctx, ref _candidates, nmem2.Ptr, last_tokens.Count, penalty);
        }

        /// <summary>
        /// Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <param name="last_tokens"></param>
        /// <param name="last_tokens_size"></param>
        /// <param name="alpha_frequency"></param>
        /// <param name="alpha_presence"></param>
        [DllImport("llama", EntryPoint = "llama_sample_frequency_and_presence_penalties")]
        private static extern void _llama_sample_frequency_and_presence_penalties(LlamaContext ctx, ref LlamaTokenDataArray candidates, nint last_tokens, int last_tokens_size, float alpha_frequency, float alpha_presence);

        public static void llama_sample_frequency_and_presence_penalties(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, List<LlamaToken> last_tokens, float alpha_frequency, float alpha_presence)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem1 = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem1.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem1.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            using var nmem2 = new NativeHGlobal(last_tokens.Count * sizeof(LlamaToken));
            Marshal.Copy(last_tokens.ToArray(), 0, nmem2.Ptr, last_tokens.Count);

            _llama_sample_frequency_and_presence_penalties(ctx, ref _candidates, nmem2.Ptr, last_tokens.Count, alpha_frequency, alpha_presence);
        }

        /// <summary>
        /// Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        [DllImport("llama", EntryPoint = "llama_sample_softmax")]
        private static extern void _llama_sample_softmax(LlamaContext ctx, ref LlamaTokenDataArray candidates);

        public static void llama_sample_softmax(LlamaContext ctx, LlamaTokenDataArrayManaged candidates)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            _llama_sample_softmax(ctx, ref _candidates);
        }

        /// <summary>
        /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <param name="k"></param>
        /// <param name="min_keep"></param>
        [DllImport("llama", EntryPoint = "llama_sample_top_k")]
        private static extern void _llama_sample_top_k(LlamaContext ctx, ref LlamaTokenDataArray candidates, int k, int min_keep = 1);

        public static void llama_sample_top_k(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, int k, int min_keep = 1)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            _llama_sample_top_k(ctx, ref _candidates, k, min_keep);
        }

        /// <summary>
        /// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <param name="p"></param>
        /// <param name="min_keep"></param>
        [DllImport("llama", EntryPoint = "llama_sample_top_p")]
        public static extern void _llama_sample_top_p(LlamaContext ctx, ref LlamaTokenDataArray candidates, float p, int min_keep = 1);

        public static void llama_sample_top_p(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, float p, int min_keep = 1)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            _llama_sample_top_p(ctx, ref _candidates, p, min_keep);
        }

        /// <summary>
        /// Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <param name="z"></param>
        /// <param name="min_keep"></param>
        [DllImport("llama", EntryPoint = "llama_sample_tail_free")]
        private static extern void _llama_sample_tail_free(LlamaContext ctx, ref LlamaTokenDataArray candidates, float z, int min_keep = 1);

        public static void llama_sample_tail_free(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, float z, int min_keep = 1)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            _llama_sample_tail_free(ctx, ref _candidates, z, min_keep);
        }

        /// <summary>
        /// Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <param name="p"></param>
        /// <param name="min_keep"></param>
        [DllImport("llama", EntryPoint = "llama_sample_typical")]
        private static extern void _llama_sample_typical(LlamaContext ctx, ref LlamaTokenDataArray candidates, float p, int min_keep = 1);

        public static void llama_sample_typical(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, float p, int min_keep = 1)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            _llama_sample_typical(ctx, ref _candidates, p, min_keep);
        }

        [DllImport("llama", EntryPoint = "llama_sample_temperature")]
        private static extern void _llama_sample_temperature(LlamaContext ctx, ref LlamaTokenDataArray candidates, float temp);

        public static void llama_sample_temperature(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, float temp)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            _llama_sample_temperature(ctx, ref _candidates, temp);
        }

        /// <summary>
        /// Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates">A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.</param>
        /// <param name="tau">The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.</param>
        /// <param name="eta">The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.</param>
        /// <param name="m">The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.</param>
        /// <param name="mu">Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.</param>
        /// <returns></returns>
        [DllImport("llama", EntryPoint = "llama_sample_token_mirostat")]
        private static extern LlamaToken _llama_sample_token_mirostat(LlamaContext ctx, ref LlamaTokenDataArray candidates, float tau, float eta, int m, ref float mu);

        public static LlamaToken llama_sample_token_mirostat(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, float tau, float eta, int m, ref float mu)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_mirostat(ctx, ref _candidates, tau, eta, m, ref mu);
        }

        /// <summary>
        /// Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates">A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.</param>
        /// <param name="tau">The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.</param>
        /// <param name="eta">The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.</param>
        /// <param name="mu">Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.</param>
        /// <returns></returns>
        [DllImport("llama", EntryPoint = "llama_sample_token_mirostat_v2")]
        private static extern LlamaToken _llama_sample_token_mirostat_v2(LlamaContext ctx, ref LlamaTokenDataArray candidates, float tau, float eta, ref float mu);

        public static LlamaToken llama_sample_token_mirostat_v2(LlamaContext ctx, LlamaTokenDataArrayManaged candidates, float tau, float eta, ref float mu)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_mirostat_v2(ctx, ref _candidates, tau, eta, ref mu);
        }

        /// <summary>
        /// Selects the token with the highest probability.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <returns></returns>
        [DllImport("llama", EntryPoint = "llama_sample_token_greedy")]
        private static extern LlamaToken _llama_sample_token_greedy(LlamaContext ctx, ref LlamaTokenDataArray candidates);

        public static LlamaToken llama_sample_token_greedy(LlamaContext ctx, LlamaTokenDataArrayManaged candidates)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_greedy(ctx, ref _candidates);
        }

        /// <summary>
        /// Randomly selects a token from the candidates based on their probabilities.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="candidates"></param>
        /// <returns></returns>
        [DllImport("llama", EntryPoint = "llama_sample_token")]
        private static extern LlamaToken _llama_sample_token(LlamaContext ctx, ref LlamaTokenDataArray candidates);

        public static LlamaToken llama_sample_token(LlamaContext ctx, LlamaTokenDataArrayManaged candidates)
        {
            var usize = Marshal.SizeOf<LlamaTokenData>();
            var tsize = usize * candidates.data.Count;

            using var nmem = new NativeHGlobal(tsize);

            for (var i = 0; i < candidates.data.Count; i++)
                Marshal.StructureToPtr(candidates.data[i], nmem.Ptr + i * usize, false);

            var _candidates = new LlamaTokenDataArray()
            {
                data = nmem.Ptr,
                size = candidates.data.Count,
                sorted = candidates.sorted,
            };

            return _llama_sample_token(ctx, ref _candidates);
        }

        /// <summary>
        /// Performance information
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        [DllImport("llama")]
        public static extern void llama_print_timings(LlamaContext ctx);

        /// <summary>
        /// Performance information
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        [DllImport("llama")]
        public static extern void llama_reset_timings(LlamaContext ctx);

        /// <summary>
        /// Print system information
        /// </summary>
        /// <returns>System information</returns>
        [DllImport("llama", EntryPoint = "llama_print_system_info")]
        private static extern nint _llama_print_system_info();

        public static string llama_print_system_info() => Marshal.PtrToStringUTF8(_llama_print_system_info()) ?? String.Empty;
    }
}
