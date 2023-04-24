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
            public float p;
            public float plog;
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
            public IntPtr progress_callback_user_data;
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
        //[DllImport("llama")]
        //public static extern int llama_model_quantize(string fname_inp, out string fname_out, LlamaFType ftype, int nthread);

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
        public static extern int llama_apply_lora_from_file(LlamaContext ctx, string path_lora, string path_base_model, int n_threads);

        /// <summary>
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <returns>Returns the number of tokens in the KV cache</returns>
        [DllImport("llama")]
        public static extern int llama_get_kv_cache_token_count(LlamaContext ctx);

        /// <summary>
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
        //[DllImport("llama")]
        //public static extern int llama_copy_state_data(LlamaContext ctx, nint dest);

        /// <summary>
        /// Set the state reading from the specified address
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="src">State source</param>
        /// <returns>Returns the number of bytes read</returns>
        //[DllImport("llama")]
        //public static extern int llama_set_state_data(LlamaContext ctx, nint src);

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
            var ptr = Marshal.AllocHGlobal(len * sizeof(LlamaToken));
            Marshal.Copy(tokens.ToArray(), 0, ptr, len);
            var res = _llama_eval(ctx, ptr, len, n_past, n_threads);
            Marshal.FreeHGlobal(ptr);
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
            var ptr = Marshal.AllocHGlobal(len * sizeof(LlamaToken));
            var cnt = _llama_tokenize(ctx, text, ptr, len, add_bos);

            if (cnt == 0)
                return new();

            var res = new LlamaToken[cnt];
            Marshal.Copy(ptr, res, 0, res.Length);
            Marshal.FreeHGlobal(ptr);
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
            var len = llama_n_ctx(ctx);
            var ptr = _llama_get_logits(ctx);

            if (ptr == nint.Zero)
                return new();

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

        public static string llama_token_to_str(LlamaContext ctx, LlamaToken token)
        {
            var ptr = _llama_token_to_str(ctx, token);
            return Marshal.PtrToStringUTF8(ptr) ?? String.Empty;
        }

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
        /// TODO: improve the last_n_tokens interface ? (from llama.cpp)
        /// </summary>
        /// <param name="ctx">LlamaContext</param>
        /// <param name="last_n_tokens_data"></param>
        /// <param name="last_n_tokens_size"></param>
        /// <param name="top_k"></param>
        /// <param name="top_p"></param>
        /// <param name="temp"></param>
        /// <param name="repeat_penalty"></param>
        /// <returns>The next token</returns>
        [DllImport("llama", EntryPoint = "llama_sample_top_p_top_k")]
        private static extern LlamaToken _llama_sample_top_p_top_k(LlamaContext ctx, nint last_n_tokens_data, int last_n_tokens_size, int top_k, float top_p, float temp, float repeat_penalty);

        public static LlamaToken llama_sample_top_p_top_k(LlamaContext ctx, List<LlamaToken> last_n_tokens_data, int top_k, float top_p, float temp, float repeat_penalty)
        {
            var len = last_n_tokens_data.Count;
            var ptr = Marshal.AllocHGlobal(len * sizeof(LlamaToken));
            Marshal.Copy(last_n_tokens_data.ToArray(), 0, ptr, len);
            var res = _llama_sample_top_p_top_k(ctx, ptr, len, top_k, top_p, temp, repeat_penalty);
            Marshal.FreeHGlobal(ptr);
            return res;
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
        [DllImport("llama")]
        public static extern string llama_print_system_info();
    }
}
