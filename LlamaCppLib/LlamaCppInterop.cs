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

        [DllImport("llama")]
        public static extern LlamaContext llama_init_from_file(string path_model, LlamaContextParams cparams);

        [DllImport("llama")]
        public static extern void llama_free(LlamaContext ctx);

        //[DllImport("llama")]
        //public static extern int llama_model_quantize(string fname_inp, out string fname_out, LlamaFType ftype, int nthread);

        [DllImport("llama")]
        public static extern int llama_apply_lora_from_file(LlamaContext ctx, string path_lora, string path_base_model, int n_threads);

        [DllImport("llama")]
        public static extern IntPtr llama_get_kv_cache(LlamaContext ctx);

        [DllImport("llama")]
        public static extern int llama_get_kv_cache_size(LlamaContext ctx);

        [DllImport("llama")]
        public static extern int llama_get_kv_cache_token_count(LlamaContext ctx);

        //[DllImport("llama")]
        //public static extern void llama_set_kv_cache(LlamaContext ctx, nint kv_cache, int n_size, int n_token_count);

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

        [DllImport("llama", EntryPoint = "llama_token_to_str")]
        private static extern nint _llama_token_to_str(LlamaContext ctx, LlamaToken token);

        public static string llama_token_to_str(LlamaContext ctx, LlamaToken token)
        {
            var ptr = _llama_token_to_str(ctx, token);
            return Marshal.PtrToStringUTF8(ptr) ?? String.Empty;
        }

        [DllImport("llama")]
        public static extern LlamaToken llama_token_bos();

        [DllImport("llama")]
        public static extern LlamaToken llama_token_eos();

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

        [DllImport("llama")]
        public static extern void llama_print_timings(LlamaContext ctx);

        [DllImport("llama")]
        public static extern void llama_reset_timings(LlamaContext ctx);

        [DllImport("llama")]
        public static extern string llama_print_system_info();
    }
}
