using System.Runtime.InteropServices;

namespace LlamaCppDotNet
{
    internal static class LlamaCppInterop
    {
        public delegate void llama_progress_callback(float progress, nint ctx);

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public int n_ctx;
            public int n_parts;
            public int seed;

            public bool f16_kv;
            public bool logits_all;
            public bool vocab_only;
            public bool use_mlock;
            public bool embedding;

            public llama_progress_callback progress_callback;
            public nint progress_callback_user_data;
        };

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_context_default_params")]
        public static extern llama_context_params llama_context_default_params();

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_init_from_file")]
        public static extern nint llama_init_from_file(string path_model, llama_context_params ctx_params);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_print_system_info")]
        private static extern nint _llama_print_system_info();

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_print_system_info")]
        private static extern void _llama_print_system_info_free(nint info);

        public static string llama_print_system_info()
        {
            var str = LlamaCppInterop._llama_print_system_info();
            var info_str = Marshal.PtrToStringAnsi(str) ?? String.Empty;
            LlamaCppInterop._llama_print_system_info_free(str);
            return info_str;
        }

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_tokenize")]
        private static extern void _llama_tokenize(nint ctx, string text, bool add_bos, out nint r_tokens, out int r_tokens_len);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_tokenize_free")]
        private static extern void _llama_tokenize_free(nint tokens);

        public static int[] llama_tokenize(nint ctx, string text, bool add_bos)
        {
            LlamaCppInterop._llama_tokenize(ctx, text, false, out var r_tokens, out var r_tokens_len);
            var tokens = new int[r_tokens_len];
            Marshal.Copy(r_tokens, tokens, 0, r_tokens_len);
            LlamaCppInterop._llama_tokenize_free(r_tokens);
            return tokens;
        }

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_n_ctx")]
        public static extern int llama_n_ctx(nint ctx);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_eval")]
        public static extern int llama_eval(nint ctx, int[] tokens, int n_tokens, int n_past, int n_threads);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_sample_top_p_top_k")]
        public static extern int llama_sample_top_p_top_k(nint ctx, int[] last_n_tokens_data, int last_n_tokens_size, int top_k, float top_p, float temp, float repeat_penalty);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_token_to_str")]
        private static extern nint _llama_token_to_str(nint ctx, int token);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_token_to_str_free")]
        private static extern void _llama_token_to_str_free(nint token_str);

        public static string llama_token_to_str(nint ctx, int token)
        {
            var str = _llama_token_to_str(ctx, token);
            var token_str = Marshal.PtrToStringAnsi(str) ?? String.Empty;
            _llama_tokenize_free(str);
            return token_str;
        }

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_print_timings")]
        public static extern void llama_print_timings(nint ctx);

        [DllImport("llama.cpp-interop", EntryPoint = "_llama_free")]
        public static extern void llama_free(nint ctx);
    }
}
