using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCppLib
{
    using llama_model = System.IntPtr;
    using llama_context = System.IntPtr;
    using llama_token = System.Int32;
    using llama_grammar = System.IntPtr;

    public static class LlamaCppInterop
    {
        public enum llama_log_level
        {
            LLAMA_LOG_LEVEL_ERROR = 2,
            LLAMA_LOG_LEVEL_WARN = 3,
            LLAMA_LOG_LEVEL_INFO = 4
        }

        public enum llama_vocab_type_
        {
            LLAMA_VOCAB_TYPE_SPM = 0,
            LLAMA_VOCAB_TYPE_BPE = 1,
        }

        public enum llama_token_type
        {
            LLAMA_TOKEN_TYPE_UNDEFINED = 0,
            LLAMA_TOKEN_TYPE_NORMAL = 1,
            LLAMA_TOKEN_TYPE_UNKNOWN = 2,
            LLAMA_TOKEN_TYPE_CONTROL = 3,
            LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
            LLAMA_TOKEN_TYPE_UNUSED = 5,
            LLAMA_TOKEN_TYPE_BYTE = 6,
        }

        public enum llama_ftype
        {
            LLAMA_FTYPE_ALL_F32 = 0,
            LLAMA_FTYPE_MOSTLY_F16 = 1,
            LLAMA_FTYPE_MOSTLY_Q4_0 = 2,
            LLAMA_FTYPE_MOSTLY_Q4_1 = 3,
            LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,
            // LLAMA_FTYPE_MOSTLY_Q4_2 = 5,         
            // LLAMA_FTYPE_MOSTLY_Q4_3 = 6,         
            LLAMA_FTYPE_MOSTLY_Q8_0 = 7,
            LLAMA_FTYPE_MOSTLY_Q5_0 = 8,
            LLAMA_FTYPE_MOSTLY_Q5_1 = 9,
            LLAMA_FTYPE_MOSTLY_Q2_K = 10,
            LLAMA_FTYPE_MOSTLY_Q3_K_S = 11,
            LLAMA_FTYPE_MOSTLY_Q3_K_M = 12,
            LLAMA_FTYPE_MOSTLY_Q3_K_L = 13,
            LLAMA_FTYPE_MOSTLY_Q4_K_S = 14,
            LLAMA_FTYPE_MOSTLY_Q4_K_M = 15,
            LLAMA_FTYPE_MOSTLY_Q5_K_S = 16,
            LLAMA_FTYPE_MOSTLY_Q5_K_M = 17,
            LLAMA_FTYPE_MOSTLY_Q6_K = 18,
            LLAMA_FTYPE_GUESSED = 1024,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_token_data
        {
            public llama_token id;
            public float logit;
            public float p;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct _llama_token_data_array
        {
            public nint data;
            public nuint size;
            [MarshalAs(UnmanagedType.I1)] public bool sorted;
        }

        public struct llama_token_data_array
        {
            public Memory<llama_token_data> data;
            public nuint size;
            [MarshalAs(UnmanagedType.I1)] public bool sorted;
        }

        public delegate void llama_progress_callback(float progress, llama_context ctx);

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public uint seed;
            public int n_ctx;
            public int n_batch;
            public int n_gpu_layers;
            public int main_gpu;

            public nint tensor_split;

            public float rope_freq_base;
            public float rope_freq_scale;

            public llama_progress_callback progress_callback;
            public nint progress_callback_user_data;

            [MarshalAs(UnmanagedType.I1)] public bool low_vram;
            [MarshalAs(UnmanagedType.I1)] public bool mul_mat_q;
            [MarshalAs(UnmanagedType.I1)] public bool f16_kv;
            [MarshalAs(UnmanagedType.I1)] public bool logits_all;
            [MarshalAs(UnmanagedType.I1)] public bool vocab_only;
            [MarshalAs(UnmanagedType.I1)] public bool use_mmap;
            [MarshalAs(UnmanagedType.I1)] public bool use_mlock;
            [MarshalAs(UnmanagedType.I1)] public bool embedding;
        }

        public delegate void llama_log_callback(llama_log_level level, string text, object user_data);

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_quantize_params
        {
            public int nthread;
            public llama_ftype ftype;
            [MarshalAs(UnmanagedType.I1)] public bool allow_requantize;
            [MarshalAs(UnmanagedType.I1)] public bool quantize_output_tensor;
        }

        public enum llama_gretype
        {
            LLAMA_GRETYPE_END = 0,
            LLAMA_GRETYPE_ALT = 1,
            LLAMA_GRETYPE_RULE_REF = 2,
            LLAMA_GRETYPE_CHAR = 3,
            LLAMA_GRETYPE_CHAR_NOT = 4,
            LLAMA_GRETYPE_CHAR_RNG_UPPER = 5,
            LLAMA_GRETYPE_CHAR_ALT = 6,
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_grammar_element
        {
            public llama_gretype type;
            public uint value;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_timings
        {
            public double t_start_ms;
            public double t_end_ms;
            public double t_load_ms;
            public double t_sample_ms;
            public double t_p_eval_ms;
            public double t_eval_ms;
            public int n_sample;
            public int n_p_eval;
            public int n_eval;
        }

#if WINDOWS
        private const string LibName = $"{nameof(LlamaCppLib)}/llama";
#elif LINUX
        private const string LibName = $"{nameof(LlamaCppLib)}/libllama";
#endif

        [DllImport(LibName)] public static extern llama_context_params llama_context_default_params();
        [DllImport(LibName)] public static extern llama_model_quantize_params llama_model_quantize_default_params();

        [DllImport(LibName)] public static extern void llama_backend_init(bool numa = false);
        [DllImport(LibName)] public static extern void llama_backend_free();

        [DllImport(LibName)] public static extern llama_model llama_load_model_from_file(string path_model, llama_context_params cparams);
        [DllImport(LibName)] public static extern void llama_free_model(llama_model model);
        [DllImport(LibName)] public static extern llama_context llama_new_context_with_model(llama_model model, llama_context_params cparams);
        [DllImport(LibName)] public static extern void llama_free(llama_context model);

        [DllImport(LibName)] public static extern long llama_time_us();

        [DllImport(LibName)] public static extern int llama_max_devices();
        [DllImport(LibName)][return: MarshalAs(UnmanagedType.I1)] public static extern bool llama_mmap_supported();
        [DllImport(LibName)][return: MarshalAs(UnmanagedType.I1)] public static extern bool llama_mlock_supported();

        [DllImport(LibName)] public static extern int llama_n_vocab(llama_context ctx);
        [DllImport(LibName)] public static extern int llama_n_ctx(llama_context ctx);
        [DllImport(LibName)] public static extern int llama_n_embd(llama_context ctx);

        [DllImport(LibName, EntryPoint = "llama_vocab_type")] public static extern llama_vocab_type_ llama_vocab_type(llama_context ctx);

        [DllImport(LibName)] public static extern int llama_model_n_vocab(llama_model model);
        [DllImport(LibName)] public static extern int llama_model_n_ctx(llama_model model);
        [DllImport(LibName)] public static extern int llama_model_n_embd(llama_model model);

        [DllImport(LibName)] public static extern int llama_model_type(llama_model model, char[] buf, nuint buf_size);

        [DllImport(LibName)] public static extern int llama_model_quantize(string fname_inp, string fname_out, ref llama_model_quantize_params qparams);

        [DllImport(LibName)] public static extern int llama_model_apply_lora_from_file(llama_model model, string path_lora, string? path_base_model, int n_threads);

        [DllImport(LibName)] public static extern int llama_get_kv_cache_token_count(llama_context ctx);

        [DllImport(LibName)] public static extern void llama_set_rng_seed(llama_context ctx, uint seed);

        [DllImport(LibName)] public static extern nuint llama_get_state_size(llama_context ctx);

        [DllImport(LibName, EntryPoint = "llama_copy_state_data")] private static extern nuint _llama_copy_state_data(llama_context ctx, byte[] dest);
        public static byte[] llama_copy_state_data(llama_context ctx)
        {
            // This is a hack because llama_get_state_size() returns the maximum state size (>2GB for 8192 n_ctx)
            // Hardcoding 1MiB as this is used exclusively for saving the initial state which is always < 1MiB
            // WARNING -- Using this method to save a non-intial state will most likely crash (i.e. when kv cache is larger than 1 MiB)
            var state = new byte[1024 * 1024];
            var count = (int)_llama_copy_state_data(ctx, state);
            return state.Take(count).ToArray();
        }
        [DllImport(LibName)] public static extern nuint llama_set_state_data(llama_context ctx, byte[] src);

        [DllImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)] public static extern bool llama_load_session_file(llama_context ctx, string path_session, llama_token[] tokens_out, int n_token_capacity, out int n_token_count_out);
        [DllImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)] public static extern bool llama_save_session_file(llama_context ctx, string path_session, llama_token[] tokens, int n_token_count);

        [DllImport(LibName)] public static extern int llama_eval(llama_context ctx, llama_token[] tokens, int n_tokens, int n_past, int n_threads);
        [DllImport(LibName)] public static extern int llama_eval_embd(llama_context ctx, float[] embd, int n_tokens, int n_past, int n_threads);
        [DllImport(LibName)] public static extern int llama_eval_export(llama_context ctx, string fname);

        [DllImport(LibName, EntryPoint = "llama_get_logits")] private static extern nint _llama_get_logits(llama_context ctx);
        public static unsafe Span<float> llama_get_logits(llama_context ctx)
        {
            return new(_llama_get_logits(ctx).ToPointer(), llama_n_vocab(ctx));
        }

        [DllImport(LibName, EntryPoint = "llama_get_embeddings")] private static extern nint _llama_get_embeddings(llama_context ctx);
        public static float[] llama_get_embeddings(llama_context ctx)
        {
            var count = llama_n_embd(ctx);
            var native_mem = _llama_get_embeddings(ctx);

            if (native_mem == nint.Zero)
                return new float[0];

            var embeddings = new float[count];
            Marshal.Copy(native_mem, embeddings, 0, count);

            return embeddings;
        }

        //
        // Vocab
        //

        [DllImport(LibName, EntryPoint = "llama_token_get_text")] private static extern nint _llama_token_get_text(llama_context ctx, llama_token token);
        public static string llama_token_get_text(llama_context ctx, llama_token token)
        {
            return Marshal.PtrToStringAnsi(_llama_token_get_text(ctx, token)) ?? String.Empty;
        }

        [DllImport(LibName)] private static extern float llama_token_get_score(llama_context ctx, llama_token token);
        [DllImport(LibName)] private static extern llama_token_type llama_token_get_type(llama_context ctx, llama_token token);

        //
        // Special tokens
        //

        [DllImport(LibName)] public static extern llama_token llama_token_bos(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_eos(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_nl(llama_context ctx);

        //
        // Tokenization
        //

        [DllImport(LibName, EntryPoint = "llama_tokenize")] private static extern int _llama_tokenize(llama_context ctx, string text, llama_token[] tokens, int n_max_tokens, bool add_bos);
        public static void llama_tokenize(llama_context ctx, string text, out ReadOnlySpan<llama_token> tokens, bool add_bos)
        {
            var n_tokens = text.Length + (add_bos ? 1 : 0);
            var result = new llama_token[n_tokens];
            n_tokens = _llama_tokenize(ctx, text, result, result.Length, add_bos);
            if (n_tokens < 0)
            {
                result = new llama_token[-n_tokens];
                var check = _llama_tokenize(ctx, text, result, result.Length, add_bos);
                Debug.Assert(check == -n_tokens);
                n_tokens = result.Length;
            }

            tokens = result.AsSpan(0, n_tokens);
        }

        [DllImport(LibName)] public static extern int llama_tokenize_with_model(llama_model model, string text, llama_token[] tokens, int n_max_tokens, bool add_bos);

        [DllImport(LibName, EntryPoint = "llama_token_to_str")] private static extern int _llama_token_to_str(llama_context ctx, llama_token token, byte[] buf, int length);
        public static string llama_token_to_str(llama_context ctx, llama_token token)
        {
            var result = new byte[8];
            var n_tokens = _llama_token_to_str(ctx, token, result, result.Length);
            if (n_tokens < 0)
            {
                result = new byte[-n_tokens];
                var check = _llama_token_to_str(ctx, token, result, result.Length);
                Debug.Assert(check == -n_tokens);
                n_tokens = result.Length;
            }

            return Encoding.ASCII.GetString(result, 0, n_tokens);
        }

        [DllImport(LibName)] public static extern nint llama_token_to_str_with_model(llama_model model, llama_token token, byte[] buf, int length);

        //
        // Grammar
        //

        [DllImport(LibName)] public static extern llama_grammar llama_grammar_init(llama_grammar_element[] rules, nuint n_rules, nuint start_rule_index);
        [DllImport(LibName)] public static extern void llama_grammar_free(llama_grammar grammar);

        //
        // Sampling functions
        //

        [DllImport(LibName, EntryPoint = "llama_sample_repetition_penalty")] private static extern void _llama_sample_repetition_penalty(llama_context ctx, nint candidates, llama_token[] last_tokens, int last_tokens_size, float penalty);
        public static unsafe void llama_sample_repetition_penalty(llama_context ctx, llama_token_data_array candidates, llama_token[] last_tokens, float penalty)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_repetition_penalty(ctx, new(&_candidates), last_tokens, last_tokens.Length, penalty);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_frequency_and_presence_penalties")]
        private static extern void _llama_sample_frequency_and_presence_penalties(llama_context ctx, nint candidates, llama_token[] last_tokens, int last_tokens_size, float alpha_frequency, float alpha_presence);
        public static unsafe void llama_sample_frequency_and_presence_penalties(llama_context ctx, llama_token_data_array candidates, llama_token[] last_tokens, float alpha_frequency, float alpha_presence)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_frequency_and_presence_penalties(ctx, new(&_candidates), last_tokens, last_tokens.Length, alpha_frequency, alpha_presence);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_classifier_free_guidance")]
        private static extern void _llama_sample_classifier_free_guidance(llama_context ctx, nint candidates, llama_context guidance_ctx, float scale);
        public static unsafe void llama_sample_classifier_free_guidance(llama_context ctx, llama_token_data_array candidates, llama_context guidance_ctx, float scale)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_classifier_free_guidance(ctx, new(&_candidates), guidance_ctx, scale);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_softmax")] private static extern void _llama_sample_softmax(llama_context ctx, nint candidates);
        public static unsafe void llama_sample_softmax(llama_context ctx, llama_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_softmax(ctx, new(&_candidates));
        }

        [DllImport(LibName, EntryPoint = "llama_sample_top_k")] private static extern void _llama_sample_top_k(llama_context ctx, nint candidates, int k, int min_keep = 1);
        public static unsafe void llama_sample_top_k(llama_context ctx, llama_token_data_array candidates, int k, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_top_k(ctx, new(&_candidates), k, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_top_p")] private static extern void _llama_sample_top_p(llama_context ctx, nint candidates, float p, int min_keep = 1);
        public static unsafe void llama_sample_top_p(llama_context ctx, llama_token_data_array candidates, float p, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_top_p(ctx, new(&_candidates), p, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_tail_free")] private static extern void _llama_sample_tail_free(llama_context ctx, nint candidates, float z, int min_keep = 1);
        public static unsafe void llama_sample_tail_free(llama_context ctx, llama_token_data_array candidates, float z, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_tail_free(ctx, new(&_candidates), z, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_typical")] private static extern void _llama_sample_typical(llama_context ctx, nint candidates, float p, int min_keep = 1);
        public static unsafe void llama_sample_typical(llama_context ctx, llama_token_data_array candidates, float p, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_typical(ctx, new(&_candidates), p, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_temperature")] private static extern void _llama_sample_temperature(llama_context ctx, nint candidates, float temp);
        public static unsafe void llama_sample_temperature(llama_context ctx, llama_token_data_array candidates, float temp)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_temperature(ctx, new(&_candidates), temp);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_grammar")] private static extern void _llama_sample_grammar(llama_context ctx, nint candidates, llama_grammar grammar);
        private static unsafe void llama_sample_grammar(llama_context ctx, llama_token_data_array candidates, llama_grammar grammar)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_grammar(ctx, new(&_candidates), grammar);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token_mirostat")]
        private static extern llama_token _llama_sample_token_mirostat(llama_context ctx, nint candidates, float tau, float eta, int m, ref float mu);
        public static unsafe llama_token llama_sample_token_mirostat(llama_context ctx, llama_token_data_array candidates, float tau, float eta, int m, ref float mu)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            return _llama_sample_token_mirostat(ctx, new(&_candidates), tau, eta, m, ref mu);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token_mirostat_v2")]
        private static extern llama_token _llama_sample_token_mirostat_v2(llama_context ctx, nint candidates, float tau, float eta, ref float mu);
        public static unsafe llama_token llama_sample_token_mirostat_v2(llama_context ctx, llama_token_data_array candidates, float tau, float eta, ref float mu)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            return _llama_sample_token_mirostat_v2(ctx, new(&_candidates), tau, eta, ref mu);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token_greedy")] private static extern llama_token _llama_sample_token_greedy(llama_context ctx, nint candidates);
        public static unsafe llama_token llama_sample_token_greedy(llama_context ctx, llama_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            return _llama_sample_token_greedy(ctx, new(&_candidates));
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token")] private static extern llama_token _llama_sample_token(llama_context ctx, nint candidates);
        public static unsafe llama_token llama_sample_token(llama_context ctx, llama_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            return _llama_sample_token(ctx, new(&_candidates));
        }

        [DllImport(LibName)] public static extern void llama_grammar_accept_token(llama_context ctx, llama_grammar grammar, llama_token token);

        //
        // Performance information
        //

        [DllImport(LibName)] public static extern llama_timings llama_get_timings(llama_context ctx);
        [DllImport(LibName)] public static extern void llama_print_timings(llama_context ctx);
        [DllImport(LibName)] public static extern void llama_reset_timings(llama_context ctx);

        [DllImport(LibName, EntryPoint = "llama_print_system_info")] private static extern nint _llama_print_system_info();
        public static string llama_print_system_info()
        {
            return Marshal.PtrToStringAnsi(_llama_print_system_info()) ?? String.Empty;
        }

        [DllImport(LibName)] public static extern void llama_log_set(llama_log_callback log_callback, object user_data);

        public static byte[] llama_token_to_bytes(llama_context ctx, llama_token token)
        {
            var result = new byte[8];
            var n_tokens = _llama_token_to_str(ctx, token, result, result.Length);
            if (n_tokens >= 0)
            {
                var bytes = new byte[n_tokens];
                Array.Copy(result, bytes, bytes.Length);
                return bytes;
            }

            result = new byte[-n_tokens];
            var check = _llama_token_to_str(ctx, token, result, result.Length);
            Debug.Assert(check == -n_tokens);
            return result;
        }
    }
}
