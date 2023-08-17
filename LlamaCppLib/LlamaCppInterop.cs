using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    using llama_context = System.IntPtr;
    using llama_model = System.IntPtr;
    using llama_token = System.Int32;
    using llama_grammar = System.IntPtr;

    public static unsafe class LlamaCppInterop
    {
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
            public ulong size;
            [MarshalAs(UnmanagedType.I1)]
            public bool sorted;
        }

        public struct llama_token_data_array
        {
            public Memory<llama_token_data> data;
            public ulong size;
            public bool sorted;
        }

        public delegate void llama_progress_callback(float progress, llama_context ctx);

        public enum llama_log_level
        {
            LLAMA_LOG_LEVEL_ERROR = 2,
            LLAMA_LOG_LEVEL_WARN = 3,
            LLAMA_LOG_LEVEL_INFO = 4
        }

        public delegate void llama_log_callback(llama_log_level level, string text, object user_data);

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public uint seed;
            public int n_ctx;
            public int n_batch;
            public int n_gqa;
            public float rms_norm_eps;
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

        public enum llama_ftype
        {
            LLAMA_FTYPE_ALL_F32 = 0,
            LLAMA_FTYPE_MOSTLY_F16 = 1,             // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q4_0 = 2,            // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q4_1 = 3,            // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,   // tok_embeddings.weight and output.weight are F16
            // LLAMA_FTYPE_MOSTLY_Q4_2 = 5,         // support has been removed
            // LLAMA_FTYPE_MOSTLY_Q4_3 = 6,         // support has been removed
            LLAMA_FTYPE_MOSTLY_Q8_0 = 7,            // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q5_0 = 8,            // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q5_1 = 9,            // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q2_K = 10,           // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q3_K_S = 11,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q3_K_M = 12,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q3_K_L = 13,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q4_K_S = 14,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q4_K_M = 15,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q5_K_S = 16,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q5_K_M = 17,         // except 1d tensors
            LLAMA_FTYPE_MOSTLY_Q6_K = 18,           // except 1d tensors
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_quantize_params
        {
            public int nthread;
            public llama_ftype ftype;
            public bool allow_requantize;
            public bool quantize_output_tensor;
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

        [DllImport(LibName)]
        public static extern void llama_log_set(llama_log_callback log_callback, void* user_data);

        [DllImport(LibName)]
        public static extern int llama_max_devices();

        [DllImport(LibName)]
        public static extern llama_context_params llama_context_default_params();

        [DllImport(LibName)]
        public static extern llama_model_quantize_params llama_model_quantize_default_params();

        [DllImport(LibName)]
        public static extern bool llama_mmap_supported();

        [DllImport(LibName)]
        public static extern bool llama_mlock_supported();

        [DllImport(LibName)]
        public static extern void llama_backend_init(bool numa = false);

        [DllImport(LibName)]
        public static extern void llama_backend_free();

        [DllImport(LibName)]
        public static extern long llama_time_us();

        [DllImport(LibName)]
        public static extern llama_model llama_load_model_from_file(string path_model, llama_context_params cparams);

        [DllImport(LibName)]
        public static extern void llama_free_model(llama_model model);

        [DllImport(LibName)]
        public static extern llama_context llama_new_context_with_model(llama_model model, llama_context_params cparams);

        [DllImport(LibName)]
        public static extern void llama_free(llama_context model);

        [DllImport(LibName)]
        public static extern int llama_model_quantize(string fname_inp, string fname_out, ref llama_model_quantize_params qparams);

        [DllImport(LibName)]
        public static extern int llama_model_apply_lora_from_file(llama_model model, string path_lora, string? path_base_model, int n_threads);

        [DllImport(LibName)]
        public static extern int llama_get_kv_cache_token_count(llama_context ctx);

        [DllImport(LibName)]
        public static extern void llama_set_rng_seed(llama_context ctx, uint seed);

        [DllImport(LibName)]
        public static extern nuint llama_get_state_size(llama_context ctx);

        [DllImport(LibName, EntryPoint = "llama_copy_state_data")]
        private static extern nuint _llama_copy_state_data(llama_context ctx, byte[] dest);

        public static byte[] llama_copy_state_data(llama_context ctx)
        {
            // This is a hack because llama_get_state_size() returns the maximum state size (>2GB for 8192 n_ctx)
            // Hardcoding 1M as this is used exclusively for saving the initial state which is always <1MB
            // WARNING -- Using this method to save a non-intial state will most likely crash (i.e. when kv cache is larger)
            var state = new byte[1024 * 1024];
            var count = (int)_llama_copy_state_data(ctx, state);
            return state.Take(count).ToArray();
        }

        [DllImport(LibName)]
        public static extern nuint llama_set_state_data(llama_context ctx, byte[] src);

        [DllImport(LibName)]
        public static extern bool llama_load_session_file(llama_context ctx, string path_session, llama_token[] tokens_out, int n_token_capacity, out int n_token_count_out);

        [DllImport(LibName)]
        public static extern bool llama_save_session_file(llama_context ctx, string path_session, llama_token[] tokens, int n_token_count);

        [DllImport(LibName)]
        public static extern int llama_eval(llama_context ctx, llama_token[] tokens, int n_tokens, int n_past, int n_threads);

        [DllImport(LibName)]
        public static extern int llama_eval_embd(llama_context ctx, float[] embd, int n_tokens, int n_past, int n_threads);

        [DllImport(LibName)]
        public static extern int llama_eval_export(llama_context ctx, string fname);

        [DllImport(LibName)]
        public static extern int llama_tokenize(llama_context ctx, string text, llama_token[] tokens, int n_max_tokens, bool add_bos);

        [DllImport(LibName)]
        public static extern int llama_tokenize_with_model(llama_model model, string text, llama_token[] tokens, int n_max_tokens, bool add_bos);

        [DllImport(LibName)]
        public static extern int llama_n_vocab(llama_context ctx);

        [DllImport(LibName)]
        public static extern int llama_n_ctx(llama_context ctx);

        [DllImport(LibName)]
        public static extern int llama_n_embd(llama_context ctx);

        [DllImport(LibName)]
        public static extern int llama_n_vocab_from_model(llama_model model);

        [DllImport(LibName)]
        public static extern int llama_n_ctx_from_model(llama_model model);

        [DllImport(LibName)]
        public static extern int llama_n_embd_from_model(llama_model model);

        [DllImport(LibName)]
        public static extern int llama_get_vocab(llama_context ctx, string[] strings, float[] scores, int capacity);

        [DllImport(LibName)]
        public static extern int llama_get_vocab_from_model(llama_model model, string[] strings, float[] scores, int capacity);

        [DllImport(LibName, EntryPoint = "llama_get_logits")]
        private static extern nint _llama_get_logits(llama_context ctx);

        public static List<float> llama_get_logits(llama_context ctx)
        {
            var count = llama_n_vocab(ctx);
            var native_mem = _llama_get_logits(ctx);
            var logits = new float[count];
            Marshal.Copy(native_mem, logits, 0, count);

            return new(logits);
        }

        [DllImport(LibName, EntryPoint = "llama_get_embeddings")]
        private static extern nint _llama_get_embeddings(llama_context ctx);

        public static List<float> llama_get_embeddings(llama_context ctx)
        {
            var count = llama_n_embd(ctx);
            var native_mem = _llama_get_embeddings(ctx);

            if (native_mem == nint.Zero)
                return new();

            var embeddings = new float[count];
            Marshal.Copy(native_mem, embeddings, 0, count);

            return new(embeddings);
        }

        [DllImport(LibName, EntryPoint = "llama_token_to_str")]
        [return: MarshalAs(UnmanagedType.SysUInt)]
        private static extern nint _llama_token_to_str(llama_context ctx, llama_token token);

        public static string llama_token_to_str(llama_context ctx, llama_token token)
        {
            return Marshal.PtrToStringUTF8(_llama_token_to_str(ctx, token)) ?? String.Empty;
        }

        [DllImport(LibName, EntryPoint = "llama_token_to_str_with_model")]
        public static extern nint _llama_token_to_str_with_model(llama_model model, llama_token token);

        public static string llama_token_to_str_with_model(llama_model model, llama_token token)
        {
            return Marshal.PtrToStringUTF8(_llama_token_to_str_with_model(model, token)) ?? String.Empty;
        }

        public static byte[] llama_token_to_bytes(llama_context ctx, llama_token token)
        {
            var ptr = (byte*)_llama_token_to_str(ctx, token).ToPointer();
            var bytes = new List<byte>();
            while (*ptr != '\0') bytes.Add(*ptr++);
            return bytes.ToArray();
        }

        [DllImport(LibName)]
        public static extern llama_token llama_token_bos();

        [DllImport(LibName)]
        public static extern llama_token llama_token_eos();

        [DllImport(LibName)]
        public static extern llama_token llama_token_nl();

        [DllImport(LibName)]
        public static extern llama_grammar llama_grammar_init(llama_grammar_element[] rules, nuint n_rules, nuint start_rule_index);

        [DllImport(LibName)]
        public static extern void llama_grammar_free(llama_grammar grammar);

        [DllImport(LibName, EntryPoint = "llama_sample_repetition_penalty")]
        private static extern void _llama_sample_repetition_penalty(llama_context ctx, nint candidates, llama_token[] last_tokens, int last_tokens_size, float penalty);

        public static void llama_sample_repetition_penalty(llama_context ctx, llama_token_data_array candidates, List<llama_token> last_tokens, float penalty)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_repetition_penalty(ctx, new(&_candidates), last_tokens.ToArray(), last_tokens.Count, penalty);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_frequency_and_presence_penalties")]
        private static extern void _llama_sample_frequency_and_presence_penalties(llama_context ctx, nint candidates, llama_token[] last_tokens, int last_tokens_size, float alpha_frequency, float alpha_presence);

        public static void llama_sample_frequency_and_presence_penalties(llama_context ctx, llama_token_data_array candidates, List<llama_token> last_tokens, float alpha_frequency, float alpha_presence)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_frequency_and_presence_penalties(ctx, new(&_candidates), last_tokens.ToArray(), last_tokens.Count, alpha_frequency, alpha_presence);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_classifier_free_guidance")]
        private static extern void _llama_sample_classifier_free_guidance(llama_context ctx, nint candidates, llama_context guidance_ctx, float scale);

        public static void llama_sample_classifier_free_guidance(llama_context ctx, llama_token_data_array candidates, llama_context guidance_ctx, float scale)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_classifier_free_guidance(ctx, new(&_candidates), guidance_ctx, scale);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_softmax")]
        private static extern void _llama_sample_softmax(llama_context ctx, nint candidates);

        public static void llama_sample_softmax(llama_context ctx, llama_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_softmax(ctx, new(&_candidates));
        }

        [DllImport(LibName, EntryPoint = "llama_sample_top_k")]
        private static extern void _llama_sample_top_k(llama_context ctx, nint candidates, int k, int min_keep = 1);

        public static void llama_sample_top_k(llama_context ctx, llama_token_data_array candidates, int k, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_top_k(ctx, new(&_candidates), k, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_top_p")]
        private static extern void _llama_sample_top_p(llama_context ctx, nint candidates, float p, int min_keep = 1);

        public static void llama_sample_top_p(llama_context ctx, llama_token_data_array candidates, float p, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_top_p(ctx, new(&_candidates), p, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_tail_free")]
        private static extern void _llama_sample_tail_free(llama_context ctx, nint candidates, float z, int min_keep = 1);

        public static void llama_sample_tail_free(llama_context ctx, llama_token_data_array candidates, float z, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_tail_free(ctx, new(&_candidates), z, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_typical")]
        private static extern void _llama_sample_typical(llama_context ctx, nint candidates, float p, int min_keep = 1);

        public static void llama_sample_typical(llama_context ctx, llama_token_data_array candidates, float p, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_typical(ctx, new(&_candidates), p, min_keep);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_temperature")]
        private static extern void _llama_sample_temperature(llama_context ctx, nint candidates, float temp);

        public static void llama_sample_temperature(llama_context ctx, llama_token_data_array candidates, float temp)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_temperature(ctx, new(&_candidates), temp);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_grammar")]
        private static extern void _llama_sample_grammar(llama_context ctx, nint candidates, llama_grammar grammar);

        private static void llama_sample_grammar(llama_context ctx, llama_token_data_array candidates, llama_grammar grammar)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_grammar(ctx, new(&_candidates), grammar);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token_mirostat")]
        private static extern llama_token _llama_sample_token_mirostat(llama_context ctx, nint candidates, float tau, float eta, int m, ref float mu);

        public static llama_token llama_sample_token_mirostat(llama_context ctx, llama_token_data_array candidates, float tau, float eta, int m, ref float mu)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_mirostat(ctx, new(&_candidates), tau, eta, m, ref mu);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token_mirostat_v2")]
        private static extern llama_token _llama_sample_token_mirostat_v2(llama_context ctx, nint candidates, float tau, float eta, ref float mu);

        public static llama_token llama_sample_token_mirostat_v2(llama_context ctx, llama_token_data_array candidates, float tau, float eta, ref float mu)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_mirostat_v2(ctx, new(&_candidates), tau, eta, ref mu);
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token_greedy")]
        private static extern llama_token _llama_sample_token_greedy(llama_context ctx, nint candidates);

        public static llama_token llama_sample_token_greedy(llama_context ctx, llama_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_greedy(ctx, new(&_candidates));
        }

        [DllImport(LibName, EntryPoint = "llama_sample_token")]
        private static extern llama_token _llama_sample_token(llama_context ctx, nint candidates);

        public static llama_token llama_sample_token(llama_context ctx, llama_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new LlamaCppInterop._llama_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token(ctx, new(&_candidates));
        }

        [DllImport(LibName)]
        public static extern void llama_grammar_accept_token(llama_context ctx, llama_grammar grammar, llama_token token);

        [DllImport(LibName)]
        public static extern void llama_print_timings(llama_context ctx);

        [DllImport(LibName)]
        public static extern void llama_reset_timings(llama_context ctx);

        [DllImport(LibName, EntryPoint = "llama_print_system_info")]
        private static extern nint _llama_print_system_info();

        public static string llama_print_system_info()
        {
            return Marshal.PtrToStringUTF8(_llama_print_system_info()) ?? String.Empty;
        }
    }
}
