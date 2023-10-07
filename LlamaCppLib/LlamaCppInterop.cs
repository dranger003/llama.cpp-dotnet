using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCppLib
{
    // ggml.h

    using ggml_tensor = System.IntPtr;

    // llama.h

    using llama_model = System.IntPtr;
    using llama_context = System.IntPtr;

    using llama_pos = System.Int32;
    using llama_token = System.Int32;
    using llama_seq_id = System.Int32;

    using llama_grammar = System.IntPtr;

    public static class LlamaCppInterop
    {
        // ggml.h

        public enum ggml_log_level
        {
            GGML_LOG_LEVEL_ERROR = 2,
            GGML_LOG_LEVEL_WARN = 3,
            GGML_LOG_LEVEL_INFO = 4
        };

        public delegate void ggml_log_callback(ggml_log_level level, string text, object user_data);

        // llama.h

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
            //LLAMA_FTYPE_MOSTLY_Q4_2 = 5,
            //LLAMA_FTYPE_MOSTLY_Q4_3 = 6,
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
        public unsafe struct llama_batch
        {
            public int n_tokens;

            private llama_token* _token;
            private float* _embd;
            private llama_pos* _pos;
            private llama_seq_id* _seq_id;
            private sbyte* _logits;

            public llama_pos all_pos_0;
            public llama_pos all_pos_1;
            public llama_seq_id all_seq_id;

            public Span<llama_token> token(int n_tokens) { return new(_token, n_tokens); }
            public Span<float> embd(int n_tokens, int embd) { return new(_embd, n_tokens * embd); }
            public Span<llama_pos> pos(int n_tokens) { return new(_pos, n_tokens); }
            public Span<llama_seq_id> seq_id(int n_tokens) { return new(_seq_id, n_tokens); }
            public Span<sbyte> logits(int n_tokens) { return new(_logits, n_tokens); }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_params
        {
            public int n_gpu_layers;
            public int main_gpu;
            public readonly float[] tensor_split;

            public llama_progress_callback progress_callback;
            public object progress_callback_user_data;

            [MarshalAs(UnmanagedType.I1)] public bool vocab_only;
            [MarshalAs(UnmanagedType.I1)] public bool use_mmap;
            [MarshalAs(UnmanagedType.I1)] public bool use_mlock;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public uint seed;
            public uint n_ctx;
            public uint n_batch;
            public uint n_threads;
            public uint n_threads_batch;

            public float rope_freq_base;
            public float rope_freq_scale;

            [MarshalAs(UnmanagedType.I1)] public bool mul_mat_q;
            [MarshalAs(UnmanagedType.I1)] public bool f16_kv;
            [MarshalAs(UnmanagedType.I1)] public bool logits_all;
            [MarshalAs(UnmanagedType.I1)] public bool embedding;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_quantize_params
        {
            public int nthread;
            public llama_ftype ftype;
            [MarshalAs(UnmanagedType.I1)] public bool allow_requantize;
            [MarshalAs(UnmanagedType.I1)] public bool quantize_output_tensor;
            [MarshalAs(UnmanagedType.I1)] public bool only_copy;
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
#elif LINUX || MACOS
        private const string LibName = $"{nameof(LlamaCppLib)}/libllama";
#endif

        // Helpers for getting default parameters
        [DllImport(LibName)] public static extern llama_model_params llama_model_default_params();
        [DllImport(LibName)] public static extern llama_context_params llama_context_default_params();
        [DllImport(LibName)] public static extern llama_model_quantize_params llama_model_quantize_default_params();

        // Initialize the llama + ggml backend
        [DllImport(LibName)] public static extern void llama_backend_init(bool numa = false);
        [DllImport(LibName)] public static extern void llama_backend_free();

        [DllImport(LibName)] public static extern llama_model llama_load_model_from_file(string path_model, llama_model_params cparams);
        [DllImport(LibName)] public static extern void llama_free_model(llama_model model);

        [DllImport(LibName)] public static extern llama_context llama_new_context_with_model(llama_model model, llama_context_params cparams);
        [DllImport(LibName)] public static extern void llama_free(llama_context ctx);

        [DllImport(LibName)] public static extern long llama_time_us();

        [DllImport(LibName)] public static extern int llama_max_devices();
        [DllImport(LibName)][return: MarshalAs(UnmanagedType.I1)] public static extern bool llama_mmap_supported();
        [DllImport(LibName)][return: MarshalAs(UnmanagedType.I1)] public static extern bool llama_mlock_supported();

        [DllImport(LibName)] public static extern llama_model llama_get_model(llama_context ctx);

        [DllImport(LibName)] public static extern int llama_n_ctx(llama_context ctx);

        [DllImport(LibName)] public static extern llama_vocab_type_ llama_vocab_type(llama_model model);

        [DllImport(LibName)] public static extern int llama_n_vocab(llama_model model);
        [DllImport(LibName)] public static extern int llama_n_ctx_train(llama_model model);
        [DllImport(LibName)] public static extern int llama_n_embd(llama_model model);

        // Get the model's RoPE frequency scaling factor
        [DllImport(LibName)] public static extern float llama_rope_freq_scale_train(llama_model model);

        // Get a string describing the model type
        [DllImport(LibName, EntryPoint = "llama_model_desc")] private static extern int _llama_model_desc(llama_model model, StringBuilder buf, nuint buf_size);
        public static int llama_model_desc(llama_model model, out string buf, int capacity = 0x4000) // 16KiB
        {
            var _buf = new StringBuilder(capacity);
            var result = _llama_model_desc(model, _buf, (nuint)_buf.Capacity);
            buf = _buf.ToString();
            return result;
        }

        // Returns the total size of all the tensors in the model in bytes
        [DllImport(LibName)] public static extern ulong llama_model_size(llama_model model);

        // Returns the total number of parameters in the model
        [DllImport(LibName)] public static extern ulong llama_model_n_params(llama_model model);

        // Get a llama model tensor
        [DllImport(LibName)] public static extern ggml_tensor llama_get_model_tensor(llama_model model, string name);

        // Returns 0 on success
        [DllImport(LibName)] public static extern int llama_model_quantize(string fname_inp, string fname_out, ref llama_model_quantize_params qparams);

        // Apply a LoRA adapter to a loaded model
        [DllImport(LibName)] public static extern int llama_model_apply_lora_from_file(llama_model model, string path_lora, float scale, string? path_base_model, int n_threads);

        //
        // KV cache
        //

        [DllImport(LibName)] public static extern void llama_kv_cache_tokens_rm(llama_context ctx, int c0, int c1);
        [DllImport(LibName)] public static extern void llama_kv_cache_seq_rm(llama_context ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
        [DllImport(LibName)] public static extern void llama_kv_cache_seq_cp(llama_context ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
        [DllImport(LibName)] public static extern void llama_kv_cache_seq_keep(llama_context ctx, llama_seq_id seq_id);
        [DllImport(LibName)] public static extern void llama_kv_cache_seq_shift(llama_context ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);

        //
        // State / sessions
        //

        // Returns the maximum size in bytes of the state (rng, logits, embedding
        // and kv_cache) - will often be smaller after compacting tokens
        [DllImport(LibName)] public static extern nuint llama_get_state_size(llama_context ctx);

        [DllImport(LibName, EntryPoint = "llama_copy_state_data")] private static extern nuint _llama_copy_state_data(llama_context ctx, byte[] dest);
        public static byte[] llama_copy_state_data(llama_context ctx)
        {
            // This is a hack because llama_get_state_size() returns the maximum state size (>2GB for >8192 n_ctx)
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

        //
        // Decoding
        //

        // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
        [DllImport(LibName)] private static extern llama_batch llama_batch_get_one(llama_token[] tokens, int n_tokens, llama_pos pos_0, llama_seq_id seq_id);

        [DllImport(LibName)] public static extern llama_batch llama_batch_init(int n_tokens, int embd);
        [DllImport(LibName)] public static extern void llama_batch_free(llama_batch batch);

        [DllImport(LibName)] public static extern int llama_decode(llama_context ctx, llama_batch batch);

        [DllImport(LibName)] public static extern void llama_set_n_threads(llama_context ctx, uint n_threads, uint n_threads_batch);

        [DllImport(LibName, EntryPoint = "llama_get_logits")] private static extern nint _llama_get_logits(llama_context ctx);
        public static unsafe Span<float> llama_get_logits(llama_context ctx)
        {
            return new(_llama_get_logits(ctx).ToPointer(), llama_n_vocab(llama_get_model(ctx)) * llama_n_ctx(ctx));
        }

        [DllImport(LibName, EntryPoint = "llama_get_logits_ith")] private static extern nint _llama_get_logits_ith(llama_context ctx, int i);
        public static unsafe Span<float> llama_get_logits_ith(llama_context ctx, int i)
        {
            return new(_llama_get_logits_ith(ctx, i).ToPointer(), llama_n_vocab(llama_get_model(ctx)) * llama_n_ctx(ctx));
        }

        [DllImport(LibName, EntryPoint = "llama_get_embeddings")] private static extern nint _llama_get_embeddings(llama_context ctx);
        public static unsafe Span<float> llama_get_embeddings(llama_context ctx)
        {
            return new(_llama_get_embeddings(ctx).ToPointer(), llama_n_embd(llama_get_model(ctx)));
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

        // Special tokens
        [DllImport(LibName)] public static extern llama_token llama_token_bos(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_eos(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_nl(llama_context ctx);

        // codellama infill tokens
        [DllImport(LibName)] public static extern llama_token llama_token_prefix(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_middle(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_suffix(llama_context ctx);
        [DllImport(LibName)] public static extern llama_token llama_token_eot(llama_context ctx);

        //
        // Tokenization
        //

        [DllImport(LibName, EntryPoint = "llama_tokenize")] private static extern int _llama_tokenize(llama_model model, string text, int text_len, llama_token[] tokens, int n_max_tokens, bool add_bos);
        public static ReadOnlySpan<llama_token> llama_tokenize(llama_context ctx, string text, bool add_bos = false)
        {
            return llama_tokenize_with_model(llama_get_model(ctx), text, add_bos);
        }
        public static ReadOnlySpan<llama_token> llama_tokenize_with_model(llama_model model, string text, bool add_bos)
        {
            var n_tokens = text.Length + (add_bos ? 1 : 0);
            var result = new llama_token[n_tokens];
            n_tokens = _llama_tokenize(model, text, text.Length, result, result.Length, add_bos);
            if (n_tokens < 0)
            {
                result = new llama_token[-n_tokens];
                var check = _llama_tokenize(model, text, text.Length, result, result.Length, add_bos);
                Debug.Assert(check == -n_tokens);
                n_tokens = result.Length;
            }

            return result.AsSpan(0, n_tokens);
        }

        [DllImport(LibName, EntryPoint = "llama_token_to_piece")] private static extern int _llama_token_to_piece(llama_model model, llama_token token, byte[] buf, int length);
        public static ReadOnlySpan<byte> llama_token_to_piece(llama_context ctx, llama_token token)
        {
            var result = new byte[8];
            var n_pieces = _llama_token_to_piece(llama_get_model(ctx), token, result, result.Length);
            if (n_pieces < 0)
            {
                result = new byte[-n_pieces];
                var check = _llama_token_to_piece(llama_get_model(ctx), token, result, result.Length);
                Debug.Assert(check == -n_pieces);
                n_pieces = result.Length;
            }

            return result.AsSpan(0, n_pieces);
        }

        //
        // Grammar
        //

        [DllImport(LibName)] public static extern llama_grammar llama_grammar_init(llama_grammar_element[] rules, nuint n_rules, nuint start_rule_index);
        [DllImport(LibName)] public static extern void llama_grammar_free(llama_grammar grammar);
        [DllImport(LibName)] public static extern llama_grammar llama_grammar_copy(llama_grammar grammar);

        //
        // Sampling functions
        //

        [DllImport(LibName)] public static extern void llama_set_rng_seed(llama_context ctx, uint seed);

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

        [DllImport(LibName, EntryPoint = "llama_sample_temp")] private static extern void _llama_sample_temp(llama_context ctx, nint candidates, float temp);
        public static unsafe void llama_sample_temp(llama_context ctx, llama_token_data_array candidates, float temp)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new _llama_token_data_array { data = new(handle.Pointer), size = candidates.size, sorted = candidates.sorted };
            _llama_sample_temp(ctx, new(&_candidates), temp);
        }

        // DEPRECATED: void llama_sample_temperature(llama_context ctx, llama_token_data_array[] candidates, float temp);

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
        // Beam search
        //

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_beam_view
        {
            public llama_token[] tokens;
            public nuint n_tokens;
            public float p;
            [MarshalAs(UnmanagedType.I1)] public bool eob;
        };

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_beams_state
        {
            public llama_beam_view[] beam_views;
            public nuint n_beams;
            public nuint common_prefix_length;
            [MarshalAs(UnmanagedType.I1)] public bool last_call;
        };

        public delegate void llama_beam_search_callback_fn_t(object callback_data, llama_beams_state state);

        [DllImport(LibName)]
        public static extern void llama_beam_search(llama_context ctx, llama_beam_search_callback_fn_t callback, object callback_data, nuint n_beams, int n_past, int n_predict);

        // Performance information
        [DllImport(LibName)] public static extern llama_timings llama_get_timings(llama_context ctx);

        [DllImport(LibName)] public static extern void llama_print_timings(llama_context ctx);
        [DllImport(LibName)] public static extern void llama_reset_timings(llama_context ctx);

        // Print system information
        [DllImport(LibName, EntryPoint = "llama_print_system_info")] private static extern nint _llama_print_system_info();
        public static string llama_print_system_info()
        {
            return Marshal.PtrToStringAnsi(_llama_print_system_info()) ?? String.Empty;
        }

        [DllImport(LibName)] public static extern void llama_log_set(ggml_log_callback log_callback, object user_data);

        [DllImport(LibName)] public static extern void llama_dump_timing_info_yaml(nint stream, llama_context ctx);

        //
        // Other
        //

        //public static byte[] llama_token_to_bytes(llama_context ctx, llama_token token)
        //{
        //    var result = new byte[8];
        //    var n_tokens = _llama_token_to_piece(ctx, token, result, result.Length);
        //    if (n_tokens >= 0)
        //    {
        //        var bytes = new byte[n_tokens];
        //        Array.Copy(result, bytes, bytes.Length);
        //        return bytes;
        //    }

        //    result = new byte[-n_tokens];
        //    var check = _llama_token_to_piece(ctx, token, result, result.Length);
        //    Debug.Assert(check == -n_tokens);
        //    return result;
        //}
    }
}
