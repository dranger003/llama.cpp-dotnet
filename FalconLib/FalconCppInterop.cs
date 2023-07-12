using System.Runtime.InteropServices;

namespace FalconCppLib
{
    using falcon_context = System.IntPtr;
    using falcon_model = System.IntPtr;
    using falcon_token = System.Int32;

    public static unsafe class FalconCppInterop
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct falcon_token_data
        {
            public falcon_token id;
            public float logit;
            public float p;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct _falcon_token_data_array
        {
            public nint data;
            public ulong size;
            [MarshalAs(UnmanagedType.I1)]
            public bool sorted;
        }

        public struct falcon_token_data_array
        {
            public Memory<falcon_token_data> data;
            public ulong size;
            public bool sorted;
        }

        public delegate void falcon_progress_callback(float progress, falcon_context ctx, nint status);

        [StructLayout(LayoutKind.Sequential)]
        public struct falcon_context_params
        {
            public int n_ctx;
            public int n_batch;
            public int n_gpu_layers;
            public int i_gpu_start;
            public int i_gpu_last;
            public int main_gpu;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
            public float[] tensor_split;
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

            public falcon_progress_callback progress_callback;
            public nint progress_callback_user_data;
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
            [MarshalAs(UnmanagedType.I1)]
            public bool allow_requantize;
            [MarshalAs(UnmanagedType.I1)]
            public bool quantize_output_tensor;
        }

        [DllImport("falcon")]
        public static extern falcon_context_params falcon_context_default_params();
        //LLAMA_API struct falcon_context_params falcon_context_default_params();

        [DllImport("falcon")]
        public static extern llama_model_quantize_params llama_model_quantize_default_params();
        //LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params();

        [DllImport("falcon")]
        public static extern bool llama_mmap_supported();
        //LLAMA_API bool llama_mmap_supported();

        [DllImport("falcon")]
        public static extern bool llama_mlock_supported();
        //LLAMA_API bool llama_mlock_supported();

        [DllImport("falcon")]
        public static extern void falcon_init_backend();
        //LLAMA_API void falcon_init_backend();

        [DllImport("falcon")]
        public static extern long llama_time_us();
        //LLAMA_API int64_t llama_time_us();

        [DllImport("falcon")]
        public static extern falcon_context falcon_init_from_file(string path_model, falcon_context_params cparams);
        //LLAMA_API struct falcon_context * falcon_init_from_file(const char * path_model, struct falcon_context_params   params);

        [DllImport("falcon")]
        public static extern void falcon_context_set_buffers(falcon_context ctx, int n_batch, int n_ctx);
        //LLAMA_API void falcon_context_set_buffers(falcon_context *ctx, int n_batch, int n_ctx);

        [DllImport("falcon")]
        public static extern falcon_model falcon_get_falcon_model(falcon_context ctx);
        //LLAMA_API struct falcon_model * falcon_get_falcon_model(falcon_context * ctx);

        [DllImport("falcon")]
        public static extern void llama_free(falcon_context model);
        //LLAMA_API void llama_free(struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern int falcon_model_quantize(string fname_inp, string fname_out, llama_model_quantize_params qparams);
        //LLAMA_API int falcon_model_quantize(const char * fname_inp, const char * fname_out, const llama_model_quantize_params * params);

        [DllImport("falcon")]
        public static extern int llama_apply_lora_from_file(falcon_context ctx, string path_lora, string path_base_model, int n_threads);
        //LLAMA_API int llama_apply_lora_from_file(struct falcon_context * ctx, const char * path_lora, const char * path_base_model, int   n_threads);

        [DllImport("falcon")]
        public static extern int llama_get_kv_cache_token_count(falcon_context ctx);
        //LLAMA_API int llama_get_kv_cache_token_count(const struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern void llama_set_rng_seed(falcon_context ctx, int seed);
        //LLAMA_API void llama_set_rng_seed(struct falcon_context * ctx, int seed);

        [DllImport("falcon")]
        public static extern long llama_get_state_size(falcon_context ctx);
        //LLAMA_API size_t llama_get_state_size(const struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern long falcon_copy_state_data(falcon_context ctx, byte[] dst);
        //LLAMA_API size_t falcon_copy_state_data(struct falcon_context * ctx, uint8_t * dst);

        [DllImport("falcon")]
        public static extern long falcon_set_state_data(falcon_context ctx, byte[] src);
        //LLAMA_API size_t falcon_set_state_data(struct falcon_context * ctx, uint8_t * src);

        [DllImport("falcon")]
        public static extern bool llama_load_session_file(falcon_context ctx, string path_session, falcon_token tokens_out, long n_token_capacity, out long n_token_count_out);
        //LLAMA_API bool llama_load_session_file(struct falcon_context * ctx, const char * path_session, falcon_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);

        [DllImport("falcon")]
        public static extern bool llama_save_session_file(falcon_context ctx, string path_session, falcon_token tokens, long n_token_count);
        //LLAMA_API bool llama_save_session_file(struct falcon_context * ctx, const char * path_session, const falcon_token * tokens, size_t n_token_count);

        [DllImport("falcon")]
        public static extern int falcon_eval(falcon_context ctx, falcon_token[] tokens, int n_tokens, int n_past, int n_threads, int debug_timings);
        //LLAMA_API int falcon_eval(struct falcon_context * ctx, const falcon_token * tokens, int   n_tokens, int   n_past, int   n_threads,  int debug_timings);

        [DllImport("falcon")]
        public static extern int falcon_eval_export(falcon_context ctx, string fname);
        //LLAMA_API int falcon_eval_export(struct falcon_context * ctx, const char * fname);

        [DllImport("falcon")]
        public static extern int falcon_tokenize(falcon_context ctx, string text, falcon_token[] tokens, int n_max_tokens, bool add_bos);
        //LLAMA_API int falcon_tokenize(struct falcon_context * ctx, const char * text, falcon_token * tokens, int   n_max_tokens, bool   add_bos);

        [DllImport("falcon")]
        public static extern int falcon_n_vocab(falcon_context ctx);
        //LLAMA_API int falcon_n_vocab(const struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern int falcon_n_ctx(falcon_context ctx);
        //LLAMA_API int falcon_n_ctx  (const struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern int falcon_n_embd(falcon_context ctx);
        //LLAMA_API int falcon_n_embd (const struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern int falcon_get_vocab(falcon_context ctx, string[] strings, float[] scores, int capacity);
        //LLAMA_API int falcon_get_vocab(const struct falcon_context * ctx, const char * * strings, float * scores, int   capacity);

        [DllImport("falcon")]
        public static extern falcon_context falcon_context_prepare(falcon_context_params cparams, falcon_model model, string context_name, bool verbose);
        //LLAMA_API struct falcon_context * falcon_context_prepare(falcon_context_params params, falcon_model *model, std::string context_name, bool verbose);

        [DllImport("falcon", EntryPoint = "falcon_get_logits")]
        public static extern nint _falcon_get_logits(falcon_context ctx);

        public static List<float> falcon_get_logits(falcon_context ctx)
        {
            var count = falcon_n_vocab(ctx);
            var native_mem = _falcon_get_logits(ctx);
            var logits = new float[count];
            Marshal.Copy(native_mem, logits, 0, count);

            return new(logits);
        }
        //LLAMA_API float * falcon_get_logits(struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern float[] falcon_get_embeddings(falcon_context ctx);
        //LLAMA_API float * falcon_get_embeddings(struct falcon_context * ctx);

        [DllImport("falcon", EntryPoint = "falcon_token_to_str")]
        public static extern nint _falcon_token_to_str(falcon_context ctx, falcon_token token);

        public static string falcon_token_to_str(falcon_context ctx, falcon_token token) => Marshal.PtrToStringUTF8(_falcon_token_to_str(ctx, token)) ?? String.Empty;
        //LLAMA_API const char * falcon_token_to_str(const struct falcon_context * ctx, falcon_token token);

        public enum t_finetune_type { FINETUNE_UNSPECIFIED, FINETUNE_NONE, FINETUNE_ALPACA, FINETUNE_OPENASSISTANT, FINETUNE_WIZARD, FINETUNE_FALCONINSTRUCT }
        //typedef enum { FINETUNE_UNSPECIFIED, FINETUNE_NONE, FINETUNE_ALPACA, FINETUNE_OPENASSISTANT, FINETUNE_WIZARD, FINETUNE_FALCONINSTRUCT } t_finetune_type;

        public static readonly string[] FINETUNE_NAME = { "UNSPECIFIED", "NONE", "ALPACA", "OPENASSISTANT", "WIZARD", "FALCONINSTRUCT" };
        //static const char *FINETUNE_NAME[6] = { "UNSPECIFIED", "NONE", "ALPACA", "OPENASSISTANT", "WIZARD", "FALCONINSTRUCT" };

        [DllImport("falcon")]
        public static extern t_finetune_type falcon_detect_finetune(falcon_context ctx, string model_path);
        //LLAMA_API t_finetune_type falcon_detect_finetune(falcon_context * ctx, std::string model_path);

        [DllImport("falcon")]
        public static extern falcon_token falcon_token_bos();
        //LLAMA_API falcon_token falcon_token_bos();

        [DllImport("falcon")]
        public static extern falcon_token falcon_token_eos();
        //LLAMA_API falcon_token falcon_token_eos();

        [DllImport("falcon")]
        public static extern falcon_token falcon_token_nl();
        //LLAMA_API falcon_token falcon_token_nl();

        [DllImport("falcon", EntryPoint = "llama_sample_repetition_penalty")]
        private static extern void _llama_sample_repetition_penalty(falcon_context ctx, nint candidates, falcon_token[] last_tokens, int last_tokens_size, float penalty);

        public static void llama_sample_repetition_penalty(falcon_context ctx, falcon_token_data_array candidates, List<falcon_token> last_tokens, float penalty)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_repetition_penalty(ctx, new(&_candidates), last_tokens.ToArray(), last_tokens.Count, penalty);
        }
        //LLAMA_API void llama_sample_repetition_penalty(struct falcon_context * ctx, falcon_token_data_array * candidates, const falcon_token * last_tokens, size_t last_tokens_size, float penalty);

        [DllImport("falcon", EntryPoint = "llama_sample_frequency_and_presence_penalties")]
        public static extern void _llama_sample_frequency_and_presence_penalties(falcon_context ctx, nint candidates, falcon_token[] last_tokens, int last_tokens_size, float alpha_frequency, float alpha_presence);

        public static void llama_sample_frequency_and_presence_penalties(falcon_context ctx, falcon_token_data_array candidates, List<falcon_token> last_tokens, float alpha_frequency, float alpha_presence)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_frequency_and_presence_penalties(ctx, new(&_candidates), last_tokens.ToArray(), last_tokens.Count, alpha_frequency, alpha_presence);
        }
        //LLAMA_API void llama_sample_frequency_and_presence_penalties(struct falcon_context * ctx, falcon_token_data_array * candidates, const falcon_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);

        [DllImport("falcon")]
        public static extern void llama_sample_softmax(falcon_context ctx, falcon_token_data_array[] candidates);
        //LLAMA_API void llama_sample_softmax(struct falcon_context * ctx, falcon_token_data_array * candidates);

        [DllImport("falcon")]
        public static extern void llama_sample_log_softmax(falcon_context ctx, falcon_token_data_array[] candidates);
        //LLAMA_API void llama_sample_log_softmax(struct falcon_context * ctx, falcon_token_data_array * candidates);

        [DllImport("falcon", EntryPoint = "llama_sample_top_k")]
        public static extern void _llama_sample_top_k(falcon_context ctx, nint candidates, int k, long min_keep);

        public static void llama_sample_top_k(falcon_context ctx, falcon_token_data_array candidates, int k, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_top_k(ctx, new(&_candidates), k, min_keep);
        }
        //LLAMA_API void llama_sample_top_k(struct falcon_context * ctx, falcon_token_data_array * candidates, int k, size_t min_keep);

        [DllImport("falcon", EntryPoint = "llama_sample_top_p")]
        public static extern void _llama_sample_top_p(falcon_context ctx, nint candidates, float p, long min_keep);

        public static void llama_sample_top_p(falcon_context ctx, falcon_token_data_array candidates, float p, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_top_p(ctx, new(&_candidates), p, min_keep);
        }
        //LLAMA_API void llama_sample_top_p(struct falcon_context * ctx, falcon_token_data_array * candidates, float p, size_t min_keep);

        [DllImport("falcon", EntryPoint = "llama_sample_tail_free")]
        public static extern void _llama_sample_tail_free(falcon_context ctx, nint candidates, float z, long min_keep);

        public static void llama_sample_tail_free(falcon_context ctx, falcon_token_data_array candidates, float z, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_tail_free(ctx, new(&_candidates), z, min_keep);
        }
        //LLAMA_API void llama_sample_tail_free(struct falcon_context * ctx, falcon_token_data_array * candidates, float z, size_t min_keep);

        [DllImport("falcon", EntryPoint = "llama_sample_typical")]
        public static extern void _llama_sample_typical(falcon_context ctx, nint candidates, float p, long min_keep);

        public static void llama_sample_typical(falcon_context ctx, falcon_token_data_array candidates, float p, int min_keep = 1)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_typical(ctx, new(&_candidates), p, min_keep);
        }
        //LLAMA_API void llama_sample_typical(struct falcon_context * ctx, falcon_token_data_array * candidates, float p, size_t min_keep);

        [DllImport("falcon", EntryPoint = "llama_sample_temperature")]
        public static extern void _llama_sample_temperature(falcon_context ctx, nint candidates, float temp);

        public static void llama_sample_temperature(falcon_context ctx, falcon_token_data_array candidates, float temp)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            _llama_sample_temperature(ctx, new(&_candidates), temp);
        }
        //LLAMA_API void llama_sample_temperature(struct falcon_context * ctx, falcon_token_data_array * candidates, float temp);

        [DllImport("falcon", EntryPoint = "llama_sample_token_mirostat")]
        public static extern falcon_token _llama_sample_token_mirostat(falcon_context ctx, nint candidates, float tau, float eta, int m, ref float mu);

        public static falcon_token llama_sample_token_mirostat(falcon_context ctx, falcon_token_data_array candidates, float tau, float eta, int m, ref float mu)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_mirostat(ctx, new(&_candidates), tau, eta, m, ref mu);
        }
        //LLAMA_API falcon_token llama_sample_token_mirostat(struct falcon_context * ctx, falcon_token_data_array * candidates, float tau, float eta, int m, float * mu);

        [DllImport("falcon", EntryPoint = "llama_sample_token_mirostat_v2")]
        public static extern falcon_token _llama_sample_token_mirostat_v2(falcon_context ctx, nint candidates, float tau, float eta, ref float mu);

        public static falcon_token llama_sample_token_mirostat_v2(falcon_context ctx, falcon_token_data_array candidates, float tau, float eta, ref float mu)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_mirostat_v2(ctx, new(&_candidates), tau, eta, ref mu);
        }
        //LLAMA_API falcon_token llama_sample_token_mirostat_v2(struct falcon_context * ctx, falcon_token_data_array * candidates, float tau, float eta, float * mu);

        [DllImport("falcon", EntryPoint = "llama_sample_token_greedy")]
        public static extern falcon_token _llama_sample_token_greedy(falcon_context ctx, nint candidates);

        public static falcon_token llama_sample_token_greedy(falcon_context ctx, falcon_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token_greedy(ctx, new(&_candidates));
        }
        //LLAMA_API falcon_token llama_sample_token_greedy(struct falcon_context * ctx, falcon_token_data_array * candidates);

        [DllImport("falcon", EntryPoint = "llama_sample_token")]
        public static extern falcon_token _llama_sample_token(falcon_context ctx, nint candidates);

        public static falcon_token llama_sample_token(falcon_context ctx, falcon_token_data_array candidates)
        {
            using var handle = candidates.data.Pin();
            var _candidates = new FalconCppInterop._falcon_token_data_array
            {
                data = new(handle.Pointer),
                size = candidates.size,
                sorted = candidates.sorted,
            };

            return _llama_sample_token(ctx, new(&_candidates));
        }
        //LLAMA_API falcon_token llama_sample_token(struct falcon_context * ctx, falcon_token_data_array * candidates);

        [DllImport("falcon")]
        public static extern void falcon_print_timings(falcon_context ctx);
        //LLAMA_API void falcon_print_timings(struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern void llama_reset_timings(falcon_context ctx);
        //LLAMA_API void llama_reset_timings(struct falcon_context * ctx);

        [DllImport("falcon")]
        public static extern string falcon_print_system_info(int n_threads, int n_cores);
        //LLAMA_API const char * falcon_print_system_info(int n_threads, int n_cores);

        [DllImport("falcon")]
        public static extern nint falcon_cuda_get_system_gpu_status();

        [DllImport("falcon")]
        public static extern void falcon_cuda_print_gpu_status(nint status, bool print_summary);

        [DllImport("falcon")]
        public static extern void falcon_cuda_set_max_gpus();

        [DllImport("falcon")]
        public static extern void falcon_cuda_set_main_device();
    }
}
