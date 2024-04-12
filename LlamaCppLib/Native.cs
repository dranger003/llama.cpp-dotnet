using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    using llama_model = System.IntPtr;
    using llama_context = System.IntPtr;
    using llama_token = System.Int32;
    using llama_pos = System.Int32;
    using llama_seq_id = System.Int32;

    using unsafe llama_progress_callback = delegate* unmanaged[Cdecl]<float, void*, sbyte>;
    using unsafe ggml_backend_sched_eval_callback = delegate* unmanaged[Cdecl]<nint, sbyte, void*, sbyte>;
    using unsafe ggml_abort_callback = delegate* unmanaged[Cdecl]<void*, sbyte>;

    public static unsafe partial class Native
    {
#if WINDOWS
        private const string LibName = $"{nameof(LlamaCppLib)}/llama";
#elif LINUX || MACOS
        private const string LibName = $"{nameof(LlamaCppLib)}/libllama";
#endif

        public enum ggml_numa_strategy
        {
            GGML_NUMA_STRATEGY_DISABLED = 0,
            GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
            GGML_NUMA_STRATEGY_ISOLATE = 2,
            GGML_NUMA_STRATEGY_NUMACTL = 3,
            GGML_NUMA_STRATEGY_MIRROR = 4,
            GGML_NUMA_STRATEGY_COUNT
        }

        public enum llama_vocab_type_t
        {
            LLAMA_VOCAB_TYPE_NONE = 0,
            LLAMA_VOCAB_TYPE_SPM = 1,
            LLAMA_VOCAB_TYPE_BPE = 2,
            LLAMA_VOCAB_TYPE_WPM = 3,
        }

        public enum llama_model_kv_override_type { LLAMA_KV_OVERRIDE_INT, LLAMA_KV_OVERRIDE_FLOAT, LLAMA_KV_OVERRIDE_BOOL };

        public enum ggml_type
        {
            GGML_TYPE_F32 = 0,
            GGML_TYPE_F16 = 1,
            GGML_TYPE_Q4_0 = 2,
            GGML_TYPE_Q4_1 = 3,
            // GGML_TYPE_Q4_2 = 4, // removed
            // GGML_TYPE_Q4_3 = 5, // removed
            GGML_TYPE_Q5_0 = 6,
            GGML_TYPE_Q5_1 = 7,
            GGML_TYPE_Q8_0 = 8,
            GGML_TYPE_Q8_1 = 9,
            GGML_TYPE_Q2_K = 10,
            GGML_TYPE_Q3_K = 11,
            GGML_TYPE_Q4_K = 12,
            GGML_TYPE_Q5_K = 13,
            GGML_TYPE_Q6_K = 14,
            GGML_TYPE_Q8_K = 15,
            GGML_TYPE_IQ2_XXS = 16,
            GGML_TYPE_IQ2_XS = 17,
            GGML_TYPE_IQ3_XXS = 18,
            GGML_TYPE_IQ1_S = 19,
            GGML_TYPE_IQ4_NL = 20,
            GGML_TYPE_IQ3_S = 21,
            GGML_TYPE_IQ2_S = 22,
            GGML_TYPE_IQ4_XS = 23,
            GGML_TYPE_I8 = 24,
            GGML_TYPE_I16 = 25,
            GGML_TYPE_I32 = 26,
            GGML_TYPE_I64 = 27,
            GGML_TYPE_F64 = 28,
            GGML_TYPE_IQ1_M = 29,
            GGML_TYPE_COUNT,
        }

        public enum llama_split_mode
        {
            LLAMA_SPLIT_NONE = 0,
            LLAMA_SPLIT_LAYER = 1,
            LLAMA_SPLIT_ROW = 2,
        }

        public enum llama_rope_scaling_type
        {
            LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
            LLAMA_ROPE_SCALING_TYPE_NONE = 0,
            LLAMA_ROPE_SCALING_TYPE_LINEAR = 1,
            LLAMA_ROPE_SCALING_TYPE_YARN = 2,
            LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_YARN,
        }

        public enum llama_pooling_type
        {
            LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
            LLAMA_POOLING_TYPE_NONE = 0,
            LLAMA_POOLING_TYPE_MEAN = 1,
            LLAMA_POOLING_TYPE_CLS = 2,
        }

        [StructLayout(LayoutKind.Explicit)]
        public struct union_d5855d6a
        {
            [FieldOffset(0)] public long int_value;
            [FieldOffset(0)] public double float_value;
            [FieldOffset(0)] public sbyte bool_value;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_kv_override
        {
            public fixed char key[128];
            public llama_model_kv_override_type tag;
            public union_d5855d6a value;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_params
        {
            public int n_gpu_layers;
            public llama_split_mode split_mode;

            public int main_gpu;
            public float* tensor_split;

            public llama_progress_callback progress_callback;
            public void* progress_callback_user_data;

            public llama_model_kv_override* kv_overrides;

            public byte vocab_only;
            public byte use_mmap;
            public byte use_mlock;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public uint seed;
            public uint n_ctx;
            public uint n_batch;
            public uint n_ubatch;
            public uint n_seq_max;
            public uint n_threads;
            public uint n_threads_batch;

            public llama_rope_scaling_type rope_scaling_type;
            public llama_pooling_type pooling_type;

            public float rope_freq_base;
            public float rope_freq_scale;
            public float yarn_ext_factor;
            public float yarn_attn_factor;
            public float yarn_beta_fast;
            public float yarn_beta_slow;
            public uint yarn_orig_ctx;
            public float defrag_thold;

            public ggml_backend_sched_eval_callback cb_eval;
            public void* cb_eval_user_data;

            public ggml_type type_k;
            public ggml_type type_v;

            public byte logits_all;
            public byte embeddings;
            public byte offload_kqv;

            public ggml_abort_callback abort_callback;
            public void* abort_callback_data;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_batch
        {
            public int n_tokens;

            public llama_token* token;
            public float* embd;
            public llama_pos* pos;
            public int* n_seq_id;
            public llama_seq_id** seq_id;
            public sbyte* logits;

            public llama_pos all_pos_0;
            public llama_pos all_pos_1;
            public llama_seq_id all_seq_id;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_token_data
        {
            public llama_token id;
            public float logit;
            public float p;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_token_data_array
        {
            public llama_token_data* data;
            public nuint size;
            public byte sorted;
        }

        [LibraryImport(LibName)]
        public static partial llama_model_params llama_model_default_params();

        [LibraryImport(LibName)]
        public static partial llama_context_params llama_context_default_params();

        [LibraryImport(LibName)]
        public static partial void llama_backend_init();

        [LibraryImport(LibName)]
        public static partial void llama_numa_init(
            ggml_numa_strategy numa);

        [LibraryImport(LibName)]
        public static partial void llama_backend_free();

        [LibraryImport(LibName)]
        public static partial llama_model llama_load_model_from_file(
            [MarshalAs(UnmanagedType.LPStr)] string path_model,
            llama_model_params cparams);

        [LibraryImport(LibName)]
        public static partial void llama_free_model(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial llama_context llama_new_context_with_model(
            llama_model model,
            llama_context_params cparams);

        [LibraryImport(LibName)]
        public static partial void llama_free(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial uint llama_n_ctx(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial uint llama_n_batch(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial uint llama_n_ubatch(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial uint llama_n_seq_max(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial llama_vocab_type_t llama_vocab_type(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_n_vocab(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_n_ctx_train(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_n_embd(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_n_layer(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial llama_model llama_get_model(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial llama_batch llama_batch_get_one(
            llama_token[] tokens,
            int n_tokens,
            llama_pos pos_0,
            llama_seq_id seq_id);

        [LibraryImport(LibName)]
        public static partial llama_batch llama_batch_get_one(
            Span<llama_token> tokens,
            int n_tokens,
            llama_pos pos_0,
            llama_seq_id seq_id);

        [LibraryImport(LibName)]
        public static partial llama_batch llama_batch_init(
            int n_tokens,
            int embd,
            int n_seq_max);

        [LibraryImport(LibName)]
        public static partial void llama_batch_free(
            llama_batch batch);

        [LibraryImport(LibName)]
        public static partial int llama_tokenize(
            llama_model model,
            [MarshalAs(UnmanagedType.LPStr)] string text,
            int text_len,
            llama_token[] tokens,
            int n_tokens_max,
            [MarshalAs(UnmanagedType.I1)] bool add_bos,
            [MarshalAs(UnmanagedType.I1)] bool special);

        [LibraryImport(LibName)]
        public static partial int llama_token_to_piece(
            llama_model model,
            llama_token token,
            byte[] buf,
            int length);

        [LibraryImport(LibName)]
        public static partial llama_token llama_token_eos(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_add_bos_token(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_add_eos_token(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial llama_token llama_token_eot(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial void llama_kv_cache_clear(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial byte llama_kv_cache_seq_rm(
            llama_context ctx,
            llama_seq_id seq_id,
            llama_pos p0,
            llama_pos p1);

        [LibraryImport(LibName)]
        public static partial int llama_decode(
            llama_context ctx,
            llama_batch batch);

        [LibraryImport(LibName)]
        public static partial void llama_set_causal_attn(
            llama_context ctx,
            [MarshalAs(UnmanagedType.I1)] bool causal_attn);

        [LibraryImport(LibName)]
        public static partial void llama_synchronize(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial float* llama_get_logits_ith(
            llama_context ctx,
            int i);

        [LibraryImport(LibName)]
        public static partial float* llama_get_embeddings(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial float* llama_get_embeddings_ith(
            llama_context ctx,
            int i);

        [LibraryImport(LibName)]
        public static partial float* llama_get_embeddings_seq(
            llama_context ctx,
            llama_seq_id seq_id);

        [LibraryImport(LibName)]
        public static partial void llama_sample_repetition_penalties(
            llama_context ctx,
            ref llama_token_data_array candidates,
            Span<llama_token> last_tokens,
            nuint penalty_last_n,
            float penalty_repeat,
            float penalty_freq,
            float penalty_present);

        [LibraryImport(LibName)]
        public static partial llama_token llama_sample_token_greedy(
            llama_context ctx,
            ref llama_token_data_array candidates);

        [LibraryImport(LibName)]
        public static partial void llama_sample_softmax(
            llama_context ctx,
            ref llama_token_data_array candidates);

        [LibraryImport(LibName)]
        public static partial void llama_sample_temp(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float temp);

        [LibraryImport(LibName)]
        public static partial llama_token llama_sample_token_mirostat(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float tau,
            float eta,
            int m,
            ref float mu);

        [LibraryImport(LibName)]
        public static partial llama_token llama_sample_token_mirostat_v2(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float tau,
            float eta,
            ref float mu);

        [LibraryImport(LibName)]
        public static partial void llama_sample_top_k(
            llama_context ctx,
            ref llama_token_data_array candidates,
            int k,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial void llama_sample_tail_free(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float z,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial void llama_sample_typical(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float p,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial void llama_sample_top_p(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float p,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial void llama_sample_min_p(
            llama_context ctx,
            ref llama_token_data_array candidates,
            float p,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial llama_token llama_sample_token(
            llama_context ctx,
            ref llama_token_data_array candidates);
    }
}
