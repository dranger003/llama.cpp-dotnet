using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    using llama_model = System.IntPtr;
    using llama_context = System.IntPtr;
    using llama_sampler = System.IntPtr;
    using llama_token = System.Int32;
    using llama_pos = System.Int32;
    using llama_seq_id = System.Int32;

    // ggml.h

    using unsafe ggml_backend_sched_eval_callback = delegate* unmanaged[Cdecl]<nint, sbyte, void*, sbyte>;
    using unsafe ggml_abort_callback = delegate* unmanaged[Cdecl]<void*, sbyte>;

    // llama.h

    using unsafe llama_progress_callback = delegate* unmanaged[Cdecl]<float, void*, sbyte>;

    public static unsafe partial class Native
    {
#if WINDOWS
        private const string LibName = $"{nameof(LlamaCppLib)}/llama";
#elif LINUX || MACOS
        private const string LibName = $"{nameof(LlamaCppLib)}/libllama";
#endif

        // ggml.h

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
            GGML_TYPE_BF16 = 30,
            GGML_TYPE_Q4_0_4_4 = 31,
            GGML_TYPE_Q4_0_4_8 = 32,
            GGML_TYPE_Q4_0_8_8 = 33,
            GGML_TYPE_TQ1_0 = 34,
            GGML_TYPE_TQ2_0 = 35,
            GGML_TYPE_COUNT,
        }

        public enum ggml_numa_strategy
        {
            GGML_NUMA_STRATEGY_DISABLED = 0,
            GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
            GGML_NUMA_STRATEGY_ISOLATE = 2,
            GGML_NUMA_STRATEGY_NUMACTL = 3,
            GGML_NUMA_STRATEGY_MIRROR = 4,
            GGML_NUMA_STRATEGY_COUNT
        }

        // llama.h

        public enum _llama_vocab_type
        {
            LLAMA_VOCAB_TYPE_NONE = 0,
            LLAMA_VOCAB_TYPE_SPM = 1,
            LLAMA_VOCAB_TYPE_BPE = 2,
            LLAMA_VOCAB_TYPE_WPM = 3,
            LLAMA_VOCAB_TYPE_UGM = 4,
            LLAMA_VOCAB_TYPE_RWKV = 5,
        };

        public enum llama_rope_scaling_type
        {
            LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
            LLAMA_ROPE_SCALING_TYPE_NONE = 0,
            LLAMA_ROPE_SCALING_TYPE_LINEAR = 1,
            LLAMA_ROPE_SCALING_TYPE_YARN = 2,
            LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_YARN,
        }

        public enum _llama_pooling_type
        {
            LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
            LLAMA_POOLING_TYPE_NONE = 0,
            LLAMA_POOLING_TYPE_MEAN = 1,
            LLAMA_POOLING_TYPE_CLS = 2,
            LLAMA_POOLING_TYPE_LAST = 3,
            LLAMA_POOLING_TYPE_RANK = 4,
        }

        public enum llama_attention_type
        {
            LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
            LLAMA_ATTENTION_TYPE_CAUSAL = 0,
            LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1,
        }

        public enum llama_split_mode
        {
            LLAMA_SPLIT_NONE = 0,
            LLAMA_SPLIT_LAYER = 1,
            LLAMA_SPLIT_ROW = 2,
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
        }

        public enum llama_model_kv_override_type
        {
            LLAMA_KV_OVERRIDE_TYPE_INT,
            LLAMA_KV_OVERRIDE_TYPE_FLOAT,
            LLAMA_KV_OVERRIDE_TYPE_BOOL,
            LLAMA_KV_OVERRIDE_TYPE_STR,
        };

        [StructLayout(LayoutKind.Explicit)]
        public struct llama_model_kv_override_value
        {
            [FieldOffset(0)] public long val_i64;
            [FieldOffset(0)] public double val_f64;
            [FieldOffset(0)] public sbyte val_bool;
            [FieldOffset(0)] public fixed byte val_str[128];
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_kv_override
        {
            public llama_model_kv_override_type tag;
            public fixed byte key[128];
            public llama_model_kv_override_value value;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_params
        {
            public void* devices;

            public int n_gpu_layers;
            public llama_split_mode split_mode;

            public int main_gpu;

            public readonly float* tensor_split;

            public byte* rpc_servers;

            public llama_progress_callback progress_callback;

            public void* progress_callback_user_data;

            public readonly llama_model_kv_override* kv_overrides;

            public sbyte vocab_only;
            public sbyte use_mmap;
            public sbyte use_mlock;
            public sbyte check_tensors;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_context_params
        {
            public uint n_ctx;
            public uint n_batch;
            public uint n_ubatch;
            public uint n_seq_max;
            public int n_threads;
            public int n_threads_batch;

            public llama_rope_scaling_type rope_scaling_type;
            public _llama_pooling_type pooling_type;
            public llama_attention_type attention_type;

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

            public sbyte logits_all;
            public sbyte embeddings;
            public sbyte offload_kqv;
            public sbyte flash_attn;
            public sbyte no_perf;

            public ggml_abort_callback abort_callback;
            public void* abort_callback_data;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_sampler_chain_params
        {
            public sbyte no_perf;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_chat_message
        {
            public byte* role;
            public byte* content;
        }

        [LibraryImport(LibName)]
        public static partial llama_model_params llama_model_default_params();

        [LibraryImport(LibName)]
        public static partial llama_context_params llama_context_default_params();

        [LibraryImport(LibName)]
        public static partial llama_sampler_chain_params llama_sampler_chain_default_params();

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
            llama_model_params mparams);

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
        public static partial _llama_pooling_type llama_pooling_type(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial _llama_vocab_type llama_vocab_type(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_model_meta_val_str(
            llama_model model,
            [In] byte[] key,
            [In, Out] byte[] buf,
            nuint buf_size);

        [LibraryImport(LibName)]
        public static partial int llama_model_meta_count(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_model_meta_key_by_index(
            llama_model model,
            int i,
            [In, Out] byte[] buf,
            nuint buf_size);

        [LibraryImport(LibName)]
        public static partial int llama_model_meta_val_str_by_index(
            llama_model model,
            int i,
            [In, Out] byte[] buf,
            nuint buf_size);

        [LibraryImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool llama_model_has_encoder(
            llama_model model);

        [LibraryImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool llama_model_has_decoder(
            llama_model model);

        //
        // KV cache
        //

        [LibraryImport(LibName)]
        public static partial void llama_kv_cache_clear(
            llama_context ctx);

        [LibraryImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool llama_kv_cache_seq_rm(
            llama_context ctx,
            llama_seq_id seq_id,
            llama_pos p0,
            llama_pos p1);

        //
        // State / sessions
        //

        [LibraryImport(LibName)]
        public static partial nuint llama_state_get_size(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_get_data(
            llama_context ctx,
            [In, Out] byte[] dst,
            nuint size);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_set_data(
            llama_context ctx,
            [In] byte[] src,
            nuint size);

        [LibraryImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool llama_state_load_file(
            llama_context ctx,
            [In] byte[] path_session,
            [In, Out] llama_token[] tokens_out,
            nuint n_token_capacity,
            ref nuint n_token_count_out);

        [LibraryImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool llama_state_save_file(
            llama_context ctx,
            [In] byte[] path_session,
            [In] llama_token[] tokens,
            nuint n_token_count);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_seq_get_size(
            llama_context ctx,
            llama_seq_id seq_id);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_seq_get_data(
            llama_context ctx,
            [In, Out] byte[] dst,
            nuint size,
            llama_seq_id seq_id);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_seq_set_data(
            llama_context ctx,
            [In] byte[] src,
            nuint size,
            llama_seq_id dest_seq_id);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_seq_save_file(
            llama_context ctx,
            [In] byte[] filepath,
            llama_seq_id seq_id,
            [In] llama_token[] tokens,
            nuint n_token_count);

        [LibraryImport(LibName)]
        public static partial nuint llama_state_seq_load_file(
            llama_context ctx,
            [In] byte[] filepath,
            llama_seq_id dest_seq_id,
            [In, Out] llama_token[] tokens_out,
            nuint n_token_capacity,
            ref nuint n_token_count_out);

        //
        // Decoding
        //

        [LibraryImport(LibName)]
        public static partial llama_batch llama_batch_init(
            int n_tokens,
            int embd,
            int n_seq_max);

        [LibraryImport(LibName)]
        public static partial void llama_batch_free(
            llama_batch batch);

        [LibraryImport(LibName)]
        public static partial int llama_encode(
            llama_context ctx,
            llama_batch batch);

        [LibraryImport(LibName)]
        public static partial int llama_decode(
            llama_context ctx,
            llama_batch batch);

        [LibraryImport(LibName)]
        public static partial void llama_set_embeddings(
            llama_context ctx,
            [MarshalAs(UnmanagedType.I1)] bool embeddings);

        [LibraryImport(LibName)]
        public static partial void llama_set_causal_attn(
            llama_context ctx,
            [MarshalAs(UnmanagedType.I1)] bool causal_attn);

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

        //
        // Vocab
        //

        [LibraryImport(LibName)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static partial bool llama_token_is_eog(
            llama_model model,
            llama_token token);

        [LibraryImport(LibName)]
        public static partial llama_token llama_token_eos(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_add_bos_token(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial int llama_add_eos_token(
            llama_model model);

        //
        // Tokenization
        //

        [LibraryImport(LibName)]
        public static partial int llama_tokenize(
            llama_model model,
            [In] byte[] text,
            int text_len,
            [In, Out] llama_token[] tokens,
            int n_tokens_max,
            [MarshalAs(UnmanagedType.I1)] bool add_special,
            [MarshalAs(UnmanagedType.I1)] bool parse_special);

        [LibraryImport(LibName)]
        public static partial int llama_token_to_piece(
            llama_model model,
            llama_token token,
            [In, Out] byte[] buf,
            int length,
            int lstrip,
            [MarshalAs(UnmanagedType.I1)] bool special);

        [LibraryImport(LibName)]
        public static partial int llama_detokenize(
            llama_model model,
            [In] llama_token[] tokens,
            int n_tokens,
            [In, Out] byte[] text,
            int text_len_max,
            [MarshalAs(UnmanagedType.I1)] bool remove_special,
            [MarshalAs(UnmanagedType.I1)] bool unparse_special);

        //
        // Chat templates
        //

        [LibraryImport(LibName)]
        public static partial int llama_chat_apply_template(
            nint model,
            [In] byte[]? tmpl,
            [In] llama_chat_message[] chat,
            nuint n_msg,
            [MarshalAs(UnmanagedType.I1)] bool add_ass,
            [In, Out] byte[] buf,
            int length);

        //
        // Sampling API
        //

        [LibraryImport(LibName)]
        public static partial void llama_sampler_reset(
            llama_sampler smpl);

        [LibraryImport(LibName)]
        public static partial void llama_sampler_free(
            nint smpl);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_chain_init(
            llama_sampler_chain_params sparams);

        [LibraryImport(LibName)]
        public static partial void llama_sampler_chain_add(
            llama_sampler chain,
            llama_sampler smpl);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_greedy();

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_dist(
            uint seed);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_top_k(
            int k);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_top_p(
            float p,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_min_p(
            float p,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_typical(
            float p,
            nuint min_keep);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_temp(
            float t);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_temp_ext(
            float t,
            float delta,
            float exponent);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_xtc(
            float p,
            float t,
            nuint min_keep,
            uint seed);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_mirostat(
            int n_vocab,
            uint seed,
            float tau,
            float eta,
            int m);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_mirostat_v2(
            uint seed,
            float tau,
            float eta);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_penalties(
            int penalty_last_n,
            float penalty_repeat,
            float penalty_freq,
            float penalty_present);

        [LibraryImport(LibName)]
        public static partial llama_sampler llama_sampler_init_dry(
            llama_model model,
            float dry_multiplier,
            float dry_base,
            int dry_allowed_length,
            int dry_penalty_last_n,
            [In] byte[][] seq_breakers,
            nuint num_breakers);

        [LibraryImport(LibName)]
        public static partial int llama_sampler_sample(
            llama_sampler smpl,
            llama_context ctx, int idx);
    }
}
