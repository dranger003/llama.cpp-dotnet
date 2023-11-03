using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    using llama_model = System.IntPtr;
    using llama_context = System.IntPtr;
    using llama_token = System.Int32;
    using llama_pos = System.Int32;
    using llama_seq_id = System.Int32;

    public static unsafe partial class Native
    {
#if WINDOWS
        private const string LibName = $"{nameof(LlamaCppLib)}/llama";
#elif LINUX || MACOS
        private const string LibName = $"{nameof(LlamaCppLib)}/libllama";
#endif

        [StructLayout(LayoutKind.Sequential)]
        public struct llama_model_params
        {
            public int n_gpu_layers;
            public int main_gpu;
            public float* tensor_split;

            public delegate* unmanaged[Cdecl]<float, void*, void> progress_callback;
            public void* progress_callback_user_data;

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
            public uint n_threads;
            public uint n_threads_batch;
            public sbyte rope_scaling_type;

            public float rope_freq_base;
            public float rope_freq_scale;
            public float yarn_ext_factor;
            public float yarn_attn_factor;
            public float yarn_beta_fast;
            public float yarn_beta_slow;
            public uint yarn_orig_ctx;

            public byte mul_mat_q;
            public byte f16_kv;
            public byte logits_all;
            public byte embedding;
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
            public byte* logits;

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
        public static partial void llama_backend_init(
            [MarshalAs(UnmanagedType.I1)] bool numa);

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
        public static partial int llama_n_ctx(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial int llama_n_vocab(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial llama_model llama_get_model(
            llama_context ctx);

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
            llama_token* tokens,
            int n_max_tokens,
            byte add_bos,
            byte special);

        [LibraryImport(LibName)]
        public static partial int llama_token_to_piece(
            llama_model model,
            llama_token token,
            byte* buf,
            int length);

        [LibraryImport(LibName)]
        public static partial llama_token llama_token_eos(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial llama_token llama_token_eot(
            llama_model model);

        [LibraryImport(LibName)]
        public static partial void llama_kv_cache_clear(
            llama_context ctx);

        [LibraryImport(LibName)]
        public static partial void llama_kv_cache_seq_rm(
            llama_context ctx,
            llama_seq_id seq_id,
            llama_pos p0,
            llama_pos p1);

        [LibraryImport(LibName)]
        public static partial int llama_decode(
            llama_context ctx,
            llama_batch batch);

        [LibraryImport(LibName)]
        public static partial float* llama_get_logits_ith(
            llama_context ctx,
            int i);

        [LibraryImport(LibName)]
        public static partial void llama_sample_repetition_penalties(
            llama_context ctx,
            ref llama_token_data_array candidates,
            llama_token* last_tokens,
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
        public static partial llama_token llama_sample_token(
            llama_context ctx,
            ref llama_token_data_array candidates);
    }
}
