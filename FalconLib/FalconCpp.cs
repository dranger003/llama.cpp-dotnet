namespace FalconCppLib
{
    using falcon_context = System.IntPtr;
    using falcon_token = System.Int32;

    public class FalconCpp
    {
        public static FalconCppInterop.falcon_context_params falcon_context_params_create()
        {
            var cparams = FalconCppInterop.falcon_context_default_params();

            cparams.n_ctx = 2048;
            cparams.n_batch = 1;
            cparams.n_gpu_layers = 28;
            cparams.main_gpu = 0;
            cparams.tensor_split = new float[16];
            cparams.seed = -1;
            cparams.f16_kv = false; // unsupported because ggml_repeat2 currently only implemented for f32
            cparams.use_mmap = true;
            cparams.use_mlock = false;
            cparams.logits_all = false;
            cparams.embedding = false;

            return cparams;
        }

        public static List<falcon_token> falcon_tokenize(falcon_context ctx, string text, bool addBos = false)
        {
            var tokens = new falcon_token[FalconCppInterop.falcon_n_ctx(ctx)];
            var count = FalconCppInterop.falcon_tokenize(ctx, text, tokens, tokens.Length, addBos);
            return new(tokens.Take(count));
        }
    }
}
