using System.Reflection;
using System.Runtime.InteropServices;

namespace BertCppLib
{
    using bert_ctx = System.IntPtr;
    using bert_vocab_id = System.Int32;

    public static class BertCppInterop
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct _bert_params
        {
            public int n_threads;
            public int port;
            public nint model;
            public nint prompt;
        }

        public struct bert_params
        {
            public int n_threads;
            public int port;
            public string model;
            public string prompt;
        }

        [DllImport("bert", EntryPoint = "bert_params_parse")]
        private static extern bool _bert_params_parse(int argc, nint[] argv, ref _bert_params bparams);

        public static bool bert_params_parse(string[] args, out bert_params bparams)
        {
            var _bparams = new _bert_params();

            var _args = new[] { Path.GetFileName(Assembly.GetEntryAssembly()?.Location ?? String.Empty) }
                .Concat(args)
                .Select(Marshal.StringToHGlobalAnsi)
                .ToArray();

            var result = _bert_params_parse(_args.Length, _args, ref _bparams);

            bparams = new();
            bparams.n_threads = _bparams.n_threads;
            bparams.port = _bparams.port;
            bparams.model = Marshal.PtrToStringUTF8(_bparams.model) ?? String.Empty;
            bparams.prompt = Marshal.PtrToStringUTF8(_bparams.prompt) ?? String.Empty;

            foreach (var _arg in _args)
                Marshal.FreeHGlobal(_arg);

            return result;
        }

        [DllImport("bert")]
        public static extern bert_ctx bert_load_from_file(string fname);

        [DllImport("bert")]
        public static extern void bert_free(bert_ctx ctx);

        [DllImport("bert")]
        public static extern void bert_encode(bert_ctx ctx, int n_threads, string texts, float[] embeddings);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="n_threads"></param>
        /// <param name="n_batch_size">how many to process at a time</param>
        /// <param name="n_inputs">total size of texts and embeddings arrays</param>
        /// <param name="texts"></param>
        /// <param name="embeddings"></param>
        [DllImport("bert")]
        public static extern void bert_encode_batch(bert_ctx ctx, int n_threads, int n_batch_size, int n_inputs, string[] texts, float[][] embeddings);

        /// <summary>
        /// Api for separate tokenization & eval
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="text"></param>
        /// <param name="tokens"></param>
        /// <param name="n_tokens"></param>
        /// <param name="n_max_tokens"></param>
        [DllImport("bert", EntryPoint = "bert_tokenize")]
        private static extern void _bert_tokenize(bert_ctx ctx, string text, bert_vocab_id[] tokens, ref int n_tokens, int n_max_tokens);

        public static bert_vocab_id[] bert_tokenize(bert_ctx ctx, string text)
        {
            var n_ctx = bert_n_max_tokens(ctx);
            var _tokens = new bert_vocab_id[n_ctx];
            var n_tokens = 0;
            _bert_tokenize(ctx, text, _tokens, ref n_tokens, n_ctx);
            return _tokens.Take(n_tokens).ToArray();
        }

        [DllImport("bert", EntryPoint = "bert_eval")]
        private static extern void _bert_eval(bert_ctx ctx, int n_threads, bert_vocab_id[] tokens, int n_tokens, float[] embeddings);

        public static float[] bert_eval(bert_ctx ctx, int n_threads, bert_vocab_id[] tokens, int? n_tokens = null)
        {
            var n_embd = bert_n_embd(ctx);
            var embeddings = new float[n_embd];
            _bert_eval(ctx, n_threads, tokens, n_tokens ?? tokens.Length, embeddings);
            return embeddings;
        }

        /// <summary>
        /// NOTE: for batch processing the longest input must be first
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="n_threads"></param>
        /// <param name="n_batch_size"></param>
        /// <param name="batch_tokens"></param>
        /// <param name="n_tokens"></param>
        /// <param name="batch_embeddings"></param>
        [DllImport("bert")]
        public static extern void bert_eval_batch(bert_ctx ctx, int n_threads, int n_batch_size, bert_vocab_id[][] batch_tokens, ref int n_tokens, float[][] batch_embeddings);

        [DllImport("bert")]
        public static extern int bert_n_embd(bert_ctx ctx);

        [DllImport("bert")]
        public static extern int bert_n_max_tokens(bert_ctx ctx);

        [DllImport("bert")]
        public static extern string bert_vocab_id_to_token(bert_ctx ctx, bert_vocab_id id);
    }
}
