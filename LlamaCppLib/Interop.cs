using System.Buffers;
using System.Text;

namespace LlamaCppLib
{
    using llama_model = System.IntPtr;
    using llama_context = System.IntPtr;
    using llama_token = System.Int32;
    using llama_pos = System.Int32;
    using llama_seq_id = System.Int32;

    public static unsafe class Interop
    {
        public static void llama_batch_add(ref Native.llama_batch batch, llama_token id, llama_pos pos, llama_seq_id[] seq_ids, bool logits)
        {
            batch.token[batch.n_tokens] = id;
            batch.pos[batch.n_tokens] = pos;
            batch.n_seq_id[batch.n_tokens] = seq_ids.Length;

            for (var i = 0; i < seq_ids.Length; ++i)
                batch.seq_id[batch.n_tokens][i] = seq_ids[i];

            batch.logits[batch.n_tokens] = (sbyte)(logits ? 1 : 0);
            batch.n_tokens++;
        }

        public static void llama_batch_clear(ref Native.llama_batch batch)
        {
            batch.n_tokens = 0;
        }

        public static int[] llama_tokenize(llama_model model, byte[] text, bool add_special = false, bool parse_special = false)
        {
            var length = -Native.llama_tokenize(model, text, text.Length, [], 0, add_special, parse_special);

            var tokens = new int[length];
            Native.llama_tokenize(model, text, text.Length, tokens, tokens.Length, add_special, parse_special);

            return tokens;
        }

        public static byte[] llama_detokenize(llama_model model, int[] tokens, bool remove_special = false, bool unparse_special = false)
        {
            var length = -Native.llama_detokenize(model, tokens, tokens.Length, [], 0, remove_special, unparse_special);

            var text = new byte[length];
            Native.llama_detokenize(model, tokens, tokens.Length, text, text.Length, remove_special, unparse_special);

            return text;
        }

        private static byte[] _bytes = new byte[1024];

        public static byte[] llama_token_to_piece(llama_model model, int token, bool special)
        {
            var count = Native.llama_token_to_piece(model, token, _bytes, _bytes.Length, 0, special);
            return _bytes[0..count];
        }

        public static unsafe string llama_apply_template(llama_context context, List<LlmMessage> messages, bool appendAssistant = true)
        {
            var encoding = Encoding.UTF8;

            var chat = new Native.llama_chat_message[messages.Count];

            var pinnedRoles = new Memory<byte>[messages.Count];
            var pinnedContents = new Memory<byte>[messages.Count];

            var roleHandles = new MemoryHandle[messages.Count];
            var contentHandles = new MemoryHandle[messages.Count];

            try
            {
                for (var i = 0; i < messages.Count; i++)
                {
                    pinnedRoles[i] = encoding.GetBytes(messages[i].Role ?? String.Empty);
                    pinnedContents[i] = encoding.GetBytes(messages[i].Content ?? String.Empty);

                    roleHandles[i] = pinnedRoles[i].Pin();
                    contentHandles[i] = pinnedContents[i].Pin();

                    chat[i] = new()
                    {
                        role = (byte*)roleHandles[i].Pointer,
                        content = (byte*)contentHandles[i].Pointer
                    };
                }

                var buffer = new byte[Native.llama_n_ctx(context) * 8];
                var length = Native.llama_chat_apply_template(Native.llama_get_model(context), null, chat, (nuint)chat.Length, appendAssistant, buffer, buffer.Length);
                var text = encoding.GetString(buffer, 0, length);

                return text;
            }
            finally
            {
                for (var i = 0; i < messages.Count; i++)
                {
                    roleHandles[i].Dispose();
                    contentHandles[i].Dispose();
                }
            }
        }
    }
}
