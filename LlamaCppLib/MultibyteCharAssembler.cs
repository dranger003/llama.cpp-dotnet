using System.Text;

namespace LlamaCppLib
{
    public class MultibyteCharAssembler
    {
        private List<byte> _buffer = new();

        public string Consume(Span<byte> bytes)
        {
            var result = new StringBuilder();

            _buffer.AddRange(bytes.ToArray());
            while (_buffer.Count > 0)
            {
                var validUtf8Length = _FindValidUtf8SequenceLength(_buffer.ToArray());
                if (validUtf8Length == 0)
                    break;

                result.Append(Encoding.UTF8.GetString(_buffer.GetRange(0, validUtf8Length).ToArray()));
                _buffer.RemoveRange(0, validUtf8Length);
            }

            return result.ToString();
        }

        public string Consume()
        {
            if (_buffer.Count == 0)
                return String.Empty;

            var result = Encoding.UTF8.GetString(_buffer.ToArray());
            _buffer.Clear();
            return result;
        }

        private int _FindValidUtf8SequenceLength(byte[] bytes)
        {
            var index = 0;
            while (index < bytes.Length)
            {
                var byteCount = _Utf8ByteCount(bytes[index]);
                if (index + byteCount > bytes.Length)
                    break;

                index += byteCount;
            }

            return index;
        }

        private int _Utf8ByteCount(byte b)
        {
            return b switch
            {
                _ when (b & 0x80) == 0x00 => 1,     // 1-byte character
                _ when (b & 0xE0) == 0xC0 => 2,     // 2-byte character
                _ when (b & 0xF0) == 0xE0 => 3,     // 3-byte character
                _ when (b & 0xF8) == 0xF0 => 4,     // 4-byte character
                _ => 0                              // UTF-8 start byte invalid
            };
        }
    }
}
