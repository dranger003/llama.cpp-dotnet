using System.IO.Pipes;
using System.Text;

namespace LlamaCppDotNet
{
    public class Pipe : IDisposable
    {
        private bool _disposed;
        private NamedPipeServerStream _stream;

        public Pipe(string name)
        {
            _stream = new NamedPipeServerStream(name, PipeDirection.InOut, 1, PipeTransmissionMode.Message, PipeOptions.WriteThrough);
        }

        public async Task Wait()
        {
            await _stream.WaitForConnectionAsync();
        }

        public async Task Transact(Func<int, string, Task<string>> reply, CancellationToken cancellationToken)
        {
            try
            {
                var buffer = new byte[4096];
                await _stream.ReadExactlyAsync(buffer, 0, buffer.Length, cancellationToken);

                var message = Encoding.ASCII.GetString(buffer, 0, buffer.Length);
                var response = reply(message[0].ToInt32(), message.Substring(1));

                var data = Encoding.ASCII.GetBytes(await response);
                await _stream.WriteAsync(data, 0, data.Length, cancellationToken);
            }
            catch
            { }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Managed here
                    _stream.Dispose();
                }

                // Unmanaged here

                _disposed = true;
            }
        }

        // Override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~Pipe()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}
