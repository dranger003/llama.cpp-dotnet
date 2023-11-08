namespace LlamaCppLib
{
    public class BlockingQueue<T>
    {
        private readonly Queue<T> _queue;
        private readonly ManualResetEvent _event = new(false);

        public BlockingQueue() => _queue = new Queue<T>();

        public void Enqueue(T item)
        {
            _queue.Enqueue(item);
            _event.Set();
        }

        public T Dequeue(CancellationToken? cancellationToken = default)
        {
            if (_queue.Count == 0)
                WaitForNext(cancellationToken);

            if (_queue.Count == 1)
                _event.Reset();

            return _queue.Dequeue();
        }

        public bool Any() => _queue.Count > 0;

        public void WaitForNext(CancellationToken? cancellationToken = default) => WaitHandle.WaitAny(new[] { _event, (cancellationToken ?? new()).WaitHandle });
    }
}
