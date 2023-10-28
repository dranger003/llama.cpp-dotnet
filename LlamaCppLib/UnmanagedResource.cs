namespace LlamaCppLib
{
    internal class UnmanagedResource<T> : IDisposable
    {
        protected Action<T>? _dealloc;
        protected T? _handle;

        public void Dispose()
        {
            if (EqualityComparer<T>.Default.Equals(_handle, default) || _handle == null)
                return;

            _dealloc?.Invoke(_handle);
            _handle = default;
        }

        public bool Created => !EqualityComparer<T>.Default.Equals(_handle, default);
        public T Handle => EqualityComparer<T>.Default.Equals(_handle, default) || _handle == null ? throw new NullReferenceException() : _handle;

        public T Create(Func<T> alloc, Action<T> dealloc)
        {
            _handle = alloc();
            _dealloc = dealloc;

            return _handle;
        }

        public void GetResource(out T? resource) => resource = _handle;
    }

    internal class UnmanagedResource : UnmanagedResource<bool>
    {
        public void Create(Action alloc, Action dealloc)
        {
            try
            {
                alloc();
                _handle = true;
            }
            catch
            {
                _handle = false;
            }

            _dealloc = _ => dealloc();
        }
    }
}
