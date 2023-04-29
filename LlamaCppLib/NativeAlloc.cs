using System.Runtime.InteropServices;

namespace LlamaCppLib
{
    internal class NativeAlloc : IDisposable
    {
        private nint _ptr;
        private int _size;
        private Action<nint> _deallocator;

        public NativeAlloc(Func<int, nint> allocator, Action<nint> deallocator, int size)
        {
            _size = size;
            _deallocator = deallocator;
            _ptr = allocator(size);
        }

        public void Dispose()
        {
            _deallocator(_ptr);
            _ptr = nint.Zero;
        }

        public nint Ptr { get => _ptr; }
        public int Size { get => _size; }
    }

    internal class NativeHGlobal : NativeAlloc
    {
        public NativeHGlobal(int size) : base(Marshal.AllocHGlobal, Marshal.FreeHGlobal, size)
        { }
    }

    internal class NativeCoTaskMem : NativeAlloc
    {
        public NativeCoTaskMem(int size) : base(Marshal.AllocCoTaskMem, Marshal.FreeCoTaskMem, size)
        { }
    }
}
