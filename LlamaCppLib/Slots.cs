using System.Collections;

namespace LlamaCppLib
{
    public class Slots<T> : IEnumerable<T>
    {
        private readonly Dictionary<int, T> _items = new();
        private readonly Queue<int> _ids;

        public Slots(int capacity)
        {
            _ids = new Queue<int>(capacity);

            for (var i = 0; i < capacity; i++)
                _ids.Enqueue(i);
        }

        public bool HasFreeSlot => _ids.Count > 0;

        public int Add(T item)
        {
            if (_ids.Count == 0)
                throw new InvalidOperationException($"No free slots available.");

            var id = _ids.Dequeue();
            _items[id] = item;

            return id;
        }

        public void Remove(int id)
        {
            if (!_items.ContainsKey(id))
                throw new KeyNotFoundException($"Item ID \"{id}\" not found.");

            _items.Remove(id);
            _ids.Enqueue(id);
        }

        public void RemoveAll(Func<T, bool> predicate)
        {
            var ids = new List<int>();

            foreach (var item in _items)
            {
                if (predicate(item.Value))
                    ids.Add(item.Key);
            }

            foreach (var id in ids)
            {
                _items.Remove(id);
                _ids.Enqueue(id);
            }
        }

        // IEnumerable<T>
        public IEnumerator<T> GetEnumerator() => _items.Values.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
