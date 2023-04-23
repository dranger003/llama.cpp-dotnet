namespace LlamaCppWeb
{
    public class QueryList<T> : List<T> where T : class
    {
        public QueryList(T[] list) => AddRange(list);

        public static bool TryParse(string input, out QueryList<T> result)
        {
            result = new(
                input
                    .Split(',')
                    .Cast<T>()
                    .ToArray()
            );

            return true;
        }
    }
}
