using System.Text;

namespace LlamaCppCli
{
    internal static class Extensions
    {
        public static void ClearAdd<T>(this IList<T> source, T item)
        {
            source.Clear();
            source.Add(item);
        }

        public static bool EndsWith(this StringBuilder sb, string value) => sb.Length >= value.Length && sb.ToString(sb.Length - value.Length, value.Length) == value;
    }
}
