using System.Text;

namespace LlamaCppLib
{
    internal static class Extensions
    {
        public static StringBuilder AppendNewLineIfMissing(this StringBuilder sb)
        {
            if (sb.Length >= 1 && sb.ToString(sb.Length - 1, 1) == "\n")
                return sb;

            sb.Append("\n");
            return sb;
        }

        public static string AddNewLineIfMissing(this String value)
        {
            if (value.Last() == '\n')
                return value;

            return $"{value}\n";
        }

        public static void ClearAdd<T>(this IList<T> source, T item)
        {
            source.Clear();
            source.Add(item);
        }
    }
}
