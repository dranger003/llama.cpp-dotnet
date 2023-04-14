using System.Text;

namespace LlamaCppLib
{
    internal static class Extensions
    {
        public static StringBuilder AppendNewLineIfMissing(this StringBuilder sb) => sb.ToString().EndsWith("\n") ? sb : sb.Append("\n");
    }
}
