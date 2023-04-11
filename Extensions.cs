using System.Reflection;
using System.Text;

namespace LlamaCppDotNet
{
    public static class Extensions
    {
        public static string? GetConfiguration(this Assembly assembly) => assembly.GetCustomAttribute<AssemblyConfigurationAttribute>()?.Configuration;

        public static int ToInt32(this String s) => Int32.TryParse(s, out var v) ? v : 0;

        public static int ToInt32(this Char c) => Int32.TryParse($"{c}", out var v) ? v : 0;

        public static string AppendNewLineIfMissing(this String s) => s.EndsWith("\n") ? s : $"{s}";

        public static StringBuilder AppendNewLineIfMissing(this StringBuilder sb) => sb.ToString().EndsWith("\n") ? sb : sb.Append("\n");
    }
}
