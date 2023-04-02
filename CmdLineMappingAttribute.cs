using System.Text;

namespace LlamaCppDotNet
{
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false)]
    public class CmdLineMappingAttribute : Attribute
    {
        public string[] Aliases { get; set; }

        public CmdLineMappingAttribute(params string[] aliases)
        {
            Aliases = aliases;
        }

        public static string[] GenerateAliases(string propertyName)
        {
            return new[]
            {
                GenerateShortAlias(propertyName),
                GenerateLongAliases(propertyName, "_"),
                GenerateLongAliases(propertyName, "-"),
            };
        }

        private static string GenerateShortAlias(string propertyName) => $"-{propertyName.Substring(0, 1).ToLower()}";

        private static string GenerateLongAliases(string input, string separator)
        {
            var result = new StringBuilder();

            for (int i = 0; i < input.Length; i++)
            {
                if (Char.IsUpper(input[i]) && i != 0)
                    result.Append(separator);

                result.Append(input[i].ToString().ToLower());
            }

            return $"--{result}";
        }
    }
}
