namespace LlamaCppWeb
{
    public class StringList : List<string>
    {
        public StringList(string[] list) => AddRange(list);

        public static bool TryParse(string input, out StringList result)
        {
            result = new(input.Split(','));
            return true;
        }
    }
}
