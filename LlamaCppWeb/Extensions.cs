namespace LlamaCppWeb
{
    public static class Extensions
    {
        public static int ToInt32(this String value) => Int32.TryParse(value, out var result) ? result : 0;
        public static bool ToBool(this String value) => Boolean.TryParse(value, out var result) ? result : false;
    }
}
