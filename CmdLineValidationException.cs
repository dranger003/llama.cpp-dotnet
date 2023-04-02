namespace LlamaCppDotNet
{
    public class CmdLineValidationException : Exception
    {
        public CmdLineValidationException(string message) : base(message) { }
    }
}
