using System.Text;

namespace LlamaCppWeb
{
    public class LlamaContext
    {
        public string? InitialContext { get; set; }
        public StringBuilder Context { get; } = new();

        public void ResetContext()
        {
            Context.Clear();
            Context.Append(InitialContext ?? String.Empty);
        }
    }
}
