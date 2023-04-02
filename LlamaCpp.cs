using System.Reflection;
using System.Runtime.InteropServices;

namespace LlamaCppDotNet
{
    public class LlamaCpp
    {
        private delegate string NextInputDelegate();
        private delegate void NextOutputDelegate(string token);

        [DllImport("main.dll")]
        private static extern void set_callbacks(NextInputDelegate ni, NextOutputDelegate no);

        [DllImport("main.dll")]
        private static extern int main(int argc, string[] argv);

        private string[] _args;

        public LlamaCpp(string[] args)
        {
            _args = args;
        }

        public void Run()
        {
            var argc = 1 + _args.Length;
            var argv = new[] { Path.GetFileName(Assembly.GetExecutingAssembly().Location) }.Concat(_args).ToArray();

            set_callbacks(
                () => Console.ReadLine() ?? String.Empty,
                Console.Write
            );

            _ = main(argc, argv);
        }
    }
}
