using System.Reflection;
using System.Runtime.InteropServices;

namespace LlamaCppDotNet
{
    public class LlamaCpp
    {
        private delegate string NextInputDelegate();
        private delegate void NextOutputDelegate(string token);

        [DllImport("main.dll", EntryPoint = "set_callbacks")]
        private static extern void SetCallbacks(NextInputDelegate ni, NextOutputDelegate no);

#pragma warning disable CS0028
        [DllImport("main.dll", EntryPoint = "main")]
        private static extern int Main(int argc, string[] argv);
#pragma warning restore CS0028

        private string[] _args;
        private LlamaCppOptions _options;

        public LlamaCpp(string[] args)
        {
            _args = args;
            _options = new LlamaCppOptions(args);
        }

        public void Run()
        {
            var argc = 1 + _args.Length;
            var argv = new[] { Path.GetFileName(Assembly.GetExecutingAssembly().Location) }.Concat(_args).ToArray();

            SetCallbacks(
                () => Console.ReadLine() ?? String.Empty,
                Console.Write
            );

            _ = Main(argc, argv);
        }
    }
}
