using System.Reflection;
using System.Runtime.InteropServices;

namespace LlamaCppDotNet
{
    public class LlamaCpp
    {
        private delegate string NextInputDelegate();
        private delegate void NextOutputDelegate(string token);

        [DllImport("main.dll", EntryPoint = "DOTNET_set_callbacks")]
        private static extern void SetCallbacks(NextInputDelegate ni, NextOutputDelegate no);

        [DllImport("main.dll", EntryPoint = "DOTNET_set_interacting")]
        private static extern void SetInteracting(bool interacting);

        [DllImport("main.dll", EntryPoint = "DOTNET_main")]
        private static extern int MainOverride(int argc, string[] argv);

        private string[] _args;
        private LlamaCppOptions _options;

        public LlamaCpp(string[] args)
        {
            _args = args;
            _options = new LlamaCppOptions(args);
        }

        public void Run()
        {
            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                SetInteracting(true);
                Console.WriteLine();
            };

            var argc = 1 + _args.Length;
            var argv = new[] { Path.GetFileName(Assembly.GetExecutingAssembly().Location) }.Concat(_args).ToArray();

            SetCallbacks(NextInput, NextOutput);
            _ = MainOverride(argc, argv);
        }

        public void Reload(LlamaCppOptions options)
        {
        }

        private string NextInput()
        {
            while (true)
            {
                var input = Console.ReadLine() ?? String.Empty;

                if (input.ToUpper() == "QUIT")
                    return String.Empty;

                if (!String.IsNullOrWhiteSpace(input))
                    return input;
            }
        }

        private void NextOutput(string token)
        {
            Console.Write(token);
        }
    }
}
