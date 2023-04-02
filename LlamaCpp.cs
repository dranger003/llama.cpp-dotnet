using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCppDotNet
{
    public class LlamaCpp
    {
        private delegate string NextInputDelegate();
        private delegate void NextOutputDelegate(string token);

        [DllImport("main.dll", EntryPoint = "DOTNET_set_interacting")]
        private static extern void SetInteracting(bool interacting);

        [DllImport("main.dll", EntryPoint = "DOTNET_main")]
        private static extern int MainOverride(int argc, string[] argv, NextInputDelegate ni, NextOutputDelegate no);

        private bool _running = false;
        private string[] _args;
        private LlamaCppOptions _options;

        public LlamaCpp(string[] args)
        {
            _args = args;
            _options = new LlamaCppOptions(args);
        }

        public void Run()
        {
            if (_running)
                throw new Exception("Model is already running.");

            _running = true;
            Console.CancelKeyPress += CancelKeyPressHandler;
            {
                var argc = 1 + _args.Length;
                var argv = new[] { Path.GetFileName(Assembly.GetExecutingAssembly().Location) }.Concat(_args).ToArray();

                _ = MainOverride(argc, argv, NextInput, NextOutput);
            }
            Console.CancelKeyPress -= CancelKeyPressHandler;
            _running = false;
        }

        private void CancelKeyPressHandler(object? sender, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;
            SetInteracting(true);
            Console.WriteLine();
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
