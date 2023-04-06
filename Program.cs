using System.Diagnostics;
using System.Reflection;
using System.Text;

namespace LlamaCppDotNet
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            var random = new Random(DateTime.Now.Millisecond);

            var reversePrompts = new[]
            {
                "### Human:",
                "### Assistant:",
                "### Instruction:",
                "### Evaluation:",
                "### Explanation:",
            };

            args = new[]
            {
                $"--interactive-first",
                $"--instruct",
                reversePrompts.SelectMany(x => new[] { "--reverse-prompt", $"\"{x}\"" }).Aggregate((a, b) => $"{a} {b}"),
                $"--seed {random.Next(10)}",
                $"--threads 16",
                $"--prompt \"You are a helpful assistant and you answer questions truthfully and concisely.\"",
                $"--n_predict 512",
                $"--ctx_size 2048",
                //$"--ignore-eos",
                $"--temp 0",
                $"--model \"D:\\LLM_MODELS\\eachadea\\ggml-vicuna-13b-4bit\\ggml-vicuna-13b-4bit.bin\""
            };

            Console.WriteLine($"{args.Aggregate((a, b) => $"{a} {b}")}\n");

            var path = new[]
            {
                $@"main.exe",
                $@"x64\{Assembly.GetExecutingAssembly().GetConfiguration()}\main.exe",
            }.First(File.Exists);

            var cancellationTokenSource = new CancellationTokenSource();

            Console.CancelKeyPress += (s, e) =>
            {
                e.Cancel = true;
                cancellationTokenSource.Cancel();
            };

            using var process = new Process()
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = path,
                    Arguments = args.Aggregate((a, b) => $"{a} {b}"),
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardInput = true,
                }
            };

            using var pipe = new Pipe("_DOTNET_llama.cpp_fddd8d90-14ae-44f4-b3ae-3c5999a299fd");

            process.Start();
            await pipe.Wait();

            var tokens = new StringBuilder();

            while (!process.HasExited && !cancellationTokenSource.IsCancellationRequested)
            {
                await pipe.Transact(
                    async (streamId, token) => await (
                        streamId switch
                        {
                            // Input
                            0 => Task.Run(() => {
                                return Console.ReadLine() ?? String.Empty;
                            }),
                            // Output
                            1 => Task.Run(() => {
                                tokens.Append(token);
                                Console.Write(token);
                                return "!";
                            }),
                            _ => throw new InvalidDataException("Unexpected message.")
                        }
                    ),
                    cancellationTokenSource.Token
                );
            }

            if (process.HasExited)
            {
                Console.WriteLine($"\nUnexpected process termination.");
            }

            if (cancellationTokenSource.IsCancellationRequested)
            {
                Console.WriteLine($"\nTerminating process.");
                process.Kill(true);
            }

            await process.WaitForExitAsync();
            Console.WriteLine($"ExitCode = {process.ExitCode}");

            //var model = new LlamaCpp(args);
            //model.Run();

            //var builder = WebApplication.CreateBuilder(args);
            //var app = builder.Build();
            //app.MapGet("/", () => "Welcome to LLaMA!");
            //app.Run();
        }
    }
}
