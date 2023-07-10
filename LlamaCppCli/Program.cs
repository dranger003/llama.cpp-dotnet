using System.Net.Http.Headers;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Web;

using LlamaCppLib;

namespace LlamaCppCli
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
#if DEBUG
            //args = new[] { "0", @"C:\LLM_MODELS\WizardLM\ggml-wizardlm-v1.1-13b-q8_0.bin", "60", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n{0}\n\nASSISTANT:\n" };
            //args = new[] { "0", @"C:\LLM_MODELS\WizardLM\wizardlm-30b.ggmlv3.q4_K_M.bin", "60", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n{0}\n\nASSISTANT:\n" };
            //args = new[] { "0", @"C:\LLM_MODELS\psmathur\ggml-orca-mini-v2-13b-q8_0.bin", "60", "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:\n{0}\n\n### Response:\n" };

            args = new[] { "1", "http://localhost:5021", "wizardlm-v1.1-13b-q8_0", "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER:\n{0}\n\nASSISTANT:\n" };
#endif
            var samples = new (string Name, Func<string[], Task> Func)[]
            {
                (nameof(LocalSample), LocalSample),     // Run locally
                (nameof(RemoteSample), RemoteSample),   // Run via API
            }
                .Select((sample, index) => (sample, index))
                .ToDictionary(k => k.sample.Name, v => (Index: v.index, v.sample.Func));

            var PrintAvailableSamples = () =>
            {
                Console.WriteLine($"Available sample(s):");
                foreach (var sample in samples)
                    Console.WriteLine($"    [{sample.Value.Index}] = {sample.Key}");
            };

            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} <SampleIndex> <SampleArgs>");
                PrintAvailableSamples();
                return;
            }

            var sampleIndex = Int32.Parse(args[0]);
            var sampleName = samples.SingleOrDefault(sample => sample.Value.Index == sampleIndex).Key;

            if (sampleName == default)
            {
                Console.WriteLine($"Sample not found ({sampleIndex}).");
                PrintAvailableSamples();
                return;
            }

            await samples[sampleName].Func(args.Skip(1).ToArray());
        }

        static async Task LocalSample(string[] args)
        {
            if (args.Length < 1)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} model_path [gpu_layers] [template]");
                return;
            }

            var modelPath = args[0];
            var gpuLayers = Int32.Parse(args[1]);
            var template = args.Length > 2 ? args[2] : "{0}";

            var modelOptions = new LlamaCppModelOptions
            {
                ContextSize = 2048,
                GpuLayers = gpuLayers,
                Template = template,
            };

            using var model = new LlamaCpp();
            model.Load(modelPath, modelOptions);

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            var predictOptions = new LlamaCppPredictOptions
            {
                ThreadCount = 4,
                TopK = 40,
                TopP = 0.95f,
                Temperature = 0.1f,
                RepeatPenalty = 1.1f,
                PenalizeNewLine = false,
                Mirostat = Mirostat.Mirostat2,
                MirostatTAU = 5.0f,
                MirostatETA = 0.1f,
                ResetState = true,
            };

            await Console.Out.WriteLineAsync(
                """

                Entering interactive mode:
                    * Press <Ctrl+C> to cancel running predictions
                    * Press <Enter> on an empty input prompt to quit
                """
            );

            while (true)
            {
                await Console.Out.WriteLineAsync("\nInput:");

                var prompt = await Console.In.ReadLineAsync() ?? String.Empty;
                if (String.IsNullOrWhiteSpace(prompt))
                    break;

                await Console.Out.WriteLineAsync("\nOutput:");

                predictOptions.Prompt = String.Format(modelOptions.Template, prompt);

                await foreach (var prediction in model.Predict(predictOptions, cancellationTokenSource.Token))
                    await Console.Out.WriteAsync(prediction.Value);

                await Console.Out.WriteLineAsync();
            }

            await Console.Out.WriteLineAsync("Quitting...");
        }

        static async Task RemoteSample(string[] args)
        {
            if (args.Length < 2)
            {
                await Console.Out.WriteLineAsync($"Usage: {Path.GetFileName(Assembly.GetExecutingAssembly().Location)} base_url model_name [template]");
                return;
            }

            var baseUrl = args[0];
            var modelName = args[1];
            var template = args.Length > 2 ? args[2] : "{0}";

            var modelOptions = new LlamaCppModelOptions() { ContextSize = 2048, GpuLayers = 60, Template = template };

            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => cancellationTokenSource.Cancel(!(e.Cancel = true));

            using var httpClient = new HttpClient();

            //// List model(s)
            //{
            //    var response = await httpClient.GetAsync($"{baseUrl}/model/list");
            //    response.EnsureSuccessStatusCode();
            //    await Console.Out.WriteLineAsync(await response.Content.ReadAsStringAsync());
            //}

            // Load model
            {
                await Console.Out.WriteAsync("Loading model...");
                var response = await httpClient.GetAsync($"{baseUrl}/model/load?{nameof(modelName)}={modelName}&{nameof(modelOptions)}={HttpUtility.UrlEncode(JsonSerializer.Serialize(modelOptions))}");
                response.EnsureSuccessStatusCode();
                await Console.Out.WriteLineAsync(" OK.");
            }

            //// Model status
            //{
            //    var response = await httpClient.GetAsync($"{baseUrl}/model/status");
            //    response.EnsureSuccessStatusCode();
            //    await Console.Out.WriteLineAsync(await response.Content.ReadAsStringAsync());
            //}

            // Run prediction(s)
            {
                var predictOptions = new LlamaCppPredictOptions() { ResetState = true, Mirostat = Mirostat.Mirostat2 };

                await Console.Out.WriteLineAsync(
                    """

                    Entering interactive mode:
                        * Press <Ctrl+C> to cancel running predictions
                        * Press <Enter> on an empty input prompt to quit
                    """
                );

                while (true)
                {
                    await Console.Out.WriteLineAsync("\nInput:");

                    var prompt = await Console.In.ReadLineAsync() ?? String.Empty;
                    if (String.IsNullOrWhiteSpace(prompt))
                        break;

                    await Console.Out.WriteLineAsync("\nOutput:");

                    predictOptions.Prompt = String.Format(modelOptions.Template, prompt);

                    using var request = new HttpRequestMessage(HttpMethod.Get, $"{baseUrl}/model/predict?{nameof(predictOptions)}={HttpUtility.UrlEncode(JsonSerializer.Serialize(predictOptions))}");
                    request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

                    var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationTokenSource.Token);
                    response.EnsureSuccessStatusCode();

                    await using var stream = await response.Content.ReadAsStreamAsync(cancellationTokenSource.Token);
                    using var reader = new StreamReader(stream);

                    while (!reader.EndOfStream && !cancellationTokenSource.IsCancellationRequested)
                    {
                        var data = await reader.ReadLineAsync(cancellationTokenSource.Token);
                        if (data == null)
                            break;

                        var token = Regex.Match(data, @"(?<=data:\s).*").Value.Replace("\\n", "\n");
                        await Console.Out.WriteAsync(token);
                    }

                    await Console.Out.WriteLineAsync();
                }
            }
        }
    }
}
