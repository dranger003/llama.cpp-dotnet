using System.Reflection;
using System.Text.RegularExpressions;

namespace LlamaCppDotNet
{
    // options:
    //   -h, --help            show this help message and exit
    //   -i, --interactive     run in interactive mode
    //   --interactive-first   run in interactive mode and wait for input right away
    //   -ins, --instruct      run in instruction mode (use with Alpaca models)
    //   -r PROMPT, --reverse-prompt PROMPT
    //                         run in interactive mode and poll user input upon seeing PROMPT (can be
    //                         specified more than once for multiple prompts).
    //   --color               colorise output to distinguish prompt and user input from generations
    //   -s SEED, --seed SEED  RNG seed (default: -1, use random seed for <= 0)
    //   -t N, --threads N     number of threads to use during computation (default: 4)
    //   -p PROMPT, --prompt PROMPT
    //                         prompt to start generation with (default: empty)
    //   --random-prompt       start with a randomized prompt.
    //   --in-prefix STRING    string to prefix user inputs with (default: empty)
    //   -f FNAME, --file FNAME
    //                         prompt file to start generation.
    //   -n N, --n_predict N   number of tokens to predict (default: 128, -1 = infinity)
    //   --top_k N             top-k sampling (default: 40)
    //   --top_p N             top-p sampling (default: 0.9)
    //   --repeat_last_n N     last n tokens to consider for penalize (default: 64)
    //   --repeat_penalty N    penalize repeat sequence of tokens (default: 1.1)
    //   -c N, --ctx_size N    size of the prompt context (default: 512)
    //   --ignore-eos          ignore end of stream token and continue generating
    //   --memory_f32          use f32 instead of f16 for memory key+value
    //   --temp N              temperature (default: 0.8)
    //   --n_parts N           number of model parts (default: -1 = determine from dimensions)
    //   -b N, --batch_size N  batch size for prompt processing (default: 8)
    //   --perplexity          compute perplexity over the prompt
    //   --keep                number of tokens to keep from the initial prompt (default: 0, -1 = all)
    //   --mtest               compute maximum memory usage
    //   --verbose-prompt      print prompt before generation
    //   -m FNAME, --model FNAME
    //                         model path (default: models/llama-7B/ggml-model.bin)

    public class LlamaCppOptions
    {
        private List<string> _args = new();

        [CmdLineMapping] public bool Help { get; set; }
        [CmdLineMapping] public bool Interactive { get; set; }
        [CmdLineMapping] public bool InteractiveFirst { get; set; }
        [CmdLineMapping] public bool InstructionMode { get; set; }
        [CmdLineMapping] public string? ReversePrompt { get; set; }
        [CmdLineMapping] public bool Color { get; set; }
        [CmdLineMapping] public int Seed { get; set; }
        [CmdLineMapping] public int Threads { get; set; }
        [CmdLineMapping] public string? Prompt { get; set; }
        [CmdLineMapping] public bool RandomPrompt { get; set; }
        [CmdLineMapping] public string? InPrefix { get; set; }
        [CmdLineMapping] public string? PromptFileName { get; set; }
        [CmdLineMapping] public int NPredict { get; set; }
        [CmdLineMapping] public int TopK { get; set; }
        [CmdLineMapping] public double TopP { get; set; }
        [CmdLineMapping] public int RepeatLastN { get; set; }
        [CmdLineMapping] public double RepeatPenalty { get; set; }
        [CmdLineMapping] public int CtxSize { get; set; }
        [CmdLineMapping] public bool IgnoreEos { get; set; }
        [CmdLineMapping] public bool MemoryF32 { get; set; }
        [CmdLineMapping] public double Temp { get; set; }
        [CmdLineMapping] public int NParts { get; set; }
        [CmdLineMapping] public int BatchSize { get; set; }
        [CmdLineMapping] public bool Perplexity { get; set; }
        [CmdLineMapping] public int Keep { get; set; }
        [CmdLineMapping] public bool MTest { get; set; }
        [CmdLineMapping] public bool VerbosePrompt { get; set; }
        [CmdLineMapping] public string? Model { get; set; }

        public LlamaCppOptions()
        { }

        public LlamaCppOptions(string[] args)
        {
            _args.AddRange(args);
            ValidateAndMap(args);
        }

        private void ValidateAndMap(string[] args)
        {
            var optionProperties = typeof(LlamaCppOptions)
                .GetProperties()
                .Where(prop => prop.GetCustomAttributes<CmdLineMappingAttribute>().Any());

            for (int i = 0; i < args.Length; i++)
            {
                var arg = args[i];

                var property = optionProperties.FirstOrDefault(
                    prop =>
                    {
                        var attribute = prop.GetCustomAttribute<CmdLineMappingAttribute>();
                        if (attribute == null)
                            return false;

                        if (!attribute.Aliases.Any())
                            attribute.Aliases = CmdLineMappingAttribute.GenerateAliases(prop.Name);

                        if (!attribute.Aliases.Contains(arg))
                            return false;

                        return true;
                    }
                );

                if (property == null)
                    throw new CmdLineValidationException($"Unknown argument: {arg}");

                if (i + 1 >= args.Length && property.PropertyType != typeof(bool))
                    throw new CmdLineValidationException($"Missing value for argument: {arg}");

                if (property.PropertyType == typeof(bool))
                {
                    property.SetValue(this, true);
                }
                else if (property.PropertyType == typeof(double))
                {
                    if (Double.TryParse(args[++i], out double value))
                        property.SetValue(this, value);
                    else
                        throw new CmdLineValidationException($"Invalid value for argument {arg}: {args[i]}");
                }
                else if (property.PropertyType == typeof(string))
                {
                    property.SetValue(this, args[++i]);
                }
                else if (property.PropertyType == typeof(int))
                {
                    if (Int32.TryParse(args[++i], out int value))
                        property.SetValue(this, value);
                    else
                        throw new CmdLineValidationException($"Invalid value for argument {arg}: {args[i]}");
                }
                else
                {
                    throw new CmdLineValidationException($"Unsupported property type: {property.PropertyType}");
                }
            }
        }

        public override string ToString() => _args
            .Select(arg => Regex.IsMatch(arg, @"(?:\s+|:)") ? $"\"{arg}\"" : arg)
            .Aggregate((a, b) => $"{a} {b}");
    }
}
