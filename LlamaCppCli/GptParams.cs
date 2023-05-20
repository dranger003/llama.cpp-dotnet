namespace LlamaCppCli
{
    using LlamaToken = System.Int32;

    internal class GptParams
    {
        public int seed = -1;                                                   // RNG seed
        public int n_threads = Math.Min(4, Environment.ProcessorCount / 2);     // assumes hyperthreading (i.e. divide by 2)
        public int n_predict = -1;                                              // new tokens to predict
        public int n_ctx = 512;                                                 // context size
        public int n_batch = 512;                                               // batch size for prompt processing (must be >=32 to use BLAS)
        public int n_keep = 0;                                                  // number of tokens to keep from initial prompt
        public int n_gpu_layers = 0;                                            // number of layers to store in VRAM

        // sampling parameters
        public Dictionary<LlamaToken, float> logit_bias = new();                // logit bias for specific tokens
        public int top_k = 40;                                                  // <= 0 to use vocab size
        public float top_p = 0.95f;                                             // 1.0 = disabled
        public float tfs_z = 1.00f;                                             // 1.0 = disabled
        public float typical_p = 1.00f;                                         // 1.0 = disabled
        public float temp = 0.80f;                                              // 1.0 = disabled
        public float repeat_penalty = 1.10f;                                    // 1.0 = disabled
        public int repeat_last_n = 64;                                          // last n tokens to penalize (0 = disable penalty, -1 = context size)
        public float frequency_penalty = 0.00f;                                 // 0.0 = disabled
        public float presence_penalty = 0.00f;                                  // 0.0 = disabled
        public int mirostat = 0;                                                // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        public float mirostat_tau = 5.00f;                                      // target entropy
        public float mirostat_eta = 0.10f;                                      // learning rate

        public string model = "models/7B/ggml-model.bin";                       // model path
        public string prompt = String.Empty;
        public string path_prompt_cache = String.Empty;                         // path to file for saving/loading prompt eval state
        public string input_prefix = String.Empty;                              // string to prefix user inputs with
        public string input_suffix = String.Empty;                              // string to suffix user inputs with
        public List<string> antiprompt = new();                                 // string upon seeing which more user input is prompted

        public string? lora_adapter = default;                                  // lora adapter path
        public string? lora_base = default;                                     // base model path for the lora adapter

        public bool memory_f16 = true;                                          // use f16 instead of f32 for memory kv
        public bool random_prompt = false;                                      // do not randomize prompt if none provided
        public bool use_color = false;                                          // use color to distinguish generations and inputs
        public bool interactive = false;                                        // interactive mode
        public bool prompt_cache_all = false;                                   // save user input and generations to prompt cache

        public bool embedding = false;                                          // get only sentence embedding
        public bool interactive_first = false;                                  // wait for user input immediately
        public bool multiline_input = false;                                    // reverse the usage of `\`

        public bool instruct = false;                                           // instruction mode (used for Alpaca models)
        public bool penalize_nl = true;                                         // consider newlines as a repeatable token
        public bool perplexity = false;                                         // compute perplexity over the prompt
        public bool use_mmap = true;                                            // use mmap for faster loads
        public bool use_mlock = true;                                           // use mlock to keep model in memory (default false)
        public bool mem_test = false;                                           // compute maximum memory usage
        public bool verbose_prompt = false;                                     // print prompt tokens before generation

        public GptParams()
        { }

        public bool Parse(string[] args)
        {
            // TODO
            // Typically here we would map command line arguments to each field
            // For now we use the internal program values from the code

            seed = Program.Options.Seed ?? 0;
            n_threads = Program.Options.ThreadCount ?? n_threads;
            n_ctx = Program.Options.ContextSize ?? 512;
            top_k = Program.Options.TopK ?? top_k;
            top_p = Program.Options.TopP ?? top_p;
            temp = Program.Options.Temperature ?? temp;
            repeat_penalty = Program.Options.RepeatPenalty ?? repeat_penalty;

            model = args[0];
            use_mlock = false;

            return false;
        }
    }
}
