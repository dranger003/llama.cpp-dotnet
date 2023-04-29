using System.Runtime.CompilerServices;

using LlamaCppLib;

namespace LlamaCppCli
{
    using LlamaToken = System.Int32;

    internal static class ExampleMain
    {
        // TODO: INCOMPLETE/WIP
        // Complete port of llama.cpp/example/main
        // https://github.com/ggerganov/llama.cpp/tree/master/examples/main

        public static async Task Run(string[] args)
        {
            var is_interacting = false;

            var aparams = new gpt_params();
            aparams.model = "models/llama-7B/ggml-model.bin";
            aparams.parse(args);

            if (aparams.n_ctx > 2048)
                await Console.Error.WriteLineAsync($"{FuncName()}: warning: model does not support context sizes greater than 2048 tokens ({aparams.n_ctx} specified); expect poor results");

            if (aparams.seed <= 0)
            {
                aparams.seed = (int)DateTimeOffset.UtcNow.ToUnixTimeSeconds();
            }

            await Console.Error.WriteLineAsync($"{FuncName()}: seed = {aparams.seed}");

            var rng = new Random(aparams.seed);
            if (aparams.random_prompt)
                aparams.prompt = gpt_random_prompt(rng);

            nint ctx = nint.Zero;

            // load the model
            {
                var lparams = LlamaCppInterop.llama_context_default_params();
                lparams.n_ctx = aparams.n_ctx;
                lparams.n_parts = aparams.n_parts;
                lparams.seed = aparams.seed;
                lparams.f16_kv = aparams.memory_f16;
                lparams.use_mmap = aparams.use_mmap;
                lparams.use_mlock = aparams.use_mlock;

                ctx = LlamaCppInterop.llama_init_from_file(aparams.model, lparams);

                if (ctx == nint.Zero)
                {
                    await Console.Error.WriteLineAsync($"{FuncName()}: error: failed to load model '{aparams.model}'");
                    return;
                }
            }

            if (!String.IsNullOrWhiteSpace(aparams.lora_adapter))
            {
                var err = LlamaCppInterop.llama_apply_lora_from_file(ctx, aparams.lora_adapter, aparams.lora_base, aparams.n_threads);
                if (err != 0)
                {
                    await Console.Error.WriteLineAsync($"{FuncName()}: error: failed to apply lora adapter");
                    return;
                }
            }

            // print system information
            {
                await Console.Error.WriteLineAsync($"\n");
                await Console.Error.WriteLineAsync($"system_info: n_threads = {aparams.n_threads} / {Environment.ProcessorCount / 2} | {LlamaCppInterop.llama_print_system_info()}");
            }

            // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
            // uncomment the "used_mem" line in llama.cpp to see the results
            if (aparams.mem_test)
            {
                {
                    var tmp = new List<LlamaToken>(aparams.n_batch);
                    LlamaCppInterop.llama_eval(ctx, tmp, 0, aparams.n_threads);
                }
                {
                    var tmp = new List<LlamaToken>(aparams.n_batch);
                    LlamaCppInterop.llama_eval(ctx, tmp, aparams.n_predict - 1, aparams.n_threads);
                }

                LlamaCppInterop.llama_print_timings(ctx);
                LlamaCppInterop.llama_free(ctx);

                return;
            }

            // Add a space in front of the first character to match OG llama tokenizer behavior
            aparams.prompt = $" {aparams.prompt}";

            var path_session = aparams.path_session;
            var session_tokens = new List<LlamaToken>();

            if (!String.IsNullOrWhiteSpace(path_session))
            {
                await Console.Error.WriteLineAsync($"{FuncName()}: attempting to load saved session from {path_session}..");

                // REVIEW - fopen to check for existing session
                if (File.Exists(path_session))
                {
                    var n_session_bytes = LlamaCppInterop.llama_load_session_file(ctx, path_session, session_tokens);

                    if (n_session_bytes > 0)
                        await Console.Error.WriteLineAsync($"{FuncName()}: loaded {n_session_bytes} bytes of session data!");
                    else
                        await Console.Error.WriteLineAsync($"{FuncName()}: could not load session file, will recreate");
                }
                else
                {
                    await Console.Error.WriteLineAsync($"{FuncName()}: session file does not exist, will create");
                }
            }

            // tokenize the prompt
            var embd_inp = LlamaCppInterop.llama_tokenize(ctx, aparams.prompt, true);

            var n_ctx = LlamaCppInterop.llama_n_ctx(ctx);

            if (embd_inp.Count > n_ctx - 4)
            {
                await Console.Error.WriteLineAsync($"{FuncName()}: error: prompt is too long ({embd_inp.Count} tokens, max {n_ctx - 4})");
                return;
            }

            // debug message about similarity of saved session, if applicable
            var n_matching_session_tokens = 0;
            if (session_tokens.Any())
            {
                foreach (var id in session_tokens)
                {
                    if (n_matching_session_tokens >= embd_inp.Count || id != embd_inp[n_matching_session_tokens])
                        break;
                    n_matching_session_tokens++;
                }

                if (n_matching_session_tokens >= embd_inp.Count)
                    await Console.Error.WriteLineAsync($"{FuncName()}: session file has exact match for prompt!");
                else if (n_matching_session_tokens < (embd_inp.Count / 2))
                    await Console.Error.WriteLineAsync($"{FuncName()}: warning: session file has low similarity to prompt ({n_matching_session_tokens} / {embd_inp.Count} tokens); will mostly be reevaluated");
                else
                    await Console.Error.WriteLineAsync($"{FuncName()}: session file matches {n_matching_session_tokens} / {embd_inp.Count} tokens of prompt");
            }

            // number of tokens to keep when resetting context
            if (aparams.n_keep < 0 || aparams.n_keep > embd_inp.Count || aparams.instruct)
                aparams.n_keep = embd_inp.Count;

            // prefix & suffix for instruct mode
            var inp_pfx = LlamaCppInterop.llama_tokenize(ctx, "\n\n### Instruction:\n\n", true);
            var inp_sfx = LlamaCppInterop.llama_tokenize(ctx, "\n\n### Response:\n\n", false);

            // in instruct mode, we inject a prefix and a suffix to each input by the user
            if (aparams.instruct)
            {
                aparams.interactive_first = true;
                aparams.antiprompt.Add("### Instruction:\n\n");
            }

            // enable interactive mode if reverse prompt or interactive start is specified
            if (aparams.antiprompt.Count != 0 || aparams.interactive_first)
                aparams.interactive = true;

            if (aparams.antiprompt.Any())
            {
                foreach (var antiprompt in aparams.antiprompt)
                    await Console.Error.WriteLineAsync($"Reverse prompt: '{antiprompt}'");
            }

            // determine newline token
            var llama_token_newline = LlamaCppInterop.llama_tokenize(ctx, "\n", false);

            if (aparams.verbose_prompt)
            {
                await Console.Error.WriteLineAsync();
                await Console.Error.WriteLineAsync($"{FuncName()}: prompt: '{aparams.prompt}'");
                await Console.Error.WriteLineAsync($"{FuncName()}: number of tokens in prompt = {embd_inp.Count}");
                for (var i = 0; i < embd_inp.Count; i++)
                    await Console.Error.WriteLineAsync($"{embd_inp[i]} -> '{LlamaCppInterop.llama_token_to_str(ctx, embd_inp[i])}'");

                if (aparams.n_keep > 0)
                {
                    await Console.Error.WriteLineAsync($"{FuncName()}: static prompt based on n_keep: '");
                    for (var i = 0; i < aparams.n_keep; i++)
                        await Console.Error.WriteLineAsync($"{LlamaCppInterop.llama_token_to_str(ctx, embd_inp[i])}");
                    await Console.Error.WriteLineAsync($"'");
                }
                await Console.Error.WriteLineAsync();
            }

            if (aparams.interactive)
            {
                Console.CancelKeyPress += (s, e) =>
                {
                    Console.WriteLine();
                    if (!is_interacting)
                        is_interacting = true;
                    else
                        LlamaCppInterop.llama_print_timings(ctx);
                };

                await Console.Error.WriteLineAsync($"{FuncName()}: interactive mode on.");

                if (aparams.antiprompt.Any())
                {
                    foreach (var antiprompt in aparams.antiprompt)
                        await Console.Error.WriteLineAsync($"Reverse prompt: '{antiprompt}'");
                }

                if (!String.IsNullOrWhiteSpace(aparams.input_prefix))
                    await Console.Error.WriteLineAsync($"Input prefix: '{aparams.input_prefix}'");
            }

            await Console.Error.WriteLineAsync($"sampling: repeat_last_n = {aparams.repeat_last_n}, repeat_penalty = {aparams.repeat_penalty}, presence_penalty = {aparams.presence_penalty}, frequency_penalty = {aparams.frequency_penalty}, top_k = {aparams.top_k}, tfs_z = {aparams.tfs_z}, top_p = {aparams.top_p}, typical_p = {aparams.typical_p}, temp = {aparams.temp}, mirostat = {aparams.mirostat}, mirostat_lr = {aparams.mirostat_eta}, mirostat_ent = {aparams.mirostat_tau}");
            await Console.Error.WriteLineAsync($"generate: n_ctx = {n_ctx}, n_batch = {aparams.n_batch}, n_predict = {aparams.n_predict}, n_keep = {aparams.n_keep}");
            await Console.Error.WriteLineAsync($"\n");

            // TODO: replace with ring-buffer
            var last_n_tokens = new List<int>(n_ctx);

            if (aparams.interactive)
            {
                await Console.Error.WriteLineAsync(
                    $"== Running in interactive mode. ==\n" +
                    $" - Press Ctrl+C to interject at any time.\n" +
                    $" - Press Return to return control to LLaMa.\n" +
                    $" - If you want to submit another line, end your input in '\\'.\n"
                );

                is_interacting = aparams.interactive_first;
            }

            var is_antiprompt = false;
            var input_noecho = false;

            // HACK - because session saving incurs a non-negligible delay, for now skip re-saving session
            // if we loaded a session with at least 75% similarity. It's currently just used to speed up the
            // initial prompt so it doesn't need to be an exact match.
            var need_to_save_session = !String.IsNullOrWhiteSpace(path_session) && n_matching_session_tokens < (embd_inp.Count * 3 / 4);

            var n_past = 0;
            var n_remain = aparams.n_predict;
            var n_consumed = 0;
            var n_session_consumed = 0;

            var embd = new List<LlamaToken>();

            while (n_remain != 0 || aparams.interactive)
            {
                // predict
                if (embd.Count > 0)
                {
                    // infinite text generation via context swapping
                    // if we run out of context:
                    // - take the n_keep first tokens from the original prompt (via n_past)
                    // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                    if (n_past + embd.Count > n_ctx)
                    {
                        var n_left = n_past - aparams.n_keep;

                        n_past = aparams.n_keep;

                        // insert n_left/2 tokens at the start of embd from last_n_tokens
                        //embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left/2 - embd.size(), last_n_tokens.end() - embd.size());

                        // REVIEW - stop saving session if we run out of context
                        path_session = String.Empty;

                        //printf("\n---\n");
                        //printf("resetting: '");
                        //for (int i = 0; i < (int) embd.size(); i++) {
                        //    printf("%s", llama_token_to_str(ctx, embd[i]));
                        //}
                        //printf("'\n");
                        //printf("\n---\n");
                    }

                    // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
                    // REVIEW
                    if (n_session_consumed < session_tokens.Count)
                    {
                        var i = 0;
                        for (; i < embd.Count; i++)
                        {
                            if (embd[i] != session_tokens[n_session_consumed])
                            {
                                //session_tokens.resize(n_session_consumed);
                                break;
                            }

                            n_past++;
                            n_session_consumed++;

                            if (n_session_consumed >= session_tokens.Count)
                            {
                                break;
                            }
                        }
                        if (i > 0)
                        {
                            //embd.erase(embd.begin(), embd.begin() + i);
                        }
                    }

                    // evaluate tokens in batches
                    // embd is typically prepared beforehand to fit within a batch, but not always
                    for (int i = 0; i < embd.Count; i += aparams.n_batch)
                    {
                        int n_eval = embd.Count - i;
                        if (n_eval > aparams.n_batch)
                            n_eval = aparams.n_batch;

                        if (LlamaCppInterop.llama_eval(ctx, new[] { embd[i] }.ToList(), n_past, aparams.n_threads) != 0)
                        {
                            await Console.Error.WriteLineAsync($"{FuncName()} : failed to eval");
                            return;
                        }
                        n_past += n_eval;
                    }

                    if (embd.Count > 0 && !String.IsNullOrWhiteSpace(path_session))
                    {
                        //session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                        n_session_consumed = session_tokens.Count;
                    }
                }

                //        embd.clear();

                //        if ((int) embd_inp.size() <= n_consumed && !is_interacting) {
                //            // out of user input, sample next token
                //            const float   temp            = params.temp;
                //            const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
                //            const float   top_p           = params.top_p;
                //            const float   tfs_z           = params.tfs_z;
                //            const float   typical_p       = params.typical_p;
                //            const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
                //            const float   repeat_penalty  = params.repeat_penalty;
                //            const float   alpha_presence  = params.presence_penalty;
                //            const float   alpha_frequency = params.frequency_penalty;
                //            const int     mirostat        = params.mirostat;
                //            const float   mirostat_tau    = params.mirostat_tau;
                //            const float   mirostat_eta    = params.mirostat_eta;
                //            const bool    penalize_nl     = params.penalize_nl;

                //            // optionally save the session on first sample (for faster prompt loading next time)
                //            if (!path_session.empty() && need_to_save_session) {
                //                need_to_save_session = false;
                //                llama_save_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
                //            }

                //            llama_token id = 0;

                //            {
                //                auto logits = llama_get_logits(ctx);
                //                auto n_vocab = llama_n_vocab(ctx);

                //                // Apply params.logit_bias map
                //                for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                //                    logits[it->first] += it->second;
                //                }

                //                std::vector<llama_token_data> candidates;
                //                candidates.reserve(n_vocab);
                //                for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                //                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                //                }

                //                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                //                // Apply penalties
                //                float nl_logit = logits[llama_token_nl()];
                //                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                //                llama_sample_repetition_penalty(ctx, &candidates_p,
                //                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                //                    last_n_repeat, repeat_penalty);
                //                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                //                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                //                    last_n_repeat, alpha_frequency, alpha_presence);
                //                if (!penalize_nl) {
                //                    logits[llama_token_nl()] = nl_logit;
                //                }

                //                if (temp <= 0) {
                //                    // Greedy sampling
                //                    id = llama_sample_token_greedy(ctx, &candidates_p);
                //                } else {
                //                    if (mirostat == 1) {
                //                        static float mirostat_mu = 2.0f * mirostat_tau;
                //                        const int mirostat_m = 100;
                //                        llama_sample_temperature(ctx, &candidates_p, temp);
                //                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                //                    } else if (mirostat == 2) {
                //                        static float mirostat_mu = 2.0f * mirostat_tau;
                //                        llama_sample_temperature(ctx, &candidates_p, temp);
                //                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                //                    } else {
                //                        // Temperature sampling
                //                        llama_sample_top_k(ctx, &candidates_p, top_k);
                //                        llama_sample_tail_free(ctx, &candidates_p, tfs_z);
                //                        llama_sample_typical(ctx, &candidates_p, typical_p);
                //                        llama_sample_top_p(ctx, &candidates_p, top_p);
                //                        llama_sample_temperature(ctx, &candidates_p, temp);
                //                        id = llama_sample_token(ctx, &candidates_p);
                //                    }
                //                }
                //                // printf("`%d`", candidates_p.size);

                //                last_n_tokens.erase(last_n_tokens.begin());
                //                last_n_tokens.push_back(id);
                //            }

                //            // replace end of text token with newline token when in interactive mode
                //            if (id == llama_token_eos() && params.interactive && !params.instruct) {
                //                id = llama_token_newline.front();
                //                if (params.antiprompt.size() != 0) {
                //                    // tokenize and inject first reverse prompt
                //                    const auto first_antiprompt = ::llama_tokenize(ctx, params.antiprompt.front(), false);
                //                    embd_inp.insert(embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
                //                }
                //            }

                //            // add it to the context
                //            embd.push_back(id);

                //            // echo this to console
                //            input_noecho = false;

                //            // decrement remaining sampling budget
                //            --n_remain;
                //        } else {
                //            // some user input remains from prompt or interaction, forward it to processing
                //            while ((int) embd_inp.size() > n_consumed) {
                //                embd.push_back(embd_inp[n_consumed]);
                //                last_n_tokens.erase(last_n_tokens.begin());
                //                last_n_tokens.push_back(embd_inp[n_consumed]);
                //                ++n_consumed;
                //                if ((int) embd.size() >= params.n_batch) {
                //                    break;
                //                }
                //            }
                //        }

                //        // display text
                //        if (!input_noecho) {
                //            for (auto id : embd) {
                //                printf("%s", llama_token_to_str(ctx, id));
                //            }
                //            fflush(stdout);
                //        }
                //        // reset color to default if we there is no pending user input
                //        if (!input_noecho && (int)embd_inp.size() == n_consumed) {
                //            set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
                //        }

                //        // in interactive mode, and not currently processing queued inputs;
                //        // check if we should prompt the user for more
                //        if (params.interactive && (int) embd_inp.size() <= n_consumed) {

                //            // check for reverse prompt
                //            if (params.antiprompt.size()) {
                //                std::string last_output;
                //                for (auto id : last_n_tokens) {
                //                    last_output += llama_token_to_str(ctx, id);
                //                }

                //                is_antiprompt = false;
                //                // Check if each of the reverse prompts appears at the end of the output.
                //                for (std::string & antiprompt : params.antiprompt) {
                //                    if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
                //                        is_interacting = true;
                //                        is_antiprompt = true;
                //                        set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
                //                        fflush(stdout);
                //                        break;
                //                    }
                //                }
                //            }

                //            if (n_past > 0 && is_interacting) {
                //                // potentially set color to indicate we are taking user input
                //                set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);

                //#if defined (_WIN32)
                //                // Windows: must reactivate sigint handler after each signal
                //                signal(SIGINT, sigint_handler);
                //#endif

                //                if (params.instruct) {
                //                    printf("\n> ");
                //                }

                //                std::string buffer;
                //                if (!params.input_prefix.empty()) {
                //                    buffer += params.input_prefix;
                //                    printf("%s", buffer.c_str());
                //                }

                //                std::string line;
                //                bool another_line = true;
                //                do {
                //#if defined(_WIN32)
                //                    std::wstring wline;
                //                    if (!std::getline(std::wcin, wline)) {
                //                        // input stream is bad or EOF received
                //                        return 0;
                //                    }
                //                    win32_utf8_encode(wline, line);
                //#else
                //                    if (!std::getline(std::cin, line)) {
                //                        // input stream is bad or EOF received
                //                        return 0;
                //                    }
                //#endif
                //                    if (line.empty() || line.back() != '\\') {
                //                        another_line = false;
                //                    } else {
                //                        line.pop_back(); // Remove the continue character
                //                    }
                //                    buffer += line + '\n'; // Append the line to the result
                //                } while (another_line);

                //                // done taking input, reset color
                //                set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

                //                // Add tokens to embd only if the input buffer is non-empty
                //                // Entering a empty line lets the user pass control back
                //                if (buffer.length() > 1) {

                //                    // instruct mode: insert instruction prefix
                //                    if (params.instruct && !is_antiprompt) {
                //                        n_consumed = embd_inp.size();
                //                        embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());
                //                    }

                //                    auto line_inp = ::llama_tokenize(ctx, buffer, false);
                //                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

                //                    // instruct mode: insert response suffix
                //                    if (params.instruct) {
                //                        embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
                //                    }

                //                    n_remain -= line_inp.size();
                //                }

                //                input_noecho = true; // do not echo this again
                //            }

                //            if (n_past > 0) {
                //                is_interacting = false;
                //            }
                //        }

                //        // end of text token
                //        if (!embd.empty() && embd.back() == llama_token_eos()) {
                //            if (params.instruct) {
                //                is_interacting = true;
                //            } else {
                //                fprintf(stderr, " [end of text]\n");
                //                break;
                //            }
                //        }

                //        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
                //        if (params.interactive && n_remain <= 0 && params.n_predict != -1) {
                //            n_remain = params.n_predict;
                //            is_interacting = true;
                //        }
            }

            await Task.CompletedTask;
        }

        private static string FuncName([CallerMemberName] string? callerName = default) => callerName ?? String.Empty;

        private static string gpt_random_prompt(Random rng)
        {
            var r = rng.Next() % 10;
            switch (r)
            {
                case 0: return "So";
                case 1: return "Once upon a time";
                case 2: return "When";
                case 3: return "The";
                case 4: return "After";
                case 5: return "If";
                case 6: return "import";
                case 7: return "He";
                case 8: return "She";
                case 9: return "They";
                default: return "To";
            }
        }
    }
}
