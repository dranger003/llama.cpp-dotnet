#include <iostream>
#include <fstream>
#include <format>

#include "common.h"
#include "llama.h"

void run_vicuna()
{
	gpt_params params;
	{
		params.seed = (int)::time(nullptr);
		params.n_threads = 16;
		params.prompt = "";
		params.antiprompt = { "### Human:" };
		params.n_predict = -1;
		params.repeat_last_n = 2048;
		params.repeat_penalty = 2.0f;
		params.n_ctx = 2048;
		params.memory_f16 = false;
		params.temp = 0.0f;
		params.n_batch = 1024;
		params.n_keep = 0;
		params.model = "D:\\LLM_MODELS\\eachadea\\ggml-vicuna-13b-4bit\\ggml-vicuna-13b-4bit-rev1.bin";
	}

	{
		std::ifstream file("prompt.txt");
		std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
	}

	llama_context* ctx;
	{
		auto lparams = ::llama_context_default_params();
		lparams.n_ctx = params.n_ctx;
		lparams.n_parts = params.n_parts;
		lparams.seed = params.seed;
		lparams.f16_kv = params.memory_f16;
		lparams.use_mlock = params.use_mlock;
		ctx = ::llama_init_from_file(params.model.c_str(), lparams);
	}

	std::fprintf(stderr, "llama_print_system_info: %s\n", __func__, ::llama_print_system_info());

	auto embd_inp = ::llama_tokenize(ctx, params.prompt, false);
	const int n_ctx = ::llama_n_ctx(ctx);

	params.n_keep = (int)embd_inp.size();

	int n_past = 0;
	int n_consumed = 0;

	std::vector<llama_token> last_n_tokens(n_ctx);
	std::vector<llama_token> embd;

	//std::ofstream ctx_file("context.txt", std::ios::out);
	while (true) {
		//{
		//	auto context_length = std::count_if(last_n_tokens.begin(), last_n_tokens.end(), [](const llama_token& id) { return id != 0; });
		//	std::ofstream ctx_file("context.txt", std::ios::out | std::ios::app);
		//	ctx_file << std::format("[n_ctx={}][embd_inp={}][last_n_tokens={}][embd={}][n_consumed={}][n_past={}]", n_ctx, embd_inp.size(), context_length, embd.size(), n_consumed, n_past) << std::endl;
		//	//std::ostream_iterator<std::string> out_iterator(ctx_file);
		//	//std::transform(last_n_tokens.begin(), last_n_tokens.end(), out_iterator, [&](const llama_token& id) { return id <= 0 ? "" : llama_token_to_str(ctx, id); });
		//}

		if (embd.size() > 0) {
			if (n_past + (int)embd.size() > n_ctx) {
				const int n_left = n_past - params.n_keep;
				n_past = params.n_keep;
				embd.insert(embd.begin(), last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(), last_n_tokens.end() - embd.size());
			}

			::llama_eval(ctx, embd.data(), (int)embd.size(), n_past, params.n_threads);
		}

		n_past += (int)embd.size();
		embd.clear();

		if ((int)embd_inp.size() <= n_consumed) {
			auto id = ::llama_sample_top_p_top_k(
				ctx,
				last_n_tokens.data() + n_ctx - params.repeat_last_n,
				params.repeat_last_n,
				params.top_k,
				params.top_p,
				params.temp,
				params.repeat_penalty
			);

			last_n_tokens.erase(last_n_tokens.begin());
			last_n_tokens.push_back(id);

			embd.push_back(id);
		}
		else {
			while ((int)embd_inp.size() > n_consumed) {
				embd.push_back(embd_inp[n_consumed]);
				last_n_tokens.erase(last_n_tokens.begin());
				last_n_tokens.push_back(embd_inp[n_consumed]);
				++n_consumed;
				if ((int)embd.size() >= params.n_batch) {
					break;
				}
			}
		}

		for (auto id : embd) {
			printf("%s", ::llama_token_to_str(ctx, id));
			fflush(stdout);
		}

		{
			std::string last_output;
			for (auto id : last_n_tokens) {
				last_output += llama_token_to_str(ctx, id);
			}
			for (std::string& antiprompt : params.antiprompt) {
				if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
					goto quit;
				}
			}
		}
	}

quit:
	::llama_print_timings(ctx);
	::llama_free(ctx);
}

int main(/*int argc, char* argv[]*/)
{
	run_vicuna();

	return 0;
}
