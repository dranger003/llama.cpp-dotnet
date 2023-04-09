#include "common.h"
#include "llama.h"

#define DLLEXPORT __declspec(dllexport)

extern "C"
{
	// llama_context_params llama_context_default_params()
	DLLEXPORT llama_context_params _llama_context_default_params() {
		return llama_context_default_params();
	}

	// llama_context* llama_init_from_file(const char* path_model, llama_context_params params)
	DLLEXPORT llama_context* _llama_init_from_file(const char* path_model, llama_context_params params) {
		return llama_init_from_file(path_model, params);
	}

	// std::vector<llama_token> llama_tokenize(llama_context* ctx, const std::string& text, bool add_bos)
	DLLEXPORT void _llama_tokenize(llama_context* ctx, const char* text, bool add_bos, llama_token** r_tokens, int* r_tokens_len) {
		auto tokens = llama_tokenize(ctx, text, add_bos);

		auto len = sizeof(llama_token) * tokens.size();
		auto buf = (llama_token*)malloc(len);
		if (buf) {
			std::memcpy(buf, tokens.data(), len);
			*r_tokens = buf;
			*r_tokens_len = (int)tokens.size();
		}
	}

	DLLEXPORT void _llama_tokenize_free(llama_token* tokens) {
		free(tokens);
	}

	// int llama_n_ctx(llama_context* ctx)
	DLLEXPORT int _llama_n_ctx(llama_context* ctx) {
		return llama_n_ctx(ctx);
	}

	// int llama_eval(llama_context* ctx, const llama_token* tokens, int n_tokens, int n_past, int n_threads)
	DLLEXPORT int _llama_eval(llama_context* ctx, const llama_token* tokens, int n_tokens, int n_past, int n_threads) {
		return llama_eval(ctx, tokens, n_tokens, n_past, n_threads);
	}

	// llama_token llama_sample_top_p_top_k(llama_context* ctx, const llama_token* last_n_tokens_data, int last_n_tokens_size, int top_k, float top_p, float temp, float repeat_penalty)
	DLLEXPORT llama_token _llama_sample_top_p_top_k(llama_context* ctx, const llama_token* last_n_tokens_data, int last_n_tokens_size, int top_k, float top_p, float temp, float repeat_penalty) {
		return llama_sample_top_p_top_k(ctx, last_n_tokens_data, last_n_tokens_size, top_k, top_p, temp, repeat_penalty);
	}

	// const char* llama_token_to_str(llama_context* ctx, llama_token token)
	DLLEXPORT const char* _llama_token_to_str(llama_context* ctx, llama_token token) {
		auto token_str = llama_token_to_str(ctx, token);

		auto len = std::strlen(token_str);
		auto buf = (char*)malloc(len + 1);
		if (buf) {
			std::strcpy(buf, token_str);
			return buf;
		}

		return nullptr;
	}

	DLLEXPORT void _llama_token_to_str_free(char* token_str) {
		free(token_str);
	}

	// void llama_print_timings(llama_context* ctx)
	DLLEXPORT void _llama_print_timings(llama_context* ctx) {
		llama_print_timings(ctx);
	}

	// void llama_free(llama_context* ctx)
	DLLEXPORT void _llama_free(llama_context* ctx) {
		llama_free(ctx);
	}
}
