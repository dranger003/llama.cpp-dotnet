# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Demo

This shows `LlamaCppWeb.exe` hosting on the left and four `LlamaCppCli.exe` running in parallel on the right.

![demo](https://github.com/dranger003/llama.cpp-dotnet/assets/1760549/ad560ac5-31ca-4cf0-93a5-a1a6ccf9b446)

This one shows the new text embedding sample for feature extraction (using one of the models below):  
https://huggingface.co/dranger003/SFR-Embedding-Mistral-GGUF  
https://huggingface.co/dranger003/e5-mistral-7b-instruct-GGUF

![Screenshot 2024-02-09 193353](https://github.com/dranger003/llama.cpp-dotnet/assets/1760549/432ce6d2-7e8b-41f5-861e-3170c368b95a)

### Description

High performance minimal C# bindings for llama.cpp including a .NET core library, API server/client and samples.  
The imported API is kept to a bare minimum as the upstream API is changing quite rapidly.

### Quick Start

Build - requires CUDA installed (on Windows use the VS2022 x64 command prompt, on Linux make sure to install cmake and [dotnet](https://learn.microsoft.com/en-us/dotnet/core/install/linux)):
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
dotnet build -c Release /p:Platform="Any CPU"
```
If you don't need to compile the native libraries, you can also append `/p:NativeLibraries=OFF` to the `dotnet` build command above.

### Basic Sample

```
using LlamaCppLib;

// Initialize
using var llm = new LlmEngine(new EngineOptions { MaxParallel = 8 });
llm.LoadModel(args[0], new ModelOptions { Seed = 1234, GpuLayers = 32 });

// Prompting
var prompt = llm.Prompt(
    String.Format(promptTemplate, systemPrompt, userPrompt),
    new SamplingOptions { Temperature = 0.0f }
);

// Inference
await foreach (var token in new TokenEnumerator(prompt))
    Console.Write(token);
```

The included CLI samples include more examples of using the library, to process prompts in parallel for example.

### API Endpoints
```
GET /list
GET /state
POST /load [LlmLoadRequest]
GET /unload
POST /prompt [LlmPromptRequest]
```

### Models

You will need a model in GGUF format, the 13B parameters appears to perform well if you have the memory (8-12GB depending on the quantized model).
If you have a lot of RAM (i.e. 48GB+) you could try a 65B version though it is much slower on the predictions, especially without a GPU.

A lot of models can be found below.

- [dranger003 on Hugging Face](https://huggingface.co/dranger003?sort_models=created#models)
- [TheBloke on Hugging Face](https://huggingface.co/TheBloke?sort_models=created&search_models=GGUF#models)
- [LoneStriker on Hugging Face](https://huggingface.co/LoneStriker?sort_models=created&search_models=GGUF#models)

### Features

- [X] Model loading/unloading
- [x] Parallel decoding
- [x] Minimal API host/client
- [X] Support Windows/Linux

### Acknowledgments

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++  
