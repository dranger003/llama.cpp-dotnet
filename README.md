# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### *Update 11/2023*
*There has been a major overhaul of the upstream repo to support parallel decoding.*  
*The upstream changes have been merged.*

### Demo

![demo](https://github.com/dranger003/llama.cpp-dotnet/assets/1760549/ad560ac5-31ca-4cf0-93a5-a1a6ccf9b446)

### Description

Minimal C# bindings for llama.cpp including a .NET core library, API server/client and samples.

### Quick Start

Build - requires CUDA installed (on Windows use the VS2022 x64 command prompt, on Linux make sure to install cmake and [dotnet](https://learn.microsoft.com/en-us/dotnet/core/install/linux)):
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
dotnet build -c Release /p:Platform="Any CPU"
```
If you don't need to compile the native libraries, you can also append `/p:NativeLibraries=OFF` to the `dotnet` build command above.

### Minimal Sample

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

### Models

You will need a model in GGUF format, the 13B parameters appears to perform well if you have the memory (8-12GB depending on the quantized model).
If you have a lot of RAM (i.e. 48GB+) you could try a 65B version though it is much slower on the predictions, especially without a GPU.

A lot of models can be found below.

- [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

### Todo

- [x] Refactor the Web API stack to match the new Library API
- [x] Support parallel decoding

### Acknowledgments

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++  
