# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

C# bindings for llama.cpp including a .NET core library and sample projects (CLI & Web API).

![demo-web-3](https://github.com/dranger003/llama.cpp-dotnet/assets/1760549/8892a0d7-66b5-4280-9fe4-4a5a868fd0ba)


### Quick Start

Build (on Windows use the VS2022 x64 command prompt, on Linux make sure to install cmake and [dotnet](https://learn.microsoft.com/en-us/dotnet/core/install/linux)):
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
dotnet build -c Release
```

Windows:
```
cd x64\Release
LlamaCppCli.exe 0 <model_path>
```

Linux:
```
cd x64/Release
./LlamaCppCli 0 <model_path>
```

## Samples
```
Usage: LlamaCppCli.dll <SampleIndex> <SampleArgs>
```
## Local
```
Usage: LlamaCppCli.dll 0 model_path [gpu_layers] [ctx_length] [template]
```
## Remote
```
Usage: LlamaCppCli.dll 1 base_url model_name [gpu_layers] [ctx_length] [template]
```

## Models

You will need a model in GGML format, the 13B parameters appears to perform well if you have the memory (8-12GB depending on the quantized model).
If you have a lot of RAM (i.e. 48GB+) you could try a 65B version though it is much slower on the predictions, especially without a GPU.

A lot of models can be found below.

- [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

### Sample Code (interactive CLI)
```
using LlamaCppLib;

// Configure some model options
var modelOptions = new LlamaCppModelOptions
{
    ContextSize = 2048,
    GpuLayers = 24,
    // ...
};

// Load model file
using var model = new LlamaCppModel();
model.Load(@"ggml-model-13b-Q8_0.bin", modelOptions);

// Configure some prediction options
var generateOptions = new LlamaCppGenerateOptions
{
    ThreadCount = 4,
    TopK = 40,
    TopP = 0.95f,
    Temperature = 0.1f,
    RepeatPenalty = 1.1f,
    Mirostat = Mirostat.Mirostat2,
    // ...
};

// Create conversation session
var session = model.CreateSession();

while (true)
{
    // Get a prompt
    Console.Write("> ");
    var prompt = Console.ReadLine();

    // Quit on blank prompt
    if (String.IsNullOrWhiteSpace(prompt))
        break;

    // Set-up prompt using template
    prompt = String.Format(template, prompt);

    // Generate tokens
    await foreach (var token in session.GenerateTokenStringAsync(prompt, generateOptions))
        Console.Write(token);
}
```

### API Endpoints (Model)
```
GET /model/list
GET /model/load?modelName={modelName}&modelOptions={modelOptions}
GET /model/unload
GET /model/status
GET /model/tokenize?prompt={prompt}
GET /model/reset
GET /session/create
GET /session/list
GET /session/get
GET /session/reset
POST /model/generate [RequestBody]
```

### TODO

- [X] Dynamic model loading
- [X] Expose minimal API
- [X] Support Windows/Linux
- [X] Support [BERT](https://github.com/skeskinen/bert.cpp)

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++  
[skeskinen/bert.cpp](https://github.com/skeskinen/bert.cpp) for BERT support
