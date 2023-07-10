# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

C# bindings for llama.cpp including a .NET core library and sample projects (CLI & Web API).

![demo-web-2](https://github.com/dranger003/llama.cpp-dotnet/assets/1760549/f261ae13-20e1-4f41-964f-e1942649dcd4)

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
LlamaCppCli.exe 0 ggml-vicuna-13b-1.1-q8_0.bin
```

Linux:
```
cd x64/Release
./LlamaCppCli 0 ggml-vicuna-13b-1.1-q8_0.bin
```

## Samples
```
Usage: LlamaCppCli.dll <SampleIndex> <SampleArgs>
Available sample(s):
    [0] = LocalSample
    [1] = RemoteSample
```
## Local
```
Usage: LlamaCppCli.dll 0 model_path [gpu_layers] [template]
```
## Remote
```
Usage: LlamaCppCli.dll 1 base_url model_name [template]
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
var options = new LlamaCppModelOptions
{
    ContextSize = 2048,
    GpuLayers = 24,
    Template = "You are a helpful assistant.\n\nUSER:\n{0}\n\nASSISTANT:\n",
};

// Load model file
using var model = new LlamaCpp();
model.Load(@"ggml-vicuna-13b-v1.1-q8_0.bin", options);

// Configure some prediction options
var predictOptions = new LlamaCppPredictOptions
{
    ThreadCount = 4,
    TopK = 40,
    TopP = 0.95f,
    Temperature = 0.1f,
    RepeatPenalty = 1.1f,
    PenalizeNewLine = false,
    Mirostat = Mirostat.Mirostat2,
};

while (true)
{
    // Get a prompt
    Console.Write("> ");
    var prompt = Console.ReadLine();

    // Quit on blank prompt
    if (String.IsNullOrWhiteSpace(prompt))
        break;

    // Set-up prompt using template
    predictOptions.Prompt = String.Format(modelOptions.Template, prompt);

    // Run the predictions
    await foreach (var prediction in model.Predict(predictOptions))
        Console.Write(prediction.Value);
}
```

### API Endpoints (Model)
```
GET /model/list
GET /model/load?modelName={modelName}&modelOptions={modelOptions}
GET /model/unload
GET /model/status
GET /model/predict?predictOptions={predictOptions}
```

### TODO

- [X] Dynamic model loading
- [X] Expose minimal API
- [X] Support Windows/Linux
- [ ] Support [Falcon LLM](https://github.com/cmp-nct/ggllm.cpp)

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
