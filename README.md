# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

C# bindings for llama.cpp including a .NET core library and sample projects (cli & web).

![demo-web](https://user-images.githubusercontent.com/1760549/233868319-59dda027-4279-462f-9233-2825856cded9.gif)

![demo](https://user-images.githubusercontent.com/1760549/233812516-e1504362-8379-4c20-baef-763ffacf8ef1.gif)

### Quick Start

Build (on Windows use the VS2022 x64 command prompt, on Linux make sure to [install dotnet](https://learn.microsoft.com/en-us/dotnet/core/install/linux)):
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
dotnet build -c Release
```

Windows:
```
cd x64\Release
LlamaCppCli.exe 4 ggml-vicuna-13b-1.1-q4_0.bin
```

Linux:
```
cd x64/Release
./LlamaCppCli 4 ggml-vicuna-13b-1.1-q4_0.bin
```

## Usage
```
USAGE:
    LlamaCppCli <SampleIndex> <ModelPath> [TemplateName]
SAMPLES:
    [1] = RawInterfaceSample
    [2] = WrappedInterfaceSampleWithoutSession
    [3] = WrappedInterfaceSampleWithSession
    [4] = WrappedInterfaceSampleWithSessionInteractive
    [5] = GetEmbeddings
TEMPLATES:
    "Vicuna v1.1"
    "Alpaca (no input)"
    "Alpaca (input)"
    "WizardLM"
```

## Models

You will need a model in GGML format, the 13B parameters appears to perform well if you have the memory (8-12GB depending on the model).
If you have a lot of RAM (i.e. 48GB+) you could try a 65B version though it is much slower on the predictions.

Some models can be found below.

- [eachadea/ggml-vicuna-13b-1.1](https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/tree/main)
- [TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g-GGML](https://huggingface.co/TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g-GGML/tree/main)
- [TheBloke/wizardLM-7B-GGML](https://huggingface.co/TheBloke/wizardLM-7B-GGML)

### Sample Code (CLI)
```
using LlamaCppLib;

using (var model = new LlamaCpp("Model X"))
{
    // Load model
    model.Load("ggml-vicuna-13b-v1.1-q4_0.bin");

    // Configure model
    model.Configure(options =>
    {
        options.ThreadCount = 4;
        options.TopK = 50;
        options.TopP = 0.95f;
        options.Temperature = 0.1f;
        options.RepeatPenalty = 1.1f;
    });

    // Create a new conversation session
    var session = model.CreateSession("Conversation X");

    while (true)
    {
        // Get a prompt
        Console.Write("> ");
        var prompt = Console.ReadLine();

        // Quit on blank prompt
        if (String.IsNullOrWhiteSpace(prompt))
            break;

        // If your model needs a template, here you would format it

        // Run the predictions
        await foreach (var token in session.Predict(prompt))
            Console.Write(token);
    }
}
```

### API Endpoints (Model)
```
GET /model/list
GET /model/load?modelName={modelName}
GET /model/unload?modelName={modelName}
GET /model/status
GET /model/configure?threadCount={threadCount}&topK={topK}&topP{topP}&temperature={temperature}&repeatPenalty={repeatPenalty}
```

### API Endpoints (Session - i.e. Conversation)
```
GET /session/list
GET /session/create?sessionName={sessionName}
GET /session/destroy?sessionName={sessionName}
GET /session/configure?sessionName={sessionName}&template={template}
GET /session/predict?sessionName={sessionName}&prompt={prompt}
```

### Future Ideas

- [X] Dynamic model loading
- [X] Expose minimal API
- [X] Session/conversation support
- [X] Support Windows/Linux
- [ ] Add basic web app (WIP)

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
