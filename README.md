# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

C# bindings for llama.cpp including a .NET core library and sample projects (CLI & Web).

![demo-web](https://user-images.githubusercontent.com/1760549/233868319-59dda027-4279-462f-9233-2825856cded9.gif)

![demo](https://user-images.githubusercontent.com/1760549/233812516-e1504362-8379-4c20-baef-763ffacf8ef1.gif)

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
LlamaCppCli.exe 3 ggml-vicuna-13b-1.1-q8_0.bin
```

Linux:
```
cd x64/Release
./LlamaCppCli 3 ggml-vicuna-13b-1.1-q8_0.bin
```

## Usage
```
USAGE:
    LlamaCppCli.dll <SampleIndex> <ModelPath> [TemplatePath]
SAMPLES:
    [0] = RawInterfaceSample
    [1] = WrappedInterfaceSampleWithoutSession
    [2] = WrappedInterfaceSampleWithSession
    [3] = WrappedInterfaceSampleWithSessionInteractive
    [4] = GetEmbeddings
```

## Models

You will need a model in GGML format, the 13B parameters appears to perform well if you have the memory (8-12GB depending on the model).
If you have a lot of RAM (i.e. 48GB+) you could try a 65B version though it is much slower on the predictions.

Some models can be found below.

- [TheBloke on Hugging Face](https://huggingface.co/TheBloke)

### Sample Code (interactive CLI)
```
using LlamaCppLib;

// Configure some model options
var options = new LlamaCppOptions
{
    ThreadCount = 4,
    TopK = 40,
    TopP = 0.95f,
    Temperature = 0.8f,
    RepeatPenalty = 1.1f,
    Mirostat = Mirostat.Mirostat2,
};

// Create new named model with options
using var model = new LlamaCpp("Vicuna v1.1", options);

// Load model file
model.Load(@"ggml-vicuna-13b-v1.1-q8_0.bin");

// Create new conversation session and configure prompt template
var session = model.CreateSession(ConversationName);
session.Configure(options => options.Template = File.ReadAllText(@"template_vicuna-v1.1.txt"));

while (true)
{
    // Get a prompt
    Console.Write("> ");
    var prompt = Console.ReadLine();

    // Quit on blank prompt
    if (String.IsNullOrWhiteSpace(prompt))
        break;

    // Run the predictions
    await foreach (var token in session.Predict(prompt))
        Console.Write(token);
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
