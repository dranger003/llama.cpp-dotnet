# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

C# bindings for llama.cpp including a .NET core library and sample projects (cli & web).

![demo](https://user-images.githubusercontent.com/1760549/233812516-e1504362-8379-4c20-baef-763ffacf8ef1.gif)

### Quick Start

In a VS2022 x64 developer command prompt, type:
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
MSBuild.exe /property:Configuration=Release
cd x64\Release
LlamaCppCli.exe
```

## Models

You will need a model in GGML format.

[ggml-vicuna-13b](https://huggingface.co/eachadea/ggml-vicuna-13b-1.1/tree/main)

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
GET /session/configure?sessionName={sessionName}
GET /session/predict?sessionName={sessionName}&prompt={prompt}
```

### Future Ideas

- [X] Dynamic model loading
- [X] Expose minimal API
- [X] Session/conversation support
- [ ] Add basic web app

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
