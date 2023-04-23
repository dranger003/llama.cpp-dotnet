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

using (var model = new LlamaCpp("vicuna-13b-v1.1"))
{
    model.Load("ggml-vicuna-13b-v1.1-q4_1.bin");

    model.Configure(options =>
    {
        options.ThreadCount = 16;
        options.TopK = 40;
        options.TopP = 0.95f;
        options.Temperature = 0.0f;
        options.RepeatPenalty = 1.1f;
    });

    var session = model.CreateSession("Conversation #1");
    session.Configure(options => options.InitialContext.AddRange(new[] {
        $"Hi! How can I be of service today?",
        $"Hello! How are you doing?",
        $"I am doing great! Thanks for asking.",
        $"Can you help me with some questions please?",
        $"Absolutely, what questions can I help you with?",
        $"How many planets are there in the solar system?",
    }));

    Console.WriteLine(session.InitialContext.Aggregate((a, b) => $"{a}\n{b}"));

    foreach (var prompt in new[] {
        $"Can you list the planets of our solar system?",
        $"What do you think Vicuna 13B is according to you?",
        $"Vicuna 13B is a large language model (LLM).",
    })
    {
        Console.WriteLine(prompt);
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
GET /session/configure?sessionName={sessionName}&initialContext={initialContext}
GET /session/predict?sessionName={sessionName}&prompt={prompt}
```

### Future Ideas

- [X] Dynamic model loading
- [X] Expose minimal API
- [X] Session/conversation support
- [ ] Add basic web app

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
