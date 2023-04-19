# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

Simple C# bindings for llama.cpp with two sample projects (cli & web).

### Quick Start

In a VS2022 x64 developer command prompt, type:
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
MSBuild.exe /property:Configuration=Release
cd x64\Release
LlamaCppCli.exe
```

### Sample Code (CLI)
```
using LlamaCppLib;

using (var model = new LlamaCpp("vicuna-13b-v1.1"))
{
    model.Load(@"D:\LLM_MODELS\lmsys\vicuna-13b-v1.1\ggml-vicuna-13b-v1.1-q4_0.bin");

    model.Configure(options =>
    {
        options.ThreadCount = 16;
        options.EndOfStreamToken = "</s>";
    });

    var session = model.NewSession("Conversation #1");

    session.Configure(options =>
    {
        options.InitialContext.AddRange(
            new[]
            {
                "A chat between a user and an assistant.",
                "USER: Hello!",
                "ASSISTANT: Hello!</s>",
                "USER: How are you?",
                "ASSISTANT: I am good.</s>",
            }
        );

        options.Roles.AddRange(new[] { "USER", "ASSISTANT", "ASSIST" });
    });

    var prompts = new[]
    {
        "USER: How many planets are there in the solar system?",
        "USER: Can you list the planets?",
        "USER: What is Vicuna 13B?",
        "USER: It is a large language model.",
    };

    Console.WriteLine(session.InitialContext.Aggregate((a, b) => $"{a}\n{b}"));

    foreach (var prompt in prompts)
    {
        Console.WriteLine(prompt);

        await foreach (var token in session.Predict(prompt))
            Console.Write(token);
    }

    // Print conversation
    Console.WriteLine();
    Console.WriteLine($" --------------------------------------------------------------------------------------------------");
    Console.WriteLine($"| Transcript                                                                                       |");
    Console.WriteLine($" --------------------------------------------------------------------------------------------------");

    foreach (var topic in session.Conversation)
        Console.WriteLine(topic);
}
```

### API Endpoints (Model)
```
GET /model/list
GET /model/load?modelName={modelName}
GET /model/unload?modelName={modelName}
GET /model/status
```

### API Endpoints (Session - i.e. Conversation)
```
GET /session/list
GET /session/new?sessionName={sessionName}
GET /session/delete?sessionName={sessionName}
GET /session/predict?sessionName={sessionName}&prompt={prompt}
```

### Future Ideas

- [X] Dynamic model loading
- [X] Expose minimal API
- [X] Session/conversation support
- [ ] Add basic web app

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
