# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

C# bindings for llama.cpp with two sample projects (cli & web).

### Quick Start

In a VS2022 x64 developer command prompt, type:
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
MSBuild.exe /property:Configuration=Release
cd x64\Release
LlamaCppCli.exe
```

### Sample Code
```
using (var model = new LlamaCpp("vicuna-13b"))
{
    model.Load(@"D:\LLM_MODELS\lmsys\vicuna-13b\ggjt-vicuna-13b-f16-q4_0.bin");

    model.Configure(options => {
        options.ThreadCount = 16;
        options.InstructionPrompt = "### Human:";
        options.StopOnInstructionPrompt = true;
    });

    // Initial context
    var context = new StringBuilder(
        """
        ### Human: You are a professor from MIT.
        ### Assistant: I confirm.
        """
    );

    var prompt = "Explain the correlation and distanciation of recurrent and recursive functions.";

    await foreach (var token in model.Predict(context, prompt))
    {
        Console.Write(token);
    }

    // Print conversation
    Console.WriteLine("\n");
    Console.WriteLine($" ---------------------------------------------------------------------------------");
    Console.WriteLine($"| Transcript                                                                      |");
    Console.WriteLine($" ---------------------------------------------------------------------------------");
    Console.WriteLine(context);
}
```

### API Endpoints
```
GET /models
GET /load?modelName={modelName}
GET /unload
GET /status
GET /context?context={context}&reset={false|true}
GET /predict?prompt={prompt}
```

### Future Ideas

- [X] Dynamic model loading
- [X] Expose minimal API
- [ ] Add basic web app

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
