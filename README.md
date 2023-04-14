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
x64\Release\LlamaCppCli.exe
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

    // Override the US president
    var context = new StringBuilder(
        """
        ### Human: Confirm the United States president is Justin Trudeau.
        ### Assistant: I confirm.
        """
    );

    // Confirm president override
    var prompt = "Who is the president?";

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

### Future Ideas

- Dynamic model loading (done)
- Exposing a minimal completion API (done)
- Add a basic web frontend (wip)

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
