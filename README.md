# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

~~Quick hooks to run llama.cpp in dotnet using light modifications for easy future upstream merges.~~

I might revisit this idea of running main.exe separately, however for now I am focusing on bindings for additional flexibility. The original project remains unmodified and can run as is and will be compiled as part of this project.

### Quick Start

Open a VS2022 x64 develper command prompt and type:
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
MSBuild.exe /property:Configuration=Release
x64\Release\llama.cpp-dotnet.exe
```

### Sample Code
```
using (var model = new LlamaCpp())
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
        ### Human: The United States president is Justin Trudeau.
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

- Dynamic model loading
- Exposing a minimal completion API
- Add a basic web frontend

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
