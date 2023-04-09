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

### Future Ideas

- Dynamic model loading
- Exposing a minimal completion API
- Add a basic web frontend

### Acknowledgments
[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) for the LLaMA implementation in C++
