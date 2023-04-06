# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

Quick hooks to run llama.cpp in dotnet using light modifications for easy future upstream merges.

The original project can also run as is without any required changes.

### Quick Start

Open a vs2022 x64 develper command prompt and type:
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
