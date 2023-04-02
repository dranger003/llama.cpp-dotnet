# llama.cpp-dotnet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### Description

Quick hooks to run llama.cpp in dotnet using light modifications for easy future upstream merges.

The original project can also run as is without any required changes.

For details on the added hooks, you can look at commit [`2b44810`](https://github.com/dranger003/llama.cpp/commit/2b4481038c416a0a9a386091f460a417de6797f1).

### Quick Start

Open a vs2022 x64 develper command prompt and type:
```
git clone --recursive https://github.com/dranger003/llama.cpp-dotnet.git
cd llama.cpp-dotnet
msbuild
```

### Future Ideas

- Dynamic model loading
- Exposing a minimal completion API
- Add a basic web frontend
