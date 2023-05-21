@echo off
cd llama.cpp
git clean -fdx
cd ..\bert.cpp
git clean -fdx
cd ggml
git clean -fdx
cd ..\..
for /d /r . %%d in (bin) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (obj) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (x64) do @if exist "%%d" rmdir /s /q "%%d"
