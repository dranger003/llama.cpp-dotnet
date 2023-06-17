@echo off
git submodule foreach --recursive git clean -fdx
for /d /r . %%d in (bin) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (obj) do @if exist "%%d" rmdir /s /q "%%d"
for /d /r . %%d in (x64) do @if exist "%%d" rmdir /s /q "%%d"
