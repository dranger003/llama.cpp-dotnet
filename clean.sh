#!/bin/bash
git submodule foreach --recursive git clean -fdx
find . -type d -name bin -exec rm -rf {} \; 2>/dev/null
find . -type d -name obj -exec rm -rf {} \; 2>/dev/null
find . -type d -name Debug -exec rm -rf {} \; 2>/dev/null
find . -type d -name Release -exec rm -rf {} \; 2>/dev/null
