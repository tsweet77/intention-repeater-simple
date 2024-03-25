# Intention Repeater Simple
A simple Intention Repeater created with help from WebGPT and Claude 3 Opus.

The regular Intention Repeater MAX does not use Hashing.

Code requires picosha2.h

CUDA requires: picosha2.h, zconf.h, zlib.dll, zlib.h

To compile CUDA:

nvcc Intention_Repeater_Simple_CUDA.cu -o Intention_Repeater_Simple_CUDA.exe -L/<userpath>/miniconda3/Library/lib -lz

Use the path to your miniconda3 library files.

Requires miniconda3 library for compiling on Windows.
