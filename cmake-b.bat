rm -rf build
pause
cmake -S . -B build -DGGML_CUDA=ON
pause
cmake --build build --config Release
pause