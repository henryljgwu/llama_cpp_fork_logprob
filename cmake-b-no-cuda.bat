rm -rf build
pause
cmake -S . -B build
pause
cmake --build build --config Release
pause