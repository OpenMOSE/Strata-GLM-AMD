#!/bin/sh
# Build dump_tensors against a llama.cpp build tree (default: ../../../llama.cpp/build-hip beside this checkout).
#   tools/glm_oracle/build.sh [LLAMA_CPP_DIR] [BUILD_DIR]
set -e
here="$(cd "$(dirname "$0")" && pwd)"
llama="${1:-$here/../../../llama.cpp}"
build="${2:-$llama/build-hip}"
out="$build/bin/dump_tensors"
g++ -O2 -std=c++17 -o "$out" "$here/dump_tensors.cpp" \
  -I "$llama/include" -I "$llama/ggml/include" \
  -L "$build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$build/bin" -lpthread
echo "built $out"
