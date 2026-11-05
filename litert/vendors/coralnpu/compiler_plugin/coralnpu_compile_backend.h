// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ODML_LITERT_LITERT_VENDORS_CORALNPU_COMPILER_PLUGIN_CORALNPU_COMPILE_BACKEND_H_
#define ODML_LITERT_LITERT_VENDORS_CORALNPU_COMPILER_PLUGIN_CORALNPU_COMPILE_BACKEND_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"

namespace coralnpu {

// One AOT compilation job: the whole set of selected partitions, serialized as
// a single LiteRT/TFLite model whose subgraphs are the partitions.
//
// The compiler plugin picks the partitions; how they become Coral NPU
// executables is the backend's business.  Today there is exactly one backend
// (`aot_compile_backend.cc`), which drives `scripts/compile_litert.py`.
struct CompileRequest {
  // `soc_model` string the runtime asked for, e.g. "coralnpu".  Empty when the
  // runtime did not specify one.
  std::string soc_model;
  // Serialized model bytes, owned by the caller and valid for the duration of
  // the call.
  const uint8_t* model_data = nullptr;
  size_t model_size = 0;
  // Offset of the first valid model byte inside `model_data`.  LiteRT returns
  // the serialization as a view into a larger allocation, so the flatbuffer
  // starts at `model_data + model_offset`, not at `model_data`.
  size_t model_offset = 0;
  // Number of partitions in `model_data`, i.e. the number of subgraphs the
  // plugin selected.  The backend must produce this many call infos.
  int num_partitions = 0;
};

// What the backend produced.  A single bytecode module is the norm: everything
// Coral NPU runs for this graph is compiled into one IREE VMFB, and each
// partition becomes one exported function inside it.
struct CompileArtifact {
  // VMFB bytes.  Copied into the plugin's compiled result.
  std::vector<uint8_t> byte_code;
  // Function names the runtime should call, in partition order.  Each entry is
  // the exported function name in `byte_code`, e.g. "partition_0".
  std::vector<std::string> call_infos;
};

// Compilation seam.  Replaced wholesale in tests and by out-of-tree backends
// (for example a Torq SDK backend) without touching the ABI code.
class CompileBackend {
 public:
  virtual ~CompileBackend() = default;

  // Compiles `request` into `artifact`.  On failure `artifact` is left in an
  // unspecified but destructible state and a specific LiteRtStatus is returned.
  virtual LiteRtStatus Compile(const CompileRequest& request,
                               CompileArtifact* artifact) = 0;
};

// Returns the backend linked into this build.  Never null.
CompileBackend* GetCompileBackend();

}  // namespace coralnpu

#endif  // ODML_LITERT_LITERT_VENDORS_CORALNPU_COMPILER_PLUGIN_CORALNPU_COMPILE_BACKEND_H_
