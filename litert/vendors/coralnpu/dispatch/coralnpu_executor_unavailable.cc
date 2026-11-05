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

// Default DispatchExecutor: build the ABI, link no hardware runtime.
//
// This is the translation unit that makes `libLiteRtDispatch_CoralNpu.so`
// buildable on any machine.  It exists so that the dispatch ABI, the buffer
// registry, and the argument validation in coralnpu_dispatch.cc can be
// conformance-tested without an IREE runtime or a Coral NPU toolchain.
//
// Every Invoke() fails with kLiteRtStatusErrorUnsupported and a message that
// says how to get a working build, which is far easier to debug than a link
// failure or a silent no-op.

#include <vector>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/coralnpu/common/coralnpu_common.h"
#include "litert/vendors/coralnpu/dispatch/coralnpu_executor.h"

namespace coralnpu {
namespace {

class UnavailableExecutor final : public DispatchExecutor {
 public:
  LiteRtStatus Invoke(const LiteRtMemBuffer& bytecode,
                      const char* function_name,
                      const std::vector<TensorArg>& inputs,
                      const std::vector<TensorArg>& outputs,
                      const LiteRtRuntimeContext* runtime_context) override {
    CORALNPU_ERROR(
        "no Coral NPU execution backend is linked into this build: cannot run "
        "'%s' (%zu inputs, %zu outputs, %zu byte module)",
        function_name != nullptr ? function_name : "<unnamed>", inputs.size(),
        outputs.size(), bytecode.size);
    CORALNPU_ERROR(
        "rebuild with -DCORALNPU_WITH_COMMAND_EXECUTOR=ON to run through "
        "iree-run-module, or -DCORALNPU_WITH_IREE_RUNTIME=ON to link the IREE "
        "runtime and the Coral NPU HAL driver.");
    return kLiteRtStatusErrorUnsupported;
  }
};

}  // namespace

DispatchExecutor* GetDispatchExecutor() {
  static UnavailableExecutor executor;
  return &executor;
}

}  // namespace coralnpu
