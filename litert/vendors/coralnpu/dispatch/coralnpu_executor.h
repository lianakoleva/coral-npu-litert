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

#ifndef ODML_LITERT_LITERT_VENDORS_CORALNPU_DISPATCH_CORALNPU_EXECUTOR_H_
#define ODML_LITERT_LITERT_VENDORS_CORALNPU_DISPATCH_CORALNPU_EXECUTOR_H_

#include <vector>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/vendors/c/litert_dispatch.h"

namespace coralnpu {

// One tensor an invocation reads or writes, in graph order.
//
// The dispatch library only ever holds opaque LiteRtTensorBuffer handles; the
// host address comes from runtime_context->get_tensor_buffer_host_memory.  The
// type is recorded from get_{input,output}_requirements, which the runtime
// calls before invoking.  Backends that hand tensors to another process need
// the shape and element type to describe them, so a missing type is tracked
// explicitly instead of being guessed.
struct TensorArg {
  LiteRtTensorBuffer buffer = nullptr;
  LiteRtRankedTensorType type{};
  bool has_type = false;
};

// Execution seam between the LiteRT dispatch ABI and the Coral NPU runtime.
//
// `coralnpu_dispatch.cc` implements the ABI, buffer bookkeeping, and argument
// validation.  The concrete executor is selected at link time:
//
//   coralnpu_executor_unavailable.cc  -> returns Unsupported with a build hint
//   coralnpu_executor_command.cc      -> runs iree-run-module as a subprocess
//                                        (CORALNPU_WITH_COMMAND_EXECUTOR=ON)
//   coralnpu_executor_iree.cc         -> links IREE in-process
//                                        (CORALNPU_WITH_IREE_RUNTIME=ON)
//
// Keeping the seam narrow means the ABI can be conformance-tested on a host
// with no Coral NPU toolchain installed.
class DispatchExecutor {
 public:
  virtual ~DispatchExecutor() = default;

  // Executes `function_name` from the bytecode module `bytecode`.
  //
  // `inputs` and `outputs` are in graph order.  Implementations must obtain
  // host pointers through `runtime_context` rather than assuming a particular
  // memory type, and must not outlive the call.
  virtual LiteRtStatus Invoke(const LiteRtMemBuffer& bytecode,
                              const char* function_name,
                              const std::vector<TensorArg>& inputs,
                              const std::vector<TensorArg>& outputs,
                              const LiteRtRuntimeContext* runtime_context) = 0;
};

// Implemented by exactly one executor translation unit linked into the
// dispatch library.
DispatchExecutor* GetDispatchExecutor();

}  // namespace coralnpu

#endif  // ODML_LITERT_LITERT_VENDORS_CORALNPU_DISPATCH_CORALNPU_EXECUTOR_H_
