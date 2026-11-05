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

// LiteRT Dispatch library for Coral NPU.
//
// This file implements the LiteRT vendor ABI (LiteRtDispatchQueryInterface and
// the versioned LiteRtDispatchInterface_V1 table) plus all host-side buffer
// bookkeeping.  Actual execution is delegated to a DispatchExecutor so the ABI
// can be built and tested without a Coral NPU toolchain.  See
// coralnpu_executor.h and coralnpu_executor_iree.cc.

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/coralnpu/common/coralnpu_common.h"
#include "litert/vendors/coralnpu/dispatch/coralnpu_executor.h"

namespace {

// Sentinel for "no buffer attached yet".  Handles are runtime-generated and
// start at 1.
constexpr LiteRtTensorBufferHandle kUnattached = 0;

}  // namespace

class LiteRtDispatchDeviceContextT {
 public:
  LiteRtStatus RegisterBuffer(LiteRtTensorBuffer buffer,
                              LiteRtTensorBufferHandle* out_handle) {
    if (buffer == nullptr || out_handle == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const LiteRtTensorBufferHandle handle = next_handle_++;
    buffers_.emplace(handle, buffer);
    *out_handle = handle;
    return kLiteRtStatusOk;
  }

  LiteRtStatus UnregisterBuffer(LiteRtTensorBufferHandle handle) {
    auto it = buffers_.find(handle);
    if (it == buffers_.end()) {
      return kLiteRtStatusErrorNotFound;
    }
    buffers_.erase(it);
    return kLiteRtStatusOk;
  }

  std::optional<LiteRtTensorBuffer> Lookup(
      LiteRtTensorBufferHandle handle) const {
    auto it = buffers_.find(handle);
    if (it == buffers_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

 private:
  std::unordered_map<LiteRtTensorBufferHandle, LiteRtTensorBuffer> buffers_;
  LiteRtTensorBufferHandle next_handle_ = 1;
};

class LiteRtDispatchInvocationContextT {
 public:
  LiteRtDispatchInvocationContextT(const LiteRtRuntimeContext* runtime_context,
                                   LiteRtDispatchDeviceContext device_context,
                                   LiteRtDispatchExecutableType exec_type)
      : runtime_context_(runtime_context),
        device_context_(device_context),
        exec_type_(exec_type) {}

  LiteRtStatus SetByteCode(const LiteRtMemBuffer& buffer) {
    if (buffer.base_addr == nullptr || buffer.size == 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    const auto* begin =
        static_cast<const uint8_t*>(buffer.base_addr) + buffer.offset;
    byte_code_.assign(begin, begin + buffer.size);
    return kLiteRtStatusOk;
  }

  LiteRtStatus SetFunctionName(const char* function_name) {
    if (function_name != nullptr) {
      function_name_ = function_name;
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus Resize(int num_inputs, int num_outputs) {
    if (num_inputs < 0 || num_outputs < 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    inputs_.assign(static_cast<size_t>(num_inputs), kUnattached);
    outputs_.assign(static_cast<size_t>(num_outputs), kUnattached);
    input_types_.assign(static_cast<size_t>(num_inputs), std::nullopt);
    output_types_.assign(static_cast<size_t>(num_outputs), std::nullopt);
    return kLiteRtStatusOk;
  }

  // Records the tensor type the runtime asked requirements for.  Executors
  // need it to describe the tensor to a backend; the runtime always asks
  // before it invokes.
  LiteRtStatus SetInputType(int index, const LiteRtRankedTensorType& type) {
    if (index < 0 || static_cast<size_t>(index) >= input_types_.size()) {
      return kLiteRtStatusErrorIndexOOB;
    }
    input_types_[static_cast<size_t>(index)] = type;
    return kLiteRtStatusOk;
  }

  LiteRtStatus SetOutputType(int index, const LiteRtRankedTensorType& type) {
    if (index < 0 || static_cast<size_t>(index) >= output_types_.size()) {
      return kLiteRtStatusErrorIndexOOB;
    }
    output_types_[static_cast<size_t>(index)] = type;
    return kLiteRtStatusOk;
  }

  LiteRtStatus AttachInput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0 || static_cast<size_t>(index) >= inputs_.size()) {
      return kLiteRtStatusErrorIndexOOB;
    }
    inputs_[index] = handle;
    return kLiteRtStatusOk;
  }

  LiteRtStatus AttachOutput(int index, LiteRtTensorBufferHandle handle) {
    if (index < 0 || static_cast<size_t>(index) >= outputs_.size()) {
      return kLiteRtStatusErrorIndexOOB;
    }
    outputs_[index] = handle;
    return kLiteRtStatusOk;
  }

  LiteRtStatus DetachInput(int index) {
    if (index < 0 || static_cast<size_t>(index) >= inputs_.size()) {
      return kLiteRtStatusErrorIndexOOB;
    }
    inputs_[index] = kUnattached;
    return kLiteRtStatusOk;
  }

  LiteRtStatus DetachOutput(int index) {
    if (index < 0 || static_cast<size_t>(index) >= outputs_.size()) {
      return kLiteRtStatusErrorIndexOOB;
    }
    outputs_[index] = kUnattached;
    return kLiteRtStatusOk;
  }

  LiteRtStatus SetSchedulingInfo(const LiteRtSchedulingInfo* info) {
    has_scheduling_info_ = info != nullptr;
    if (info != nullptr) {
      scheduling_info_ = *info;
    }
    return kLiteRtStatusOk;
  }

  LiteRtStatus Invoke() {
    if (device_context_ == nullptr) {
      return kLiteRtStatusErrorInvalidArgument;
    }

    // Materialize attached handles.  A fully attached context is required;
    // returning a specific error is better than a partial execution.
    std::vector<coralnpu::TensorArg> inputs;
    inputs.reserve(inputs_.size());
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (inputs_[i] == kUnattached) {
        CORALNPU_ERROR("input %zu was not attached", i);
        return kLiteRtStatusErrorMissingInputTensor;
      }
      auto buffer = device_context_->Lookup(inputs_[i]);
      if (!buffer.has_value()) {
        return kLiteRtStatusErrorNotFound;
      }
      coralnpu::TensorArg arg;
      arg.buffer = *buffer;
      if (input_types_[i].has_value()) {
        arg.type = *input_types_[i];
        arg.has_type = true;
      }
      inputs.push_back(arg);
    }

    std::vector<coralnpu::TensorArg> outputs;
    outputs.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
      if (outputs_[i] == kUnattached) {
        CORALNPU_ERROR("output %zu was not attached", i);
        return kLiteRtStatusErrorInvalidArgument;
      }
      auto buffer = device_context_->Lookup(outputs_[i]);
      if (!buffer.has_value()) {
        return kLiteRtStatusErrorNotFound;
      }
      coralnpu::TensorArg arg;
      arg.buffer = *buffer;
      if (output_types_[i].has_value()) {
        arg.type = *output_types_[i];
        arg.has_type = true;
      }
      outputs.push_back(arg);
    }

    LiteRtMemBuffer bytecode;
    bytecode.fd = -1;
    bytecode.base_addr = byte_code_.data();
    bytecode.offset = 0;
    bytecode.size = byte_code_.size();
    bytecode.alloc_base_file_offset = 0;

    coralnpu::DispatchExecutor* executor = coralnpu::GetDispatchExecutor();
    if (executor == nullptr) {
      CORALNPU_ERROR("no Coral NPU dispatch executor is linked in");
      return kLiteRtStatusErrorUnsupported;
    }
    return executor->Invoke(bytecode, function_name_.c_str(), inputs, outputs,
                            runtime_context_);
  }

  const LiteRtRuntimeContext* runtime_context() const {
    return runtime_context_;
  }

  LiteRtDispatchExecutableType exec_type() const { return exec_type_; }

 private:
  const LiteRtRuntimeContext* runtime_context_ = nullptr;
  LiteRtDispatchDeviceContext device_context_ = nullptr;
  LiteRtDispatchExecutableType exec_type_;
  std::vector<uint8_t> byte_code_;
  std::string function_name_;
  std::vector<LiteRtTensorBufferHandle> inputs_;
  std::vector<LiteRtTensorBufferHandle> outputs_;
  std::vector<std::optional<LiteRtRankedTensorType>> input_types_;
  std::vector<std::optional<LiteRtRankedTensorType>> output_types_;
  bool has_scheduling_info_ = false;
  LiteRtSchedulingInfo scheduling_info_{};
};

namespace {

LiteRtStatus GetTensorBufferRequirements(
    const LiteRtRuntimeContext* runtime_context,
    const LiteRtRankedTensorType& tensor_type,
    LiteRtTensorBufferRequirements* requirements) {
  if (runtime_context == nullptr || requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (runtime_context->create_tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorUnsupported;
  }
  size_t buffer_size = 0;
  CORALNPU_RETURN_IF_ERROR(coralnpu::TensorByteSize(tensor_type, &buffer_size));

  // Host memory is the portable baseline.  A hardware build can add
  // kLiteRtTensorBufferTypeDmaBuf here once the Coral NPU allocator is wired
  // through the executor.
  const LiteRtTensorBufferType buffer_types[] = {
      kLiteRtTensorBufferTypeHostMemory};
  return runtime_context->create_tensor_buffer_requirements(
      /*num_supported_tensor_buffer_types=*/1, buffer_types, buffer_size,
      /*num_strides=*/0, /*strides=*/nullptr, requirements);
}

// ---------------------------------------------------------------------------
// ABI entry points
// ---------------------------------------------------------------------------

constexpr char kVendorId[] = "Google Coral";
constexpr char kBuildId[] = "coralnpu-litert-dispatch-v0";

LiteRtStatus Initialize(const LiteRtRuntimeContext* runtime_context,
                        LiteRtEnvironment environment, LiteRtOptions options) {
  if (runtime_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  CORALNPU_INFO("Coral NPU dispatch initialized");
  return kLiteRtStatusOk;
}

LiteRtStatus GetVendorId(const char** vendor_id) {
  if (vendor_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *vendor_id = kVendorId;
  return kLiteRtStatusOk;
}

LiteRtStatus GetBuildId(const char** build_id) {
  if (build_id == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *build_id = kBuildId;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCapabilities(int* capabilities) {
  if (capabilities == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *capabilities = kLiteRtDispatchCapabilitiesBasic;
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextCreate(const LiteRtRuntimeContext* runtime_context,
                                 LiteRtOptions options,
                                 LiteRtDispatchDeviceContext* device_context) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *device_context = new LiteRtDispatchDeviceContextT();
  return kLiteRtStatusOk;
}

LiteRtStatus DeviceContextDestroy(
    LiteRtDispatchDeviceContext device_context) {
  delete device_context;
  return kLiteRtStatusOk;
}

LiteRtStatus GetInputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int input_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr ||
      tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  CORALNPU_RETURN_IF_ERROR(
      invocation_context->SetInputType(input_index, *tensor_type));
  return GetTensorBufferRequirements(invocation_context->runtime_context(),
                                     *tensor_type,
                                     tensor_buffer_requirements);
}

LiteRtStatus GetOutputRequirements(
    LiteRtDispatchInvocationContext invocation_context, int output_index,
    const LiteRtRankedTensorType* tensor_type,
    LiteRtTensorBufferRequirements* tensor_buffer_requirements) {
  if (invocation_context == nullptr || tensor_type == nullptr ||
      tensor_buffer_requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  CORALNPU_RETURN_IF_ERROR(
      invocation_context->SetOutputType(output_index, *tensor_type));
  return GetTensorBufferRequirements(invocation_context->runtime_context(),
                                     *tensor_type,
                                     tensor_buffer_requirements);
}

LiteRtStatus RegisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context, LiteRtTensorBuffer tensor_buffer,
    LiteRtTensorBufferHandle* tensor_buffer_handle) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return device_context->RegisterBuffer(tensor_buffer, tensor_buffer_handle);
}

LiteRtStatus UnregisterTensorBuffer(
    LiteRtDispatchDeviceContext device_context,
    LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (device_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return device_context->UnregisterBuffer(tensor_buffer_handle);
}

LiteRtStatus InvocationContextCreate(
    const LiteRtRuntimeContext* runtime_context,
    LiteRtDispatchDeviceContext device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    LiteRtDispatchInvocationContext* invocation_context) {
  if (runtime_context == nullptr || device_context == nullptr ||
      exec_bytecode_buffer == nullptr || invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // The compiler plugin emits a VMFB, which LiteRT models as an ML model
  // executable.  JIT handles and DSP libraries are not supported.
  if (exec_type != kLiteRtDispatchExecutableTypeMlModel) {
    CORALNPU_ERROR("unsupported executable type %d", static_cast<int>(exec_type));
    return kLiteRtStatusErrorUnsupported;
  }

  auto context = std::make_unique<LiteRtDispatchInvocationContextT>(
      runtime_context, device_context, exec_type);
  CORALNPU_RETURN_IF_ERROR(context->SetByteCode(*exec_bytecode_buffer));
  CORALNPU_RETURN_IF_ERROR(context->SetFunctionName(function_name));
  CORALNPU_RETURN_IF_ERROR(context->Resize(num_inputs, num_outputs));

  *invocation_context = context.release();
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextDestroy(
    LiteRtDispatchInvocationContext invocation_context) {
  delete invocation_context;
  return kLiteRtStatusOk;
}

LiteRtStatus InvocationContextSetSchedulingInfo(
    LiteRtDispatchInvocationContext invocation_context,
    const LiteRtSchedulingInfo* scheduling_info) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return invocation_context->SetSchedulingInfo(scheduling_info);
}

LiteRtStatus InvocationContextSetOptions(
    LiteRtDispatchInvocationContext invocation_context, LiteRtOptions options) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Per-invocation options are accepted but not interpreted yet.  Coral NPU
  // scheduling is controlled through SetSchedulingInfo.
  return kLiteRtStatusOk;
}

LiteRtStatus AttachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return invocation_context->AttachInput(graph_input_index,
                                         tensor_buffer_handle);
}

LiteRtStatus AttachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return invocation_context->AttachOutput(graph_output_index,
                                          tensor_buffer_handle);
}

LiteRtStatus DetachInput(LiteRtDispatchInvocationContext invocation_context,
                         int graph_input_index,
                         LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return invocation_context->DetachInput(graph_input_index);
}

LiteRtStatus DetachOutput(LiteRtDispatchInvocationContext invocation_context,
                          int graph_output_index,
                          LiteRtTensorBufferHandle tensor_buffer_handle) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return invocation_context->DetachOutput(graph_output_index);
}

LiteRtStatus Invoke(LiteRtDispatchInvocationContext invocation_context) {
  if (invocation_context == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return invocation_context->Invoke();
}

LiteRtStatus CheckRuntimeCompatibility(LiteRtApiVersion api_version,
                                       LiteRtEnvironmentOptions env,
                                       LiteRtOptions options) {
  // This library only uses the versioned interface tables and public C ABI.
  return kLiteRtStatusOk;
}

const LiteRtDispatchInterface_V1 kCoralNpuDispatchInterface = {
    /*.abi_header=*/
    {
        .struct_size = sizeof(LiteRtDispatchInterface_V1),
        .major_version = LITERT_DISPATCH_ABI_VERSION_MAJOR,
        .minor_version = LITERT_DISPATCH_ABI_VERSION_MINOR,
        .reserved = 0,
    },
    /*.initialize=*/Initialize,
    /*.get_vendor_id=*/GetVendorId,
    /*.get_build_id=*/GetBuildId,
    /*.get_capabilities=*/GetCapabilities,
    /*.device_context_create=*/DeviceContextCreate,
    /*.device_context_destroy=*/DeviceContextDestroy,
    /*.get_input_requirements=*/GetInputRequirements,
    /*.get_output_requirements=*/GetOutputRequirements,
    /*.register_tensor_buffer=*/RegisterTensorBuffer,
    /*.unregister_tensor_buffer=*/UnregisterTensorBuffer,
    /*.invocation_context_create=*/InvocationContextCreate,
    /*.invocation_context_destroy=*/InvocationContextDestroy,
    /*.invocation_context_set_scheduling_info=*/
    InvocationContextSetSchedulingInfo,
    /*.attach_input=*/AttachInput,
    /*.attach_output=*/AttachOutput,
    /*.detach_input=*/DetachInput,
    /*.detach_output=*/DetachOutput,
    /*.invoke=*/Invoke,
    /*.start_metrics_collection=*/nullptr,
    /*.stop_metrics_collection=*/nullptr,
    /*.get_num_metrics=*/nullptr,
    /*.get_metric=*/nullptr,
    /*.destroy_metrics=*/nullptr,
    /*.check_runtime_compatibility=*/CheckRuntimeCompatibility,
    /*.invocation_context_set_options=*/InvocationContextSetOptions,
    /*.get_hooks=*/nullptr,
    /*.attach_edge_buffer=*/nullptr,
    /*.detach_edge_buffer=*/nullptr,
};

}  // namespace

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtDispatchQueryInterface(
    LiteRtDispatchInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major < 1) {
    return kLiteRtStatusErrorUnsupportedRuntimeVersion;
  }
  if (interface_id == kLiteRtInterfaceBasic) {
    *out_interface = &kCoralNpuDispatchInterface;
    return kLiteRtStatusOk;
  }
  return kLiteRtStatusErrorUnsupported;
}
