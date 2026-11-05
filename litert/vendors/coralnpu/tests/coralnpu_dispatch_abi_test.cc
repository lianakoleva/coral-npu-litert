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

// Conformance test for the Coral NPU LiteRT dispatch ABI.
//
// Runs without a Coral NPU toolchain by linking the default (unavailable)
// executor.  Everything the runtime does before invoking -- interface
// negotiation, buffer registration, invocation setup, requirement queries --
// is real, and the executor failing with kLiteRtStatusErrorUnsupported is the
// documented behaviour of that build.

#include <cstdint>
#include <cstring>
#include <vector>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/coralnpu/tests/coralnpu_test_util.h"

namespace {

int g_requirements_calls = 0;
int g_last_num_buffer_types = 0;
size_t g_last_buffer_size = 0;
LiteRtTensorBufferType g_last_buffer_type = kLiteRtTensorBufferTypeUnknown;

LiteRtStatus FakeCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LiteRtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, int num_strides, const uint32_t* strides,
    LiteRtTensorBufferRequirements* requirements) {
  ++g_requirements_calls;
  g_last_num_buffer_types = num_supported_tensor_buffer_types;
  if (num_supported_tensor_buffer_types > 0 && supported_tensor_buffer_types) {
    g_last_buffer_type = supported_tensor_buffer_types[0];
  }
  g_last_buffer_size = buffer_size;
  if (requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // The dispatch library only needs a non-null handle to hand back; the
  // runtime-owned object is opaque to it.
  *requirements =
      reinterpret_cast<LiteRtTensorBufferRequirements>(static_cast<uintptr_t>(1));
  return kLiteRtStatusOk;
}

LiteRtRankedTensorType MakeTensorType(std::vector<int32_t>& dims,
                                      LiteRtElementType element_type) {
  LiteRtRankedTensorType type{};
  type.element_type = element_type;
  type.layout.rank = static_cast<uint32_t>(dims.size());
  type.layout.has_strides = false;
  for (size_t i = 0; i < dims.size(); ++i) {
    type.layout.dimensions[i] = dims[i];
  }
  return type;
}

}  // namespace

int main() {
  const LiteRtDispatchInterface_V1* iface = nullptr;

  // --- interface negotiation ------------------------------------------------
  CORALNPU_EXPECT_EQ(
      LiteRtDispatchQueryInterface(kLiteRtInterfaceBasic, {1, 0, 0},
                                   reinterpret_cast<LiteRtInterface*>(&iface)),
      kLiteRtStatusOk);
  CORALNPU_EXPECT(iface != nullptr);
  if (iface == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  CORALNPU_EXPECT_EQ(iface->abi_header.struct_size,
                     sizeof(LiteRtDispatchInterface_V1));
  CORALNPU_EXPECT_EQ(iface->abi_header.major_version,
                     LITERT_DISPATCH_ABI_VERSION_MAJOR);

  // A pre-1.0 runtime must be rejected rather than silently accepted.
  {
    const LiteRtDispatchInterface_V1* old_iface = nullptr;
    CORALNPU_EXPECT_EQ(
        LiteRtDispatchQueryInterface(kLiteRtInterfaceBasic, {0, 9, 0},
                                     reinterpret_cast<LiteRtInterface*>(
                                         &old_iface)),
        kLiteRtStatusErrorUnsupportedRuntimeVersion);
    CORALNPU_EXPECT(old_iface == nullptr);
  }

  // Unknown interface ids are unsupported, not a crash.
  {
    const LiteRtDispatchInterface_V1* other = nullptr;
    CORALNPU_EXPECT_EQ(
        LiteRtDispatchQueryInterface(static_cast<LiteRtDispatchInterfaceId>(77),
                                     {1, 0, 0},
                                     reinterpret_cast<LiteRtInterface*>(&other)),
        kLiteRtStatusErrorUnsupported);
  }

  // --- metadata -------------------------------------------------------------
  const char* vendor_id = nullptr;
  CORALNPU_EXPECT_EQ(iface->get_vendor_id(&vendor_id), kLiteRtStatusOk);
  CORALNPU_EXPECT_STREQ(vendor_id, "Google Coral");

  const char* build_id = nullptr;
  CORALNPU_EXPECT_EQ(iface->get_build_id(&build_id), kLiteRtStatusOk);
  CORALNPU_EXPECT(build_id != nullptr && build_id[0] != '\0');

  int capabilities = 0;
  CORALNPU_EXPECT_EQ(iface->get_capabilities(&capabilities), kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(capabilities, static_cast<int>(kLiteRtDispatchCapabilitiesBasic));

  // --- runtime context + device context -------------------------------------
  LiteRtRuntimeContext runtime_context{};
  runtime_context.abi_header.struct_size = sizeof(LiteRtRuntimeContext);
  runtime_context.abi_header.major_version = 1;
  runtime_context.create_tensor_buffer_requirements =
      FakeCreateTensorBufferRequirements;

  CORALNPU_EXPECT_EQ(iface->initialize(&runtime_context, nullptr, nullptr),
                     kLiteRtStatusOk);

  LiteRtDispatchDeviceContext device_context = nullptr;
  CORALNPU_EXPECT_EQ(iface->device_context_create(&runtime_context, nullptr,
                                                  &device_context),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT(device_context != nullptr);
  if (device_context == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  // Two host buffers stand in for graph input and output.
  auto input_buffer = reinterpret_cast<LiteRtTensorBuffer>(
      static_cast<uintptr_t>(0x1000));
  auto output_buffer = reinterpret_cast<LiteRtTensorBuffer>(
      static_cast<uintptr_t>(0x2000));

  LiteRtTensorBufferHandle input_handle = 0;
  LiteRtTensorBufferHandle output_handle = 0;
  CORALNPU_EXPECT_EQ(iface->register_tensor_buffer(device_context, input_buffer,
                                                   &input_handle),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->register_tensor_buffer(
                         device_context, output_buffer, &output_handle),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT(input_handle != 0);
  CORALNPU_EXPECT(input_handle != output_handle);

  // --- invocation context ---------------------------------------------------
  // A VMFB would start with a FlatBuffer header; the ABI layer treats the
  // module as opaque bytes, so a recognisable stub is enough here.
  const uint8_t kFakeByteCode[] = {0x1c, 0x00, 0x00, 0x00, 'V', 'M', 'F', 'B'};
  LiteRtMemBuffer bytecode{};
  bytecode.fd = -1;
  bytecode.base_addr = kFakeByteCode;
  bytecode.offset = 0;
  bytecode.size = sizeof(kFakeByteCode);
  bytecode.alloc_base_file_offset = 0;

  LiteRtDispatchInvocationContext invocation_context = nullptr;
  CORALNPU_EXPECT_EQ(iface->invocation_context_create(
                         &runtime_context, device_context,
                         kLiteRtDispatchExecutableTypeMlModel, &bytecode,
                         "partition_0", /*num_inputs=*/1, /*num_outputs=*/1,
                         &invocation_context),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT(invocation_context != nullptr);
  if (invocation_context == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  // JIT handles and DSP libraries are not a Coral NPU artefact.
  {
    LiteRtDispatchInvocationContext rejected = nullptr;
    CORALNPU_EXPECT_EQ(iface->invocation_context_create(
                           &runtime_context, device_context,
                           kLiteRtDispatchExecutableTypeJitHandle, &bytecode,
                           "partition_0", 1, 1, &rejected),
                       kLiteRtStatusErrorUnsupported);
    CORALNPU_EXPECT(rejected == nullptr);
  }

  // --- buffer requirements --------------------------------------------------
  std::vector<int32_t> dims = {2, 3};
  const LiteRtRankedTensorType float_type =
      MakeTensorType(dims, kLiteRtElementTypeFloat32);

  // Invoking before attaching must be reported, not executed.
  CORALNPU_EXPECT_EQ(iface->invoke(invocation_context),
                     kLiteRtStatusErrorMissingInputTensor);

  g_requirements_calls = 0;
  LiteRtTensorBufferRequirements requirements = nullptr;
  CORALNPU_EXPECT_EQ(iface->get_input_requirements(invocation_context,
                                                   /*input_index=*/0,
                                                   &float_type, &requirements),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(g_requirements_calls, 1);
  CORALNPU_EXPECT_EQ(g_last_buffer_size, static_cast<size_t>(2 * 3 * 4));
  CORALNPU_EXPECT_EQ(g_last_num_buffer_types, 1);
  CORALNPU_EXPECT_EQ(g_last_buffer_type, kLiteRtTensorBufferTypeHostMemory);

  // Dynamic shapes are a compile-time concern; the ABI rejects them loudly.
  {
    std::vector<int32_t> dynamic_dims = {2, -1};
    const LiteRtRankedTensorType dynamic_type =
        MakeTensorType(dynamic_dims, kLiteRtElementTypeFloat32);
    LiteRtTensorBufferRequirements dynamic_requirements = nullptr;
    CORALNPU_EXPECT_EQ(iface->get_output_requirements(
                           invocation_context, 0, &dynamic_type,
                           &dynamic_requirements),
                       kLiteRtStatusErrorUnsupported);
  }

  // Sub-byte types must round up to whole bytes.
  {
    std::vector<int32_t> int4_dims = {3};
    const LiteRtRankedTensorType int4_type =
        MakeTensorType(int4_dims, kLiteRtElementTypeInt4);
    LiteRtTensorBufferRequirements int4_requirements = nullptr;
    CORALNPU_EXPECT_EQ(iface->get_output_requirements(invocation_context, 0,
                                                      &int4_type,
                                                      &int4_requirements),
                       kLiteRtStatusOk);
    CORALNPU_EXPECT_EQ(g_last_buffer_size, static_cast<size_t>(2));
  }

  // --- attach / detach ------------------------------------------------------
  CORALNPU_EXPECT_EQ(
      iface->attach_input(invocation_context, 0, input_handle), kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->attach_output(invocation_context, 0, output_handle),
                     kLiteRtStatusOk);

  // Out-of-range indices are caught before they reach the executor.
  CORALNPU_EXPECT_EQ(iface->attach_input(invocation_context, 7, input_handle),
                     kLiteRtStatusErrorIndexOOB);
  CORALNPU_EXPECT_EQ(iface->attach_output(invocation_context, -1, output_handle),
                     kLiteRtStatusErrorIndexOOB);

  // Detaching then invoking must report the missing input.
  CORALNPU_EXPECT_EQ(
      iface->detach_input(invocation_context, 0, input_handle), kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->invoke(invocation_context),
                     kLiteRtStatusErrorMissingInputTensor);
  CORALNPU_EXPECT_EQ(
      iface->attach_input(invocation_context, 0, input_handle), kLiteRtStatusOk);

  // --- invoke ---------------------------------------------------------------
  // With every buffer attached the call reaches the executor, which is the
  // unavailable one in this build.
  CORALNPU_EXPECT_EQ(iface->invoke(invocation_context),
                     kLiteRtStatusErrorUnsupported);

  // A handle that was never registered must not be dereferenced.
  {
    LiteRtTensorBufferHandle stale = 0;
    CORALNPU_EXPECT_EQ(iface->unregister_tensor_buffer(device_context,
                                                       input_handle),
                       kLiteRtStatusOk);
    // Re-registering yields a fresh handle; the old one is now unknown.
    CORALNPU_EXPECT_EQ(iface->register_tensor_buffer(device_context, input_buffer,
                                                     &stale),
                       kLiteRtStatusOk);
    CORALNPU_EXPECT(stale != input_handle);
    CORALNPU_EXPECT_EQ(iface->unregister_tensor_buffer(device_context,
                                                       input_handle),
                       kLiteRtStatusErrorNotFound);
    input_handle = stale;
  }

  // --- teardown -------------------------------------------------------------
  iface->invocation_context_destroy(invocation_context);
  CORALNPU_EXPECT_EQ(iface->unregister_tensor_buffer(device_context, input_handle),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(
      iface->unregister_tensor_buffer(device_context, output_handle),
      kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->device_context_destroy(device_context),
                     kLiteRtStatusOk);

  // Null arguments are rejected at every entry point.
  CORALNPU_EXPECT_EQ(iface->get_vendor_id(nullptr),
                     kLiteRtStatusErrorInvalidArgument);
  CORALNPU_EXPECT_EQ(iface->device_context_create(&runtime_context, nullptr,
                                                  nullptr),
                     kLiteRtStatusErrorInvalidArgument);
  CORALNPU_EXPECT_EQ(iface->invoke(nullptr), kLiteRtStatusErrorInvalidArgument);

  return CORALNPU_TEST_MAIN_RESULT();
}
