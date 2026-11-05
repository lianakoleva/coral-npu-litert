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

// Loads the built dispatch library the way the LiteRT runtime does and drives
// it through the public ABI.
//
// Every other test links the vendor objects directly, which cannot catch the
// problems that only appear in a shared object: a missing SONAME dependency, an
// ABI table entry that is null after initialisation, a symbol that is not
// exported, or a static initialiser that never runs.  This test dlopen()s the
// real artefact and resolves the single entry point by name.
//
// Usage:
//   coralnpu_dlopen_smoke_test <libLiteRtDispatch_CoralNpu.so> [module.vmfb]
//
// With a VMFB the test also performs a real invocation using whatever
// `iree-run-module` CORALNPU_RUN_MODULE points at (or the one on $PATH).  That
// is the board check: a compiled Coral NPU module, run through the shipped
// dispatch library, producing the expected numbers.

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/c/litert_dispatch_api.h"
#include "litert/vendors/coralnpu/tests/coralnpu_test_util.h"

namespace {

// Two host buffers; the runtime context maps handles to these addresses.
std::vector<float> g_input;
std::vector<float> g_output;

LiteRtStatus FakeCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LiteRtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, int num_strides, const uint32_t* strides,
    LiteRtTensorBufferRequirements* requirements) {
  if (requirements == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *requirements =
      reinterpret_cast<LiteRtTensorBufferRequirements>(static_cast<uintptr_t>(1));
  return kLiteRtStatusOk;
}

LiteRtStatus FakeGetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                           void** host_memory_addr) {
  if (host_memory_addr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto index = reinterpret_cast<uintptr_t>(tensor_buffer);
  if (index == 1) {
    *host_memory_addr = g_input.data();
    return kLiteRtStatusOk;
  }
  if (index == 2) {
    *host_memory_addr = g_output.data();
    return kLiteRtStatusOk;
  }
  return kLiteRtStatusErrorNotFound;
}

LiteRtRankedTensorType MakeTensorType(const std::vector<int32_t>& dims) {
  LiteRtRankedTensorType type{};
  type.element_type = kLiteRtElementTypeFloat32;
  type.layout.rank = static_cast<uint32_t>(dims.size());
  type.layout.has_strides = false;
  for (size_t i = 0; i < dims.size(); ++i) {
    type.layout.dimensions[i] = dims[i];
  }
  return type;
}

bool ReadFile(const std::string& path, std::vector<uint8_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  std::fseek(f, 0, SEEK_END);
  const long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0) {
    std::fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  const bool ok = size == 0 ||
                  std::fread(out->data(), 1, static_cast<size_t>(size), f) ==
                      static_cast<size_t>(size);
  std::fclose(f);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <libLiteRtDispatch_CoralNpu.so> [module.vmfb]\n",
                 argv[0]);
    return 2;
  }
  const std::string library_path = argv[1];
  const std::string module_path = argc > 2 ? argv[2] : "";

  // --- load the artefact ----------------------------------------------------
  void* handle = ::dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::fprintf(stderr, "FAIL dlopen(%s): %s\n", library_path.c_str(),
                 ::dlerror());
    return 1;
  }

  using QueryInterfaceFn = LiteRtStatus (*)(
      LiteRtDispatchInterfaceId, LiteRtApiVersion, LiteRtInterface*);
  auto* query = reinterpret_cast<QueryInterfaceFn>(
      ::dlsym(handle, "LiteRtDispatchQueryInterface"));
  CORALNPU_EXPECT(query != nullptr);
  if (query == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  const LiteRtDispatchInterface_V1* iface = nullptr;
  CORALNPU_EXPECT_EQ(
      query(kLiteRtInterfaceBasic, {1, 0, 0},
            reinterpret_cast<LiteRtInterface*>(&iface)),
      kLiteRtStatusOk);
  CORALNPU_EXPECT(iface != nullptr);
  if (iface == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  // Every entry in the table must be non-null: the runtime calls them through
  // the struct without checking.
  CORALNPU_EXPECT(iface->initialize != nullptr);
  CORALNPU_EXPECT(iface->get_vendor_id != nullptr);
  CORALNPU_EXPECT(iface->get_build_id != nullptr);
  CORALNPU_EXPECT(iface->get_capabilities != nullptr);
  CORALNPU_EXPECT(iface->device_context_create != nullptr);
  CORALNPU_EXPECT(iface->device_context_destroy != nullptr);
  CORALNPU_EXPECT(iface->get_input_requirements != nullptr);
  CORALNPU_EXPECT(iface->get_output_requirements != nullptr);
  CORALNPU_EXPECT(iface->register_tensor_buffer != nullptr);
  CORALNPU_EXPECT(iface->unregister_tensor_buffer != nullptr);
  CORALNPU_EXPECT(iface->invocation_context_create != nullptr);
  CORALNPU_EXPECT(iface->invocation_context_destroy != nullptr);
  CORALNPU_EXPECT(iface->attach_input != nullptr);
  CORALNPU_EXPECT(iface->attach_output != nullptr);
  CORALNPU_EXPECT(iface->detach_input != nullptr);
  CORALNPU_EXPECT(iface->detach_output != nullptr);
  CORALNPU_EXPECT(iface->invoke != nullptr);

  CORALNPU_EXPECT_EQ(iface->abi_header.struct_size,
                     sizeof(LiteRtDispatchInterface_V1));

  const char* vendor_id = nullptr;
  CORALNPU_EXPECT_EQ(iface->get_vendor_id(&vendor_id), kLiteRtStatusOk);
  CORALNPU_EXPECT_STREQ(vendor_id, "Google Coral");

  int capabilities = 0;
  CORALNPU_EXPECT_EQ(iface->get_capabilities(&capabilities), kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(capabilities,
                     static_cast<int>(kLiteRtDispatchCapabilitiesBasic));

  // --- fake runtime context -------------------------------------------------
  LiteRtRuntimeContext runtime_context{};
  runtime_context.abi_header.struct_size = sizeof(LiteRtRuntimeContext);
  runtime_context.abi_header.major_version = 1;
  runtime_context.create_tensor_buffer_requirements =
      FakeCreateTensorBufferRequirements;
  runtime_context.get_tensor_buffer_host_memory =
      FakeGetTensorBufferHostMemory;

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

  // --- buffers --------------------------------------------------------------
  const std::vector<int32_t> dims = {2, 3};
  const LiteRtRankedTensorType tensor_type = MakeTensorType(dims);
  g_input = {1.5f, -2.5f, 3.0f, -4.0f, 0.5f, -0.25f};
  g_output.assign(6, 0.0f);

  auto input_buffer =
      reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(1));
  auto output_buffer =
      reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(2));

  LiteRtTensorBufferHandle input_handle = 0;
  LiteRtTensorBufferHandle output_handle = 0;
  CORALNPU_EXPECT_EQ(
      iface->register_tensor_buffer(device_context, input_buffer, &input_handle),
      kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->register_tensor_buffer(
                         device_context, output_buffer, &output_handle),
                     kLiteRtStatusOk);

  // --- module ---------------------------------------------------------------
  std::vector<uint8_t> module;
  if (!module_path.empty()) {
    CORALNPU_EXPECT(ReadFile(module_path, &module));
    CORALNPU_EXPECT(!module.empty());
  } else {
    // Metadata-only run: a stub is enough to exercise context creation.
    module = {'V', 'M', 'F', 'B'};
  }

  LiteRtMemBuffer bytecode{};
  bytecode.fd = -1;
  bytecode.base_addr = module.data();
  bytecode.offset = 0;
  bytecode.size = module.size();
  bytecode.alloc_base_file_offset = 0;

  LiteRtDispatchInvocationContext invocation = nullptr;
  CORALNPU_EXPECT_EQ(iface->invocation_context_create(
                         &runtime_context, device_context,
                         kLiteRtDispatchExecutableTypeMlModel, &bytecode,
                         "main", /*num_inputs=*/1, /*num_outputs=*/1,
                         &invocation),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT(invocation != nullptr);
  if (invocation == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  // The runtime always asks for requirements before allocating buffers.
  LiteRtTensorBufferRequirements requirements = nullptr;
  CORALNPU_EXPECT_EQ(iface->get_input_requirements(invocation, 0, &tensor_type,
                                                   &requirements),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->get_output_requirements(invocation, 0, &tensor_type,
                                                    &requirements),
                     kLiteRtStatusOk);

  CORALNPU_EXPECT_EQ(
      iface->attach_input(invocation, 0, input_handle), kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(
      iface->attach_output(invocation, 0, output_handle), kLiteRtStatusOk);

  // --- invoke ---------------------------------------------------------------
  const LiteRtStatus invoke_status = iface->invoke(invocation);
  if (module_path.empty()) {
    // No module and no tool: the executor is expected to refuse, not crash.
    CORALNPU_EXPECT(invoke_status != kLiteRtStatusOk);
    std::printf("metadata-only run: invoke returned %d as expected\n",
                static_cast<int>(invoke_status));
  } else {
    CORALNPU_EXPECT_EQ(invoke_status, kLiteRtStatusOk);
    if (invoke_status == kLiteRtStatusOk) {
      // abs() of the input: the test module is a single tosa.abs.
      bool matches = true;
      for (size_t i = 0; i < g_input.size(); ++i) {
        const float expected = std::fabs(g_input[i]);
        if (std::fabs(g_output[i] - expected) > 1e-6f) {
          std::fprintf(stderr, "FAIL output[%zu] = %f, expected %f\n", i,
                       g_output[i], expected);
          matches = false;
        }
      }
      CORALNPU_EXPECT(matches);
      std::printf("real invocation: %zu outputs match tosa.abs\n",
                  g_output.size());
    }
  }

  // --- teardown -------------------------------------------------------------
  iface->invocation_context_destroy(invocation);
  CORALNPU_EXPECT_EQ(iface->unregister_tensor_buffer(device_context, input_handle),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(
      iface->unregister_tensor_buffer(device_context, output_handle),
      kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(iface->device_context_destroy(device_context),
                     kLiteRtStatusOk);

  ::dlclose(handle);
  return CORALNPU_TEST_MAIN_RESULT();
}
