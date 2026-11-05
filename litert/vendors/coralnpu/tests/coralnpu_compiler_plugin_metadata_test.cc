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

// ABI-level test for the Coral NPU compiler plugin.
//
// Everything here is testable without a LiteRT runtime: the plugin metadata,
// the interface table layout, the compatibility gate, and argument validation.
// The graph-walking paths (partition/compile) need a LiteRtCompilerContext and
// are covered end to end by tests/test_litert_plugin.py on a host with the
// runtime available.

#include <cstdint>
#include <cstring>

#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/coralnpu/tests/coralnpu_test_util.h"

namespace {

constexpr char kExpectedSocModel[] = "coralnpu";
constexpr char kExpectedManufacturer[] = "Google";

}  // namespace

int main() {
  const LiteRtCompilerPluginInterface_V1* iface = nullptr;

  // --- interface negotiation ------------------------------------------------
  CORALNPU_EXPECT_EQ(
      LiteRtCompilerPluginQueryInterface(
          kLiteRtCompilerPluginInterfaceBasic, {1, 0, 0},
          reinterpret_cast<LiteRtInterface*>(&iface)),
      kLiteRtStatusOk);
  CORALNPU_EXPECT(iface != nullptr);
  if (iface == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  CORALNPU_EXPECT_EQ(iface->abi_header.struct_size,
                     sizeof(LiteRtCompilerPluginInterface_V1));
  CORALNPU_EXPECT_EQ(iface->abi_header.major_version,
                     LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR);
  CORALNPU_EXPECT_EQ(iface->abi_header.minor_version,
                     LITERT_COMPILER_PLUGIN_ABI_VERSION_MINOR);

  {
    const LiteRtCompilerPluginInterface_V1* old_iface = nullptr;
    CORALNPU_EXPECT_EQ(
        LiteRtCompilerPluginQueryInterface(
            kLiteRtCompilerPluginInterfaceBasic, {0, 1, 0},
            reinterpret_cast<LiteRtInterface*>(&old_iface)),
        kLiteRtStatusErrorUnsupportedRuntimeVersion);
    CORALNPU_EXPECT(old_iface == nullptr);
  }

  // --- metadata -------------------------------------------------------------
  LiteRtApiVersion version{};
  CORALNPU_EXPECT_EQ(iface->get_compiler_plugin_version(&version),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(version.major, LITERT_API_VERSION_MAJOR);

  CORALNPU_EXPECT_STREQ(iface->get_compiler_plugin_soc_manufacturer(),
                        kExpectedManufacturer);

  // --- plugin lifetime ------------------------------------------------------
  // A null compiler context cannot support graph inspection.
  {
    LiteRtCompilerPlugin plugin = nullptr;
    CORALNPU_EXPECT_EQ(iface->create_compiler_plugin(nullptr, &plugin, nullptr,
                                                     nullptr),
                       kLiteRtStatusErrorInvalidArgument);
    CORALNPU_EXPECT(plugin == nullptr);
  }

  // A zeroed context is enough for the metadata calls; nothing dereferences it.
  LiteRtCompilerContext compiler_context{};
  compiler_context.abi_header.struct_size = sizeof(LiteRtCompilerContext);
  compiler_context.abi_header.major_version = 1;

  LiteRtCompilerPlugin plugin = nullptr;
  CORALNPU_EXPECT_EQ(
      iface->create_compiler_plugin(&compiler_context, &plugin, nullptr, nullptr),
      kLiteRtStatusOk);
  CORALNPU_EXPECT(plugin != nullptr);
  if (plugin == nullptr) {
    return CORALNPU_TEST_MAIN_RESULT();
  }

  // --- supported hardware and SoC models ------------------------------------
  LiteRtHwAccelerators hardware = kLiteRtHwAcceleratorNone;
  CORALNPU_EXPECT_EQ(iface->get_compiler_plugin_supported_hardware(
                         plugin, &hardware),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(hardware, kLiteRtHwAcceleratorNpu);

  LiteRtParamIndex num_models = 0;
  CORALNPU_EXPECT_EQ(
      iface->get_num_compiler_plugin_supported_models(plugin, &num_models),
      kLiteRtStatusOk);
  CORALNPU_EXPECT_EQ(num_models, static_cast<LiteRtParamIndex>(1));

  const char* soc_model = nullptr;
  CORALNPU_EXPECT_EQ(
      iface->get_compiler_plugin_supported_soc_model(plugin, 0, &soc_model),
      kLiteRtStatusOk);
  CORALNPU_EXPECT_STREQ(soc_model, kExpectedSocModel);

  // Asking for a model that does not exist must not run off the end.
  CORALNPU_EXPECT_EQ(
      iface->get_compiler_plugin_supported_soc_model(plugin, 1, &soc_model),
      kLiteRtStatusErrorUnsupported);

  const char* sdk_version = nullptr;
  CORALNPU_EXPECT_EQ(iface->get_compiler_plugin_sdk_version(plugin, &sdk_version),
                     kLiteRtStatusOk);
  CORALNPU_EXPECT(sdk_version != nullptr && sdk_version[0] != '\0');

  // --- compatibility gate ---------------------------------------------------
  CORALNPU_EXPECT_EQ(iface->check_compiler_compatibility(
                         version, plugin, nullptr, nullptr, kExpectedSocModel),
                     kLiteRtStatusOk);
  // Null means "let the plugin choose".
  CORALNPU_EXPECT_EQ(iface->check_compiler_compatibility(
                         version, plugin, nullptr, nullptr, nullptr),
                     kLiteRtStatusOk);
  // A different SoC must be refused rather than silently mistargeted.
  CORALNPU_EXPECT_EQ(iface->check_compiler_compatibility(
                         version, plugin, nullptr, nullptr, "some-other-soc"),
                     kLiteRtStatusErrorUnsupportedCompilerVersion);

  // --- transformations ------------------------------------------------------
  // Coral NPU does its rewriting inside the compiler, not as LiteRT passes.
  {
    LiteRtTransformation* transformations = nullptr;
    LiteRtParamIndex num_patterns = 12345;
    CORALNPU_EXPECT_EQ(iface->register_all_transformations(
                           plugin, &transformations, &num_patterns),
                       kLiteRtStatusOk);
    CORALNPU_EXPECT_EQ(num_patterns, static_cast<LiteRtParamIndex>(0));
    CORALNPU_EXPECT(transformations == nullptr);
  }

  // --- argument validation --------------------------------------------------
  // Each of these dereferences the table entry, so a null entry in the
  // initializer above would crash here instead of failing cleanly.
  CORALNPU_EXPECT_EQ(iface->compiler_plugin_partition(plugin, kExpectedSocModel,
                                                      nullptr, nullptr),
                     kLiteRtStatusErrorInvalidArgument);
  CORALNPU_EXPECT_EQ(iface->compiler_plugin_compile(plugin, kExpectedSocModel,
                                                    nullptr, nullptr),
                     kLiteRtStatusErrorInvalidArgument);

  {
    LiteRtCompiledResult result = nullptr;
    CORALNPU_EXPECT_EQ(
        iface->compiler_plugin_compile(plugin, kExpectedSocModel, nullptr,
                                       &result),
        kLiteRtStatusErrorInvalidArgument);
    CORALNPU_EXPECT(result == nullptr);
  }

  // Compiled-result accessors must reject null results.
  {
    LiteRtParamIndex count = 0;
    CORALNPU_EXPECT_EQ(iface->get_num_compiled_result_calls(nullptr, &count),
                       kLiteRtStatusErrorInvalidArgument);
    CORALNPU_EXPECT_EQ(
        iface->get_compiled_result_num_byte_code(nullptr, &count),
        kLiteRtStatusErrorInvalidArgument);
    const void* code = nullptr;
    size_t code_size = 0;
    CORALNPU_EXPECT_EQ(
        iface->get_compiled_result_byte_code(nullptr, 0, &code, &code_size),
        kLiteRtStatusErrorInvalidArgument);
    LiteRtParamIndex byte_code_idx = 0;
    CORALNPU_EXPECT_EQ(iface->get_compiled_result_call_info(
                           nullptr, 0, &code, &code_size, &byte_code_idx),
                       kLiteRtStatusErrorInvalidArgument);
  }

  {
    LiteRtJitExecutable handle = reinterpret_cast<LiteRtJitExecutable>(
        static_cast<uintptr_t>(0x1));
    CORALNPU_EXPECT_EQ(iface->get_compiled_result_handle(nullptr, 0, &handle),
                       kLiteRtStatusErrorInvalidArgument);
  }

  iface->destroy_compiler_plugin(plugin);

  return CORALNPU_TEST_MAIN_RESULT();
}
