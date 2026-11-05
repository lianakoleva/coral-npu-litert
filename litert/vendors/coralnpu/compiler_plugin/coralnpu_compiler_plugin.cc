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

// LiteRT Compiler Plugin for Coral NPU.
//
// Shape of the implementation:
//
//   partition()  claims ops for the NPU.  The first vertical slice claims the
//                whole subgraph as partition 0, which is what the Coral NPU
//                toolchain expects for a single-device graph: IREE does its own
//                host/NPU partitioning inside the compiled module.
//   compile()    serializes the selected partitions into one LiteRT model whose
//                subgraphs carry stable signature names, hands it to the
//                CompileBackend, and returns one VMFB plus one call per
//                partition.
//
// The ABI itself is deliberately thin.  All policy lives behind the two seams
// (CompileBackend for compilation), so the ABI can be exercised by tests with a
// fake backend while the production backend shells out to
// scripts/compile_litert.py.

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/internal/litert_abi_header.h"
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/c/litert_op_code.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/c/litert_compiler_plugin_api.h"
#include "litert/vendors/coralnpu/common/coralnpu_common.h"
#include "litert/vendors/coralnpu/compiler_plugin/coralnpu_compile_backend.h"

namespace {

constexpr char kSocManufacturer[] = "Google";
constexpr char kSocModel[] = "coralnpu";
constexpr char kSdkVersion[] = "coralnpu-litert-0.1.0";

// Name of the function exported from the VMFB for partition `index`.
std::string PartitionFunctionName(int index) {
  return "partition_" + std::to_string(index);
}

}  // namespace

// Plugins may hold state.  The compiler context is the only handle we need; it
// is owned by the runtime and outlives the plugin.
// LITERT_DEFINE_HANDLE declares these as `class name##T*`, so they must be
// defined with `class` too; `struct` is a different tag under the MSVC ABI.
class LiteRtCompilerPluginT {
 public:
  explicit LiteRtCompilerPluginT(const LiteRtCompilerContext* context)
      : compiler_context(context) {}

  const LiteRtCompilerContext* compiler_context = nullptr;
};

// A single VMFB plus the per-partition entry points inside it.
class LiteRtCompiledResultT {
 public:
  std::vector<std::vector<uint8_t>> byte_codes;
  std::vector<std::string> call_infos;
};

namespace {

// ---------------------------------------------------------------------------
// Plugin metadata
// ---------------------------------------------------------------------------

LiteRtStatus GetCompilerPluginVersionImpl(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* GetCompilerPluginSocManufacturerImpl() { return kSocManufacturer; }

LiteRtStatus GetNumCompilerPluginSupportedSocModelsImpl(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = 1;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCompilerPluginSupportedSocModelImpl(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr || soc_model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (soc_model_idx != 0) {
    return kLiteRtStatusErrorUnsupported;
  }
  *soc_model_name = kSocModel;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCompilerPluginSupportedHardwareImpl(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (compiler_plugin == nullptr || supported_hardware == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus GetCompilerPluginSDKVersionImpl(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  if (compiler_plugin == nullptr || sdk_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *sdk_version = kSdkVersion;
  return kLiteRtStatusOk;
}

LiteRtStatus CreateCompilerPluginImpl(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  if (compiler_context == nullptr || compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *compiler_plugin = new LiteRtCompilerPluginT(compiler_context);
  return kLiteRtStatusOk;
}

void DestroyCompilerPluginImpl(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

LiteRtStatus CompilerPluginCheckCompilerCompatibilityImpl(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  if (compiler_plugin == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // A null soc_model means "let the plugin choose", which is always fine.
  if (soc_model_name == nullptr) {
    return kLiteRtStatusOk;
  }
  // The name comparison is intentionally exact: silently accepting a
  // different SoC would produce binaries the dispatch library cannot run.
  if (std::strcmp(soc_model_name, kSocModel) != 0) {
    CORALNPU_ERROR("unsupported soc_model '%s' (this plugin supports '%s')",
                   soc_model_name, kSocModel);
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus CompilerPluginRegisterAllTransformationsImpl(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  if (num_patterns == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  // Coral NPU accepts the graph as-is; the TOSA conversion and attention tiling
  // happen inside the compiler backend, not as LiteRT graph rewrites.
  *num_patterns = 0;
  if (transformations != nullptr) {
    *transformations = nullptr;
  }
  return kLiteRtStatusOk;
}

// ---------------------------------------------------------------------------
// Partitioning
// ---------------------------------------------------------------------------

// Returns true for ops that must stay outside the accelerator partition even
// though they are part of the graph.
//
// `tfl.embedding_lookup` is a table gather, not a matmul: Torq has no lowering
// for it, and inlining the table into the TOSA module duplicates the whole
// embedding (168 MB for Gemma 3 270M) once per exported function, which blows
// the VMFB up by hundreds of megabytes.  Leaving it to the CPU keeps the weight
// out of the accelerator artifact and costs a memcpy per token, no arithmetic.
bool MustRunOnHost(LiteRtOpCode code) {
  switch (code) {
    case kLiteRtOpCodeTflEmbeddingLookup:
      return true;
    default:
      return false;
  }
}

// Selects a single partition containing every op in `c_subgraph` that the Coral
// NPU toolchain can lower.
//
// Ops are pushed in graph order with partition_index 0.  Ops rejected by
// MustRunOnHost() are left out, so LiteRT keeps them on the CPU and feeds their
// outputs to the partition as inputs.  Everything else is either lowered or
// reported by compile(), with the op visible in the IREE diagnostics, rather
// than silently falling back mid-graph.
LiteRtStatus CompilerPluginPartitionImpl(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph c_subgraph, LiteRtOpList selected_ops) {
  if (compiler_plugin == nullptr || compiler_plugin->compiler_context == nullptr ||
      c_subgraph == nullptr || selected_ops == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const LiteRtCompilerContext* ctx = compiler_plugin->compiler_context;

  LiteRtParamIndex num_ops = 0;
  CORALNPU_RETURN_IF_ERROR(ctx->get_num_subgraph_ops(c_subgraph, &num_ops));

  LiteRtParamIndex num_selected = 0;
  LiteRtParamIndex num_host = 0;
  for (LiteRtParamIndex i = 0; i < num_ops; ++i) {
    LiteRtOp op = nullptr;
    CORALNPU_RETURN_IF_ERROR(ctx->get_subgraph_op(c_subgraph, i, &op));
    if (op == nullptr) {
      return kLiteRtStatusErrorNotFound;
    }
    LiteRtOpCode code;
    CORALNPU_RETURN_IF_ERROR(ctx->get_op_code(op, &code));
    if (MustRunOnHost(code)) {
      ++num_host;
      continue;
    }
    CORALNPU_RETURN_IF_ERROR(ctx->push_op(selected_ops, op, /*partition_index=*/0));
    ++num_selected;
  }

  CORALNPU_INFO("selected %u op(s) into partition 0 (%u kept on host)",
                static_cast<unsigned>(num_selected),
                static_cast<unsigned>(num_host));
  return kLiteRtStatusOk;
}

// ---------------------------------------------------------------------------
// Compilation
// ---------------------------------------------------------------------------

namespace {

// Builds the per-subgraph signature keys handed to
// serialize_model_with_signatures.  The ABI takes ownership of nothing here; the
// strings stay alive until the call returns.
std::vector<char*> MakeSignatureKeys(int num_partitions,
                                     std::vector<std::string>* storage) {
  storage->clear();
  storage->reserve(static_cast<size_t>(num_partitions));
  std::vector<char*> keys;
  keys.reserve(static_cast<size_t>(num_partitions));
  for (int i = 0; i < num_partitions; ++i) {
    storage->push_back(PartitionFunctionName(i));
  }
  for (std::string& name : *storage) {
    keys.push_back(name.data());
  }
  return keys;
}

}  // namespace

LiteRtStatus CompilerPluginCompileImpl(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  if (compiler_plugin == nullptr || compiler_plugin->compiler_context == nullptr ||
      partitions == nullptr || compiled_result == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (soc_model != nullptr && std::strcmp(soc_model, kSocModel) != 0) {
    CORALNPU_ERROR("compile() called with unsupported soc_model '%s'",
                   soc_model);
    return kLiteRtStatusErrorUnsupportedCompilerVersion;
  }

  const LiteRtCompilerContext* ctx = compiler_plugin->compiler_context;

  LiteRtParamIndex num_subgraphs = 0;
  CORALNPU_RETURN_IF_ERROR(ctx->get_num_model_subgraphs(partitions, &num_subgraphs));
  if (num_subgraphs == 0) {
    CORALNPU_ERROR("no partitions were selected for Coral NPU compilation");
    return kLiteRtStatusErrorInvalidArgument;
  }
  const int num_partitions = static_cast<int>(num_subgraphs);

  std::vector<std::string> signature_storage;
  std::vector<char*> signatures =
      MakeSignatureKeys(num_partitions, &signature_storage);

  LiteRtModelSerializationOptions options{};
  options.bytecode_alignment = 1;

  uint8_t* serialized = nullptr;
  size_t serialized_size = 0;
  size_t serialized_offset = 0;
  CORALNPU_RETURN_IF_ERROR(ctx->serialize_model_with_signatures(
      partitions, &serialized, &serialized_size, &serialized_offset,
      /*share_weights=*/false, signatures.data(), signatures.size(), options));

  // The runtime hands back a buffer owned by an OwningBufferRef<uint8_t>, whose
  // default allocator is Newlocator: the memory comes from `new uint8_t[]` and
  // must be released with `delete[]`.  The releaser makes that exception-safe.
  struct BufferReleaser {
    uint8_t* buffer;
    ~BufferReleaser() { delete[] buffer; }
  } releaser{serialized};

  if (serialized == nullptr || serialized_size == 0) {
    CORALNPU_ERROR("failed to serialize the selected partitions");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  coralnpu::CompileRequest request;
  request.soc_model = soc_model != nullptr ? soc_model : kSocModel;
  request.model_data = serialized;
  request.model_size = serialized_size;
  request.model_offset = serialized_offset;
  request.num_partitions = num_partitions;

  coralnpu::CompileArtifact artifact;
  CORALNPU_RETURN_IF_ERROR(
      coralnpu::GetCompileBackend()->Compile(request, &artifact));

  if (artifact.byte_code.empty()) {
    CORALNPU_ERROR("compile backend returned no bytecode");
    return kLiteRtStatusErrorCompilation;
  }
  if (static_cast<int>(artifact.call_infos.size()) != num_partitions) {
    CORALNPU_ERROR(
        "compile backend returned %zu call(s) for %d partition(s); the "
        "dispatch library would call the wrong functions",
        artifact.call_infos.size(), num_partitions);
    return kLiteRtStatusErrorCompilation;
  }

  auto result = std::make_unique<LiteRtCompiledResultT>();
  result->byte_codes.push_back(std::move(artifact.byte_code));
  result->call_infos = std::move(artifact.call_infos);

  CORALNPU_INFO("compiled %d partition(s) into a %zu byte module",
                num_partitions, result->byte_codes.front().size());

  *compiled_result = result.release();
  return kLiteRtStatusOk;
}

// ---------------------------------------------------------------------------
// Compiled result accessors
// ---------------------------------------------------------------------------

void DestroyCompiledResultImpl(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus GetCompiledResultByteCodeImpl(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  if (compiled_result == nullptr || byte_code == nullptr ||
      byte_code_size == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (byte_code_idx >= compiled_result->byte_codes.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const std::vector<uint8_t>& code = compiled_result->byte_codes[byte_code_idx];
  *byte_code = code.data();
  *byte_code_size = code.size();
  return kLiteRtStatusOk;
}

LiteRtStatus CompiledResultNumByteCodeModulesImpl(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (compiled_result == nullptr || num_byte_code == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code =
      static_cast<LiteRtParamIndex>(compiled_result->byte_codes.size());
  return kLiteRtStatusOk;
}

LiteRtStatus GetCompiledResultCallInfoImpl(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (compiled_result == nullptr || call_info == nullptr ||
      call_info_size == nullptr || byte_code_idx == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (call_idx >= compiled_result->call_infos.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }
  const std::string& info = compiled_result->call_infos[call_idx];
  *call_info = info.data();
  *call_info_size = info.size();
  // All partitions live in the single module produced above.
  *byte_code_idx = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus GetNumCompiledResultCallsImpl(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  if (compiled_result == nullptr || num_calls == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_calls =
      static_cast<LiteRtParamIndex>(compiled_result->call_infos.size());
  return kLiteRtStatusOk;
}

// Coral NPU compiles ahead of time, so there is never a JIT handle.
LiteRtStatus GetCompiledResultHandleImpl(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    LiteRtJitExecutable* handle) {
  if (compiled_result == nullptr || handle == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *handle = nullptr;
  return kLiteRtStatusOk;
}

}  // namespace

// ---------------------------------------------------------------------------
// Exported metadata entry points
// ---------------------------------------------------------------------------
//
// LiteRT's plugin loader resolves these symbols with dlsym() before it queries
// the interface table, so they need C linkage and default visibility.  The
// implementations above stay internal; the table keeps pointing at them.

extern "C" {

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompilerPluginVersion(
    LiteRtApiVersion* api_version) {
  return GetCompilerPluginVersionImpl(api_version);
}

LITERT_CAPI_EXPORT const char* LiteRtGetCompilerPluginSocManufacturer() {
  return GetCompilerPluginSocManufacturerImpl();
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  return GetNumCompilerPluginSupportedSocModelsImpl(
      compiler_plugin, num_supported_soc_models);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  return GetCompilerPluginSupportedSocModelImpl(compiler_plugin, soc_model_idx,
                                                soc_model_name);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  return GetCompilerPluginSupportedHardwareImpl(compiler_plugin,
                                                supported_hardware);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompilerPluginSDKVersion(
    LiteRtCompilerPlugin compiler_plugin, const char** sdk_version) {
  return GetCompilerPluginSDKVersionImpl(compiler_plugin, sdk_version);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCreateCompilerPlugin(
    const LiteRtCompilerContext* compiler_context,
    LiteRtCompilerPlugin* compiler_plugin, LiteRtEnvironmentOptions env,
    LiteRtOptions options) {
  return CreateCompilerPluginImpl(compiler_context, compiler_plugin, env,
                                  options);
}

LITERT_CAPI_EXPORT void LiteRtDestroyCompilerPlugin(
    LiteRtCompilerPlugin compiler_plugin) {
  DestroyCompilerPluginImpl(compiler_plugin);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  return CompilerPluginCheckCompilerCompatibilityImpl(
      api_version, compiler_plugin, env, options, soc_model_name);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  return CompilerPluginRegisterAllTransformationsImpl(
      compiler_plugin, transformations, num_patterns);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginPartition(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtSubgraph c_subgraph, LiteRtOpList selected_ops) {
  return CompilerPluginPartitionImpl(compiler_plugin, soc_model, c_subgraph,
                                     selected_ops);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  return CompilerPluginCompileImpl(compiler_plugin, soc_model, partitions,
                                   compiled_result);
}

LITERT_CAPI_EXPORT void LiteRtDestroyCompiledResult(
    LiteRtCompiledResult compiled_result) {
  DestroyCompiledResultImpl(compiled_result);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  return GetCompiledResultByteCodeImpl(compiled_result, byte_code_idx,
                                       byte_code, byte_code_size);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  return CompiledResultNumByteCodeModulesImpl(compiled_result, num_byte_code);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  return GetCompiledResultCallInfoImpl(compiled_result, call_idx, call_info,
                                       call_info_size, byte_code_idx);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  return GetNumCompiledResultCallsImpl(compiled_result, num_calls);
}

LITERT_CAPI_EXPORT LiteRtStatus LiteRtGetCompiledResultHandle(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    LiteRtJitExecutable* handle) {
  return GetCompiledResultHandleImpl(compiled_result, call_idx, handle);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Interface table
// ---------------------------------------------------------------------------

namespace {

const LiteRtCompilerPluginInterface_V1 kCoralNpuCompilerPluginInterface = {
    /*.abi_header=*/
    {
        .struct_size = sizeof(LiteRtCompilerPluginInterface_V1),
        .major_version = LITERT_COMPILER_PLUGIN_ABI_VERSION_MAJOR,
        .minor_version = LITERT_COMPILER_PLUGIN_ABI_VERSION_MINOR,
        .reserved = 0,
    },
    /*.get_compiler_plugin_version=*/GetCompilerPluginVersionImpl,
    /*.get_compiler_plugin_soc_manufacturer=*/
    GetCompilerPluginSocManufacturerImpl,
    /*.create_compiler_plugin=*/CreateCompilerPluginImpl,
    /*.destroy_compiler_plugin=*/DestroyCompilerPluginImpl,
    /*.get_compiler_plugin_supported_hardware=*/
    GetCompilerPluginSupportedHardwareImpl,
    /*.get_num_compiler_plugin_supported_models=*/
    GetNumCompilerPluginSupportedSocModelsImpl,
    /*.get_compiler_plugin_supported_soc_model=*/
    GetCompilerPluginSupportedSocModelImpl,
    /*.compiler_plugin_partition=*/CompilerPluginPartitionImpl,
    /*.compiler_plugin_compile=*/CompilerPluginCompileImpl,
    /*.destroy_compiled_result=*/DestroyCompiledResultImpl,
    /*.get_compiled_result_byte_code=*/GetCompiledResultByteCodeImpl,
    /*.get_compiled_result_num_byte_code=*/
    CompiledResultNumByteCodeModulesImpl,
    /*.get_compiled_result_call_info=*/GetCompiledResultCallInfoImpl,
    /*.get_num_compiled_result_calls=*/GetNumCompiledResultCallsImpl,
    /*.register_all_transformations=*/
    CompilerPluginRegisterAllTransformationsImpl,
    /*.get_compiler_plugin_sdk_version=*/GetCompilerPluginSDKVersionImpl,
    /*.get_compiled_result_handle=*/GetCompiledResultHandleImpl,
    /*.check_compiler_compatibility=*/
    CompilerPluginCheckCompilerCompatibilityImpl,
};

}  // namespace

extern "C" LITERT_CAPI_EXPORT LiteRtStatus LiteRtCompilerPluginQueryInterface(
    LiteRtCompilerPluginInterfaceId interface_id,
    LiteRtApiVersion litert_runtime_version, LiteRtInterface* out_interface) {
  if (out_interface == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (litert_runtime_version.major < 1) {
    return kLiteRtStatusErrorUnsupportedRuntimeVersion;
  }
  if (interface_id == kLiteRtCompilerPluginInterfaceBasic) {
    *out_interface =
        const_cast<LiteRtCompilerPluginInterface_V1*>(
            &kCoralNpuCompilerPluginInterface);
    return kLiteRtStatusOk;
  }
  return kLiteRtStatusErrorUnsupported;
}
