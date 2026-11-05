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

// AOT compile backend that drives `scripts/compile_litert.py`.
//
// Rationale: the TOSA lowering, the multi-head-attention tiling, and the shape
// binding already live in the Python driver and are covered by tests.  Rather
// than reimplement them in C++, the plugin writes the selected partitions out
// as a TFLite flatbuffer and asks the driver to produce a VMFB.  That keeps a
// single source of truth for the Coral NPU pipeline and means improvements to
// the driver reach the plugin for free.
//
// Configuration is environment-based so that no ABI-visible struct grows:
//
//   CORALNPU_AOT_DRIVER       path to scripts/compile_litert.py  (required
//                             unless CORALNPU_AOT_CACHE holds the artifact)
//   CORALNPU_PYTHON           interpreter to run it   (default: python3)
//   CORALNPU_AOT_PROFILE      compile_litert --profile (default: coralnpu)
//   CORALNPU_AOT_SEQ_LEN      value for --seq-len, when the graph is dynamic
//   CORALNPU_AOT_EXTRA_FLAGS  extra whitespace-split driver flags
//   CORALNPU_AOT_KEEP_TEMP    if set, keep the scratch directory for debugging
//   CORALNPU_AOT_CACHE        directory of prebuilt <key>.vmfb artifacts.  A hit
//                             skips the driver entirely, and a miss stores its
//                             output there.  This is how a device consumes a
//                             module that was compiled on a build host.
//
// The driver is a subprocess, so the plugin has no build-time dependency on
// Python, on IREE, or on the Coral NPU toolchain.  A missing driver is a clear
// runtime error, not a link error.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/vendors/coralnpu/common/coralnpu_common.h"
#include "litert/vendors/coralnpu/common/coralnpu_subprocess.h"
#include "litert/vendors/coralnpu/compiler_plugin/coralnpu_compile_backend.h"

namespace coralnpu {
namespace {

constexpr char kScratchModelName[] = "partitions.tflite";
constexpr char kVmfbName[] = "partitions.vmfb";

// Every artefact the driver may write into the scratch directory.  Removing a
// fixed list is enough and avoids shelling out to `rm -rf`.
const std::vector<std::string>& ScratchArtifacts() {
  static const std::vector<std::string>* const kArtifacts =
      new std::vector<std::string>{
          kScratchModelName,          kVmfbName,
          "01_tosa.mlir",             "02_static.mlir",
          "03_tiled.mlir",            "mhsa_tiled.mlir",
          "tile_plan.json",           "extracted_model.tflite",
      };
  return *kArtifacts;
}

// 64-bit FNV-1a.  Used only to name cache entries, so collisions are a
// correctness hazard rather than a security one: the key mixes in the size and
// the full serialized bytes, and a collision would have to be built on purpose.
void Fnv1a(const void* data, size_t size, uint64_t* hash) {
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  uint64_t h = *hash;
  for (size_t i = 0; i < size; ++i) {
    h ^= bytes[i];
    h *= 1099511628211ULL;
  }
  *hash = h;
}

// Names one AOT artifact by everything that can change its contents: the
// serialized partitions, the driver flags, and the partition count.
//
// The VMFB is a pure function of those inputs, so a cache hit is equivalent to
// a compile.  That is what makes it possible to build the module on a fast
// Linux host (where the Torq toolchain and a scratch disk are available) and
// have the device pick it up at load time instead of spending minutes inside
// the LiteRT process.
std::string CacheKey(const CompileRequest& request, const std::string& profile,
                     const std::string& seq_len,
                     const std::string& extra_flags) {
  uint64_t hash = 1469598103934665603ULL;
  const size_t model_size = request.model_size - request.model_offset;
  Fnv1a(request.model_data + request.model_offset, model_size, &hash);
  Fnv1a(&model_size, sizeof(model_size), &hash);
  Fnv1a(&request.num_partitions, sizeof(request.num_partitions), &hash);
  Fnv1a(profile.data(), profile.size(), &hash);
  Fnv1a(seq_len.data(), seq_len.size(), &hash);
  Fnv1a(extra_flags.data(), extra_flags.size(), &hash);
  char key[17];
  std::snprintf(key, sizeof(key), "%016llx",
                static_cast<unsigned long long>(hash));
  return key;
}

std::string JoinPath(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

class AotCompileBackend final : public CompileBackend {
 public:
  LiteRtStatus Compile(const CompileRequest& request,
                       CompileArtifact* artifact) override {
    if (artifact == nullptr || request.model_data == nullptr ||
        request.model_size == 0) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (request.model_offset >= request.model_size) {
      CORALNPU_ERROR("serialized model view is empty (offset %zu of %zu)",
                     request.model_offset, request.model_size);
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (request.num_partitions <= 0) {
      CORALNPU_ERROR("no partitions to compile");
      return kLiteRtStatusErrorInvalidArgument;
    }

#if defined(_WIN32)
    CORALNPU_ERROR(
        "the Coral NPU AOT compile backend currently requires a POSIX host");
    return kLiteRtStatusErrorUnsupported;
#else
    const std::string profile = EnvOr("CORALNPU_AOT_PROFILE", "coralnpu");
    const std::string seq_len = EnvOr("CORALNPU_AOT_SEQ_LEN", "");
    const std::string extra_flags = EnvOr("CORALNPU_AOT_EXTRA_FLAGS", "");
    const std::string cache_dir = EnvOr("CORALNPU_AOT_CACHE", "");
    const std::string cache_key = CacheKey(request, profile, seq_len, extra_flags);
    const std::string cache_path =
        cache_dir.empty() ? std::string() : JoinPath(cache_dir, cache_key + ".vmfb");
    if (FlagSet("CORALNPU_AOT_CACHE_LOG")) {
      std::fprintf(stderr, "CoralNPU AOT cache key %s (%zu model bytes)\n",
                   cache_key.c_str(), request.model_size - request.model_offset);
    }

    artifact->call_infos.clear();
    artifact->call_infos.reserve(static_cast<size_t>(request.num_partitions));
    for (int i = 0; i < request.num_partitions; ++i) {
      artifact->call_infos.push_back("partition_" + std::to_string(i));
    }

    // A cache hit skips the toolchain entirely — including the driver
    // configuration.  This is the path a device takes: the artifact was built
    // on a host with the Torq toolchain and pushed to `CORALNPU_AOT_CACHE`,
    // and the device only has to read it back.
    if (!cache_path.empty() && IsReadableFile(cache_path)) {
      if (ReadFileBytes(cache_path, &artifact->byte_code) &&
          !artifact->byte_code.empty()) {
        CORALNPU_INFO("AOT cache hit for %s (%zu bytes, %d partition(s))",
                      cache_path.c_str(), artifact->byte_code.size(),
                      request.num_partitions);
        return kLiteRtStatusOk;
      }
      CORALNPU_ERROR("AOT cache entry %s is unreadable; recompiling",
                     cache_path.c_str());
      artifact->byte_code.clear();
    }

    const std::string driver = EnvOr("CORALNPU_AOT_DRIVER", "");
    if (driver.empty()) {
      CORALNPU_ERROR(
          "CORALNPU_AOT_DRIVER is not set; point it at "
          "scripts/compile_litert.py (or a wrapper) so the compiler plugin can "
          "reach the Coral NPU toolchain, or set CORALNPU_AOT_CACHE to a "
          "directory holding a prebuilt <cache key>.vmfb");
      return kLiteRtStatusErrorUnsupported;
    }
    if (::access(driver.c_str(), R_OK) != 0) {
      CORALNPU_ERROR("CORALNPU_AOT_DRIVER is not readable: %s", driver.c_str());
      return kLiteRtStatusErrorNotFound;
    }

    std::string scratch;
    if (!MakeScratchDir("coralnpu-aot-", &scratch)) {
      CORALNPU_ERROR("could not create a scratch directory for the AOT driver");
      return kLiteRtStatusErrorRuntimeFailure;
    }

    const std::string model_path = scratch + "/" + kScratchModelName;
    const std::string vmfb_path = scratch + "/" + kVmfbName;

    LiteRtStatus status = kLiteRtStatusOk;
    std::vector<std::string> argv;
    do {
      if (!WriteFileBytes(model_path, request.model_data + request.model_offset,
                          request.model_size - request.model_offset)) {
        CORALNPU_ERROR("could not write %s", model_path.c_str());
        status = kLiteRtStatusErrorRuntimeFailure;
        break;
      }

      argv = {
          EnvOr("CORALNPU_PYTHON", "python3"),
          driver,
          model_path,
          "--out-dir", scratch,
          "--profile", profile,
          // Generic partitions are not guaranteed to contain a recognised
          // attention function; tile it when we find one, otherwise pass the
          // graph through.  LLM callers pin the function explicitly with
          // CORALNPU_AOT_EXTRA_FLAGS=--attention-func=...
          "--allow-untiled-attention",
      };

      if (!seq_len.empty()) {
        argv.push_back("--seq-len");
        argv.push_back(seq_len);
      }
      for (const std::string& extra : SplitWhitespace(extra_flags)) {
        argv.push_back(extra);
      }

      std::string command_line;
      for (const std::string& arg : argv) {
        if (!command_line.empty()) command_line.push_back(' ');
        command_line += arg;
      }
      CORALNPU_INFO("compiling %d partition(s): %s", request.num_partitions,
                    command_line.c_str());

      const int exit_code = RunProcess(argv);
      if (exit_code != 0) {
        CORALNPU_ERROR("AOT driver failed with exit code %d: %s", exit_code,
                       command_line.c_str());
        status = kLiteRtStatusErrorCompilation;
        break;
      }

      if (!ReadFileBytes(vmfb_path, &artifact->byte_code)) {
        CORALNPU_ERROR("AOT driver produced no %s", vmfb_path.c_str());
        status = kLiteRtStatusErrorCompilation;
        break;
      }
      if (artifact->byte_code.empty()) {
        CORALNPU_ERROR("AOT driver produced an empty %s", vmfb_path.c_str());
        status = kLiteRtStatusErrorCompilation;
        break;
      }

      if (!cache_path.empty()) {
        if (!MakeDirectories(cache_dir)) {
          CORALNPU_ERROR("could not create the AOT cache directory %s",
                         cache_dir.c_str());
        } else if (WriteFileBytes(cache_path, artifact->byte_code.data(),
                                  artifact->byte_code.size())) {
          CORALNPU_INFO("cached AOT artifact as %s", cache_path.c_str());
        } else {
          CORALNPU_ERROR("could not write AOT cache entry %s", cache_path.c_str());
        }
      }
    } while (false);

    if (FlagSet("CORALNPU_AOT_KEEP_TEMP")) {
      CORALNPU_INFO("keeping scratch directory %s", scratch.c_str());
    } else {
      RemoveScratchDir(scratch, ScratchArtifacts());
    }
    return status;
#endif  // defined(_WIN32)
  }
};

}  // namespace

CompileBackend* GetCompileBackend() {
  static AotCompileBackend backend;
  return &backend;
}

}  // namespace coralnpu
