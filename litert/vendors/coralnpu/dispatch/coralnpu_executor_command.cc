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

// DispatchExecutors that run a compiled module in another process.
//
// Two of them, and the choice is a runtime one (see GetDispatchExecutor):
//
//   * the *daemon* executor keeps a Coral NPU runtime process alive and sends
//     tensors over a unix socket.  The VMFB is loaded once, so constants are
//     uploaded to the NPU once instead of once per invocation — that is what
//     makes per-token decoding viable.
//   * the *command* executor runs `iree-run-module` per invocation.  It needs
//     no long-lived process and is the right tool for correctness runs.
//
// Why a subprocess instead of linking IREE:
//
//   * The dispatch library has to load into the LiteRT process, which already
//     has its own allocator, logging, and (potentially) IREE copy.  Linking a
//     second IREE runtime into that process is a good way to get duplicate
//     symbol or allocator problems on a vendor device.
//   * `iree-run-module` is the same entry point the Coral NPU team documents for
//     bringing a VMFB up on a board, and its HAL driver list is chosen at
//     runtime (`--device=torq` for the NPU, `local-task` for the CPU fallback).
//     So the executor follows whatever runtime the device ships instead of
//     pinning one at link time.
//   * It needs no IREE headers or libraries at build time, which keeps this
//     library buildable on any machine and testable with a stub script.
//
// Configuration:
//
//   CORALNPU_DAEMON_SOCKET         unix socket of `scripts/npu_daemon.py`.  When
//                                  set, the daemon executor is used.
//   CORALNPU_RUN_MODULE            path to iree-run-module.  Defaults to
//                                  $CORALNPU_IREE_BIN/iree-run-module, then to
//                                  iree-run-module on $PATH.
//   CORALNPU_DEVICE                HAL device: `torq` for the Coral NPU,
//                                  `local-task`/`local-sync` for the CPU.
//                                  Default: local-task.
//   CORALNPU_RUN_MODULE_EXTRA_FLAGS  extra whitespace-split flags
//   CORALNPU_DISPATCH_KEEP_TEMP    keep the scratch directory for debugging

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/coralnpu/common/coralnpu_common.h"
#include "litert/vendors/coralnpu/common/coralnpu_subprocess.h"
#include "litert/vendors/coralnpu/dispatch/coralnpu_executor.h"

namespace coralnpu {
namespace {

constexpr char kModuleName[] = "module.vmfb";
constexpr char kToolName[] = "iree-run-module";

// Returns the numpy dtype code for `type`, or nullptr when numpy has no
// representation for it.
//
// Packed 4-bit values, fp8, and bfloat16 are deliberately refused.  Passing
// their bytes through as int8 would produce a module that runs and returns
// wrong numbers, which is worse than an explicit unsupported error.
const char* NumpyDescr(LiteRtElementType type) {
  switch (type) {
    case kLiteRtElementTypeBool:
      return "=b1";
    case kLiteRtElementTypeInt8:
      return "=i1";
    case kLiteRtElementTypeUInt8:
      return "=u1";
    case kLiteRtElementTypeInt16:
      return "=i2";
    case kLiteRtElementTypeUInt16:
      return "=u2";
    case kLiteRtElementTypeFloat16:
      return "=f2";
    case kLiteRtElementTypeInt32:
      return "=i4";
    case kLiteRtElementTypeUInt32:
      return "=u4";
    case kLiteRtElementTypeFloat32:
      return "=f4";
    case kLiteRtElementTypeInt64:
      return "=i8";
    case kLiteRtElementTypeUInt64:
      return "=u8";
    case kLiteRtElementTypeFloat64:
      return "=f8";
    case kLiteRtElementTypeComplex64:
      return "=c8";
    case kLiteRtElementTypeComplex128:
      return "=c16";
    default:
      return nullptr;
  }
}

// Builds a .npy v1.0 header followed by `data`.
//
// The header is an ASCII dict padded with spaces so that the total header
// length is a multiple of 64 bytes, as numpy requires.  `shape` describes
// `type`; callers guarantee the data length matches.
bool AppendNpy(const LiteRtRankedTensorType& type, const uint8_t* data,
               size_t size, std::vector<uint8_t>* out) {
  const char* descr = NumpyDescr(type.element_type);
  if (descr == nullptr) {
    return false;
  }
  std::string shape;
  if (type.layout.rank == 0) {
    shape = "()";
  } else {
    shape = "(";
    for (unsigned int i = 0; i < type.layout.rank; ++i) {
      if (i > 0) {
        shape += ", ";
      }
      shape += std::to_string(type.layout.dimensions[i]);
    }
    if (type.layout.rank == 1) {
      shape += ",";
    }
    shape += ")";
  }

  std::string dict = std::string("{'descr': '") + descr +
                     "', 'fortran_order': False, 'shape': " + shape + ", }";
  // magic(6) + version(2) + header_len(2) + dict, padded to 64 bytes, with the
  // dict terminated by a newline.
  const size_t unpadded = 10 + dict.size() + 1;
  const size_t padding = (64 - (unpadded % 64)) % 64;
  dict.append(padding, ' ');
  dict.push_back('\n');
  if (dict.size() > 0xFFFF) {
    return false;
  }

  out->clear();
  out->reserve(10 + dict.size() + size);
  out->insert(out->end(), {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0});
  const uint16_t header_len = static_cast<uint16_t>(dict.size());
  out->push_back(static_cast<uint8_t>(header_len & 0xFF));
  out->push_back(static_cast<uint8_t>((header_len >> 8) & 0xFF));
  out->insert(out->end(), dict.begin(), dict.end());
  out->insert(out->end(), data, data + size);
  return true;
}

// Resolves the tool to run, or an empty string if it cannot be found.
std::string ResolveRunModule() {
  const std::string explicit_path = EnvOr("CORALNPU_RUN_MODULE", "");
  if (!explicit_path.empty()) {
    return explicit_path;
  }
  const std::string iree_bin = EnvOr("CORALNPU_IREE_BIN", "");
  if (!iree_bin.empty()) {
    std::string candidate = iree_bin;
    if (candidate.back() != '/') {
      candidate.push_back('/');
    }
    candidate += kToolName;
    if (IsExecutableFile(candidate)) {
      return candidate;
    }
  }
  return FindOnPath(kToolName);
}

// Returns the host address of `arg`, or null if the runtime cannot provide one.
void* HostPointer(const TensorArg& arg,
                  const LiteRtRuntimeContext* runtime_context) {
  void* address = nullptr;
  if (runtime_context->get_tensor_buffer_host_memory == nullptr) {
    return nullptr;
  }
  if (runtime_context->get_tensor_buffer_host_memory(arg.buffer, &address) !=
      kLiteRtStatusOk) {
    return nullptr;
  }
  return address;
}

class CommandExecutor final : public DispatchExecutor {
 public:
  LiteRtStatus Invoke(const LiteRtMemBuffer& bytecode,
                      const char* function_name,
                      const std::vector<TensorArg>& inputs,
                      const std::vector<TensorArg>& outputs,
                      const LiteRtRuntimeContext* runtime_context) override {
    if (bytecode.base_addr == nullptr || bytecode.size == 0) {
      CORALNPU_ERROR("invoke called with an empty bytecode module");
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (function_name == nullptr || function_name[0] == '\0') {
      CORALNPU_ERROR("invoke called without a function name");
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (runtime_context == nullptr ||
        runtime_context->get_tensor_buffer_host_memory == nullptr) {
      CORALNPU_ERROR(
          "the LiteRT runtime context cannot expose host memory, so the "
          "command executor cannot stage tensors for %s",
          kToolName);
      return kLiteRtStatusErrorUnsupported;
    }

    const std::string tool = ResolveRunModule();
    if (tool.empty()) {
      CORALNPU_ERROR(
          "cannot find %s; set CORALNPU_RUN_MODULE to its path (or "
          "CORALNPU_IREE_BIN to the IREE runtime bin directory)",
          kToolName);
      return kLiteRtStatusErrorNotFound;
    }
    if (!IsExecutableFile(tool)) {
      CORALNPU_ERROR("CORALNPU_RUN_MODULE is not executable: %s", tool.c_str());
      return kLiteRtStatusErrorNotFound;
    }

#if defined(_WIN32)
    CORALNPU_ERROR(
        "the Coral NPU command executor currently requires a POSIX host");
    return kLiteRtStatusErrorUnsupported;
#else
    std::string scratch;
    if (!MakeScratchDir("coralnpu-run-", &scratch)) {
      CORALNPU_ERROR("could not create a scratch directory for %s",
                     kToolName);
      return kLiteRtStatusErrorRuntimeFailure;
    }

    std::vector<std::string> staged{std::string(kModuleName)};
    LiteRtStatus status = Execute(tool, scratch, bytecode, function_name, inputs,
                                  outputs, runtime_context, &staged);

    if (FlagSet("CORALNPU_DISPATCH_KEEP_TEMP")) {
      CORALNPU_INFO("keeping scratch directory %s", scratch.c_str());
    } else {
      RemoveScratchDir(scratch, staged);
    }
    return status;
#endif  // defined(_WIN32)
  }

 private:
#if !defined(_WIN32)
  LiteRtStatus Execute(const std::string& tool, const std::string& scratch,
                       const LiteRtMemBuffer& bytecode,
                       const char* function_name,
                       const std::vector<TensorArg>& inputs,
                       const std::vector<TensorArg>& outputs,
                       const LiteRtRuntimeContext* runtime_context,
                       std::vector<std::string>* staged) const {
    const std::string module_path = scratch + "/" + kModuleName;
    if (!WriteFileBytes(module_path,
                        static_cast<const uint8_t*>(bytecode.base_addr) +
                            bytecode.offset,
                        bytecode.size)) {
      CORALNPU_ERROR("could not write %s", module_path.c_str());
      return kLiteRtStatusErrorRuntimeFailure;
    }

    std::vector<std::string> argv = {
        tool,
        "--module=" + module_path,
        "--device=" + EnvOr("CORALNPU_DEVICE", "local-task"),
        "--function=" + std::string(function_name),
    };

    // Stage inputs as .npy so iree-run-module sees both the shape and the
    // element type rather than raw bytes it would have to guess at.
    for (size_t i = 0; i < inputs.size(); ++i) {
      const TensorArg& arg = inputs[i];
      if (!arg.has_type) {
        CORALNPU_ERROR(
            "input %zu has no recorded tensor type; the runtime must call "
            "get_input_requirements before invoking",
            i);
        return kLiteRtStatusErrorInvalidArgument;
      }
      if (NumpyDescr(arg.type.element_type) == nullptr) {
        CORALNPU_ERROR(
            "input %zu has element type %d, which numpy cannot represent; "
            "use the in-process IREE executor for this model",
            i, static_cast<int>(arg.type.element_type));
        return kLiteRtStatusErrorUnsupported;
      }
      size_t byte_size = 0;
      CORALNPU_RETURN_IF_ERROR(TensorByteSize(arg.type, &byte_size));
      const void* host = HostPointer(arg, runtime_context);
      if (host == nullptr) {
        CORALNPU_ERROR("input %zu is not backed by host memory", i);
        return kLiteRtStatusErrorUnsupported;
      }
      std::vector<uint8_t> npy;
      if (!AppendNpy(arg.type, static_cast<const uint8_t*>(host), byte_size,
                     &npy)) {
        CORALNPU_ERROR("could not encode input %zu as npy", i);
        return kLiteRtStatusErrorRuntimeFailure;
      }
      const std::string name = "in" + std::to_string(i) + ".npy";
      staged->push_back(name);
      const std::string path = scratch + "/" + name;
      if (!WriteFileBytes(path, npy.data(), npy.size())) {
        CORALNPU_ERROR("could not write %s", path.c_str());
        return kLiteRtStatusErrorRuntimeFailure;
      }
      argv.push_back("--input=@" + path);
    }

    // Outputs go to raw .bin files: the expected size is known from the type,
    // so no header parsing is needed on the way back.
    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string name = "out" + std::to_string(i) + ".bin";
      staged->push_back(name);
      argv.push_back("--output=@" + scratch + "/" + name);
    }

    for (const std::string& extra :
         SplitWhitespace(EnvOr("CORALNPU_RUN_MODULE_EXTRA_FLAGS", ""))) {
      argv.push_back(extra);
    }

    std::string command_line;
    for (const std::string& arg : argv) {
      if (!command_line.empty()) command_line.push_back(' ');
      command_line += arg;
    }
    CORALNPU_INFO("running %s", command_line.c_str());

    const int exit_code = RunProcess(argv);
    if (exit_code != 0) {
      CORALNPU_ERROR("%s failed with exit code %d: %s", kToolName, exit_code,
                     command_line.c_str());
      return kLiteRtStatusErrorRuntimeFailure;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      const TensorArg& arg = outputs[i];
      if (!arg.has_type) {
        CORALNPU_ERROR(
            "output %zu has no recorded tensor type; the runtime must call "
            "get_output_requirements before invoking",
            i);
        return kLiteRtStatusErrorInvalidArgument;
      }
      size_t expected = 0;
      CORALNPU_RETURN_IF_ERROR(TensorByteSize(arg.type, &expected));
      void* host = HostPointer(arg, runtime_context);
      if (host == nullptr) {
        CORALNPU_ERROR("output %zu is not backed by host memory", i);
        return kLiteRtStatusErrorUnsupported;
      }
      std::vector<uint8_t> contents;
      if (!ReadFileBytes(scratch + "/out" + std::to_string(i) + ".bin",
                         &contents)) {
        CORALNPU_ERROR("%s produced no data for output %zu", kToolName, i);
        return kLiteRtStatusErrorRuntimeFailure;
      }
      if (contents.size() != expected) {
        CORALNPU_ERROR(
            "output %zu is %zu bytes but the tensor type says %zu; refusing to "
            "copy a partial tensor",
            i, contents.size(), expected);
        return kLiteRtStatusErrorRuntimeFailure;
      }
      if (expected > 0) {
        std::memcpy(host, contents.data(), expected);
      }
    }
    return kLiteRtStatusOk;
  }
#endif  // !defined(_WIN32)
};

#if !defined(_WIN32)
// Wire format for the daemon executor.  All integers are little-endian:
//
//   request: u64 module hash, u32 name length, function name, u32 num inputs,
//            then per input: u32 element type, u32 rank, i32 dims[rank],
//                            u64 byte size, byte size bytes of data.
//   reply:   u32 status, u32 num outputs, then per output: u64 byte size and
//            that many bytes.
//
// The hash is of the module the byte_code points at.  A daemon serves exactly
// one module, so a mismatch means the process was started with a stale VMFB or
// the wrong one; refusing is much better than copying tensors of the wrong
// shape into it.
constexpr uint32_t kDaemonStatusOk = 0;
constexpr uint32_t kDaemonStatusError = 1;
constexpr uint32_t kDaemonStatusModuleMismatch = 2;

uint64_t ModuleHash(const LiteRtMemBuffer& bytecode) {
  const uint8_t* bytes =
      static_cast<const uint8_t*>(bytecode.base_addr) + bytecode.offset;
  uint64_t hash = 1469598103934665603ULL;
  for (size_t i = 0; i < bytecode.size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

bool WriteAll(int fd, const void* data, size_t size) {
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  while (size > 0) {
    const ssize_t written = ::write(fd, bytes, size);
    if (written <= 0) {
      return false;
    }
    bytes += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool ReadAll(int fd, void* data, size_t size) {
  uint8_t* bytes = static_cast<uint8_t*>(data);
  while (size > 0) {
    const ssize_t read = ::read(fd, bytes, size);
    if (read <= 0) {
      return false;
    }
    bytes += read;
    size -= static_cast<size_t>(read);
  }
  return true;
}

// The daemon protocol only carries fixed-width integers, so the executor is a
// thin encoder/decoder with explicit byte order.
class DaemonWriter {
 public:
  explicit DaemonWriter(int fd) : fd_(fd) {}

  bool U32(uint32_t value) { return Write(&value, sizeof(value)); }
  bool U64(uint64_t value) { return Write(&value, sizeof(value)); }
  bool I32(int32_t value) { return Write(&value, sizeof(value)); }
  bool Bytes(const void* data, size_t size) { return Write(data, size); }
  bool ok() const { return ok_; }

 private:
  bool Write(const void* data, size_t size) {
    if (!ok_) return false;
    ok_ = WriteAll(fd_, data, size);
    return ok_;
  }
  int fd_;
  bool ok_ = true;
};

class DaemonReader {
 public:
  explicit DaemonReader(int fd) : fd_(fd) {}

  bool U32(uint32_t* value) { return Read(value, sizeof(*value)); }
  bool U64(uint64_t* value) { return Read(value, sizeof(*value)); }
  bool Bytes(void* data, size_t size) { return Read(data, size); }

 private:
  bool Read(void* data, size_t size) {
    if (!ok_) return false;
    ok_ = ReadAll(fd_, data, size);
    return ok_;
  }
  int fd_;
  bool ok_ = true;
};

int ConnectUnix(const std::string& path, std::string* error) {
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) {
    *error = "socket() failed";
    return -1;
  }
  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  if (path.size() >= sizeof(addr.sun_path)) {
    *error = "socket path is too long";
    ::close(fd);
    return -1;
  }
  std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
  if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    *error = "connect() failed: " + path;
    ::close(fd);
    return -1;
  }
  return fd;
}
#endif  // !defined(_WIN32)

// Runs an invocation in a long-lived `scripts/npu_daemon.py`.
//
// The VMFB stays loaded in the daemon, so its constants are written to the NPU
// once for the whole session.  That matters for decoding: a per-token process
// would re-upload the weights on every token.
class DaemonExecutor final : public DispatchExecutor {
 public:
  LiteRtStatus Invoke(const LiteRtMemBuffer& bytecode,
                      const char* function_name,
                      const std::vector<TensorArg>& inputs,
                      const std::vector<TensorArg>& outputs,
                      const LiteRtRuntimeContext* runtime_context) override {
    if (bytecode.base_addr == nullptr || bytecode.size == 0) {
      CORALNPU_ERROR("invoke called with an empty bytecode module");
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (function_name == nullptr || function_name[0] == '\0') {
      CORALNPU_ERROR("invoke called without a function name");
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (runtime_context == nullptr ||
        runtime_context->get_tensor_buffer_host_memory == nullptr) {
      CORALNPU_ERROR(
          "the LiteRT runtime context cannot expose host memory, so the "
          "daemon executor cannot stage tensors");
      return kLiteRtStatusErrorUnsupported;
    }

#if defined(_WIN32)
    CORALNPU_ERROR("the daemon executor currently requires a POSIX host");
    return kLiteRtStatusErrorUnsupported;
#else
    const std::string socket_path = EnvOr("CORALNPU_DAEMON_SOCKET", "");
    std::string error;
    const int fd = ConnectUnix(socket_path, &error);
    if (fd < 0) {
      CORALNPU_ERROR("cannot reach the Coral NPU daemon at %s (%s)",
                     socket_path.c_str(), error.c_str());
      return kLiteRtStatusErrorNotFound;
    }

    LiteRtStatus status = Execute(fd, bytecode, function_name, inputs, outputs,
                                 runtime_context);
    ::close(fd);
    return status;
#endif  // defined(_WIN32)
  }

 private:
#if !defined(_WIN32)
  LiteRtStatus Execute(int fd, const LiteRtMemBuffer& bytecode,
                       const char* function_name,
                       const std::vector<TensorArg>& inputs,
                       const std::vector<TensorArg>& outputs,
                       const LiteRtRuntimeContext* runtime_context) {
    DaemonWriter writer(fd);
    const std::string name(function_name);
    writer.U64(ModuleHash(bytecode));
    writer.U32(static_cast<uint32_t>(name.size()));
    writer.Bytes(name.data(), name.size());
    writer.U32(static_cast<uint32_t>(inputs.size()));

    for (size_t i = 0; i < inputs.size(); ++i) {
      const TensorArg& arg = inputs[i];
      if (!arg.has_type) {
        CORALNPU_ERROR(
            "input %zu has no recorded tensor type; the runtime must call "
            "get_input_requirements before invoking",
            i);
        return kLiteRtStatusErrorInvalidArgument;
      }
      size_t byte_size = 0;
      CORALNPU_RETURN_IF_ERROR(TensorByteSize(arg.type, &byte_size));
      const void* host = HostPointer(arg, runtime_context);
      if (host == nullptr) {
        CORALNPU_ERROR("input %zu is not backed by host memory", i);
        return kLiteRtStatusErrorUnsupported;
      }
      writer.U32(static_cast<uint32_t>(arg.type.element_type));
      writer.U32(arg.type.layout.rank);
      for (unsigned int d = 0; d < arg.type.layout.rank; ++d) {
        writer.I32(static_cast<int32_t>(arg.type.layout.dimensions[d]));
      }
      writer.U64(byte_size);
      writer.Bytes(host, byte_size);
    }
    if (!writer.ok()) {
      CORALNPU_ERROR("connection to the Coral NPU daemon broke while sending "
                     "%s", function_name);
      return kLiteRtStatusErrorRuntimeFailure;
    }

    DaemonReader reader(fd);
    uint32_t reply_status = 0;
    uint32_t num_outputs = 0;
    if (!reader.U32(&reply_status) || !reader.U32(&num_outputs)) {
      CORALNPU_ERROR("the Coral NPU daemon closed the connection");
      return kLiteRtStatusErrorRuntimeFailure;
    }
    if (reply_status == kDaemonStatusModuleMismatch) {
      CORALNPU_ERROR(
          "the Coral NPU daemon is serving a different module; restart it with "
          "the VMFB this model was compiled to");
      return kLiteRtStatusErrorInvalidArgument;
    }
    if (reply_status != kDaemonStatusOk) {
      CORALNPU_ERROR("the Coral NPU daemon failed to run %s", function_name);
      return kLiteRtStatusErrorRuntimeFailure;
    }
    if (num_outputs != outputs.size()) {
      CORALNPU_ERROR("%s returned %u tensor(s) but the model expects %zu",
                     function_name, num_outputs, outputs.size());
      return kLiteRtStatusErrorRuntimeFailure;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      const TensorArg& arg = outputs[i];
      if (!arg.has_type) {
        CORALNPU_ERROR(
            "output %zu has no recorded tensor type; the runtime must call "
            "get_output_requirements before invoking",
            i);
        return kLiteRtStatusErrorInvalidArgument;
      }
      size_t expected = 0;
      CORALNPU_RETURN_IF_ERROR(TensorByteSize(arg.type, &expected));
      uint64_t received = 0;
      if (!reader.U64(&received)) {
        CORALNPU_ERROR("the Coral NPU daemon sent no data for output %zu", i);
        return kLiteRtStatusErrorRuntimeFailure;
      }
      if (received != expected) {
        CORALNPU_ERROR(
            "output %zu is %llu bytes but the tensor type says %zu; refusing to "
            "copy a partial tensor",
            i, static_cast<unsigned long long>(received), expected);
        return kLiteRtStatusErrorRuntimeFailure;
      }
      void* host = HostPointer(arg, runtime_context);
      if (host == nullptr) {
        CORALNPU_ERROR("output %zu is not backed by host memory", i);
        return kLiteRtStatusErrorUnsupported;
      }
      if (!reader.Bytes(host, expected)) {
        CORALNPU_ERROR("the Coral NPU daemon truncated output %zu", i);
        return kLiteRtStatusErrorRuntimeFailure;
      }
    }
    return kLiteRtStatusOk;
  }
#endif  // !defined(_WIN32)
};

// Picks the executor once, from the environment:
//
//   CORALNPU_DAEMON_SOCKET set -> daemon (module stays loaded in the daemon)
//   otherwise                 -> iree-run-module per invocation
class EnvironmentExecutor final : public DispatchExecutor {
 public:
  LiteRtStatus Invoke(const LiteRtMemBuffer& bytecode, const char* name,
                      const std::vector<TensorArg>& inputs,
                      const std::vector<TensorArg>& outputs,
                      const LiteRtRuntimeContext* runtime_context) override {
    if (!EnvOr("CORALNPU_DAEMON_SOCKET", "").empty()) {
      static DaemonExecutor daemon;
      return daemon.Invoke(bytecode, name, inputs, outputs, runtime_context);
    }
    static CommandExecutor command;
    return command.Invoke(bytecode, name, inputs, outputs, runtime_context);
  }
};

}  // namespace

DispatchExecutor* GetDispatchExecutor() {
  static EnvironmentExecutor executor;
  return &executor;
}

}  // namespace coralnpu
