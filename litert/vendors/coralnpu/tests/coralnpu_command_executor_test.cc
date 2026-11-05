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

// End-to-end test for the command executor: ABI -> buffer registry -> staging
// -> subprocess -> copy back.
//
// `iree-run-module` is replaced by a shell script, so this runs anywhere and
// checks the parts that actually break in practice:
//
//   * the module is written to disk byte for byte,
//   * each input becomes a valid .npy with the right dtype and shape,
//   * the device/function flags reach the tool,
//   * output bytes land in the registered output buffer, and
//   * a size mismatch is refused instead of silently truncating.
//
// The real tool is exercised by `coralnpu_dlopen_smoke_test` when it is given
// a VMFB; pass CORALNPU_RUN_MODULE to point it at the runtime under test. The
// board recipe (compile for aarch64, push, run) is in docs/litert-backend.md.

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model_types.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/coralnpu/common/coralnpu_subprocess.h"
#include "litert/vendors/coralnpu/dispatch/coralnpu_executor.h"
#include "litert/vendors/coralnpu/tests/coralnpu_test_util.h"

namespace {

// --- fake runtime context ----------------------------------------------------

// Handle values are just small integers; the runtime context maps them to real
// host pointers, exactly like the LiteRT runtime would.
void* g_host_pointers[8] = {};

LiteRtStatus FakeGetTensorBufferHostMemory(LiteRtTensorBuffer tensor_buffer,
                                          void** host_memory_addr) {
  if (host_memory_addr == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const auto index = reinterpret_cast<uintptr_t>(tensor_buffer);
  if (index >= 8) {
    return kLiteRtStatusErrorNotFound;
  }
  *host_memory_addr = g_host_pointers[index];
  return *host_memory_addr != nullptr ? kLiteRtStatusOk
                                      : kLiteRtStatusErrorNotFound;
}

LiteRtRankedTensorType MakeTensorType(std::vector<int32_t> dims,
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

std::string MakeTempDir() {
  std::string dir;
  if (!coralnpu::MakeScratchDir("coralnpu-run-test-", &dir)) {
    std::perror("MakeScratchDir");
    std::exit(2);
  }
  return dir;
}

bool WriteScript(const std::string& path, const std::string& contents) {
  if (!coralnpu::WriteFileText(path, contents)) return false;
  return ::chmod(path.c_str(), 0755) == 0;
}

// The stub stands in for iree-run-module.  It captures its arguments and the
// staged input, then emits a fixed payload as the output.
std::string StubToolScript() {
  return R"SH(#!/bin/sh
set -e
out=""
in=""
mod=""
fn=""
dev=""
for a in "$@"; do
  case "$a" in
    --output=@*)   out="${a#--output=@}" ;;
    --input=@*)    in="${a#--input=@}" ;;
    --module=*)    mod="${a#--module=}" ;;
    --function=*)  fn="${a#--function=}" ;;
    --device=*)    dev="${a#--device=}" ;;
  esac
done
[ -n "$mod" ] || exit 10
[ -n "$fn" ] || exit 11
[ -n "$in" ] || exit 12
[ -n "$out" ] || exit 13
[ -s "$mod" ] || exit 14
[ -f "$in" ] || exit 15
cp "$in" "$CAPTURE/in0.npy"
printf '%s|%s|%s\n' "$dev" "$fn" "$(basename "$mod")" > "$CAPTURE/args.txt"
cp "$PAYLOAD" "$out"
exit 0
)SH";
}

// Parses "=f4"-style descr and "(2, 3)"-style shape out of an npy header and
// checks it against `expected`.  Also verifies the payload after the header.
bool CheckNpy(const std::string& path, const std::string& expected_descr,
              const std::string& expected_shape,
              const std::vector<uint8_t>& expected_payload) {
  std::vector<uint8_t> bytes;
  if (!coralnpu::ReadFileBytes(path, &bytes)) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    return false;
  }
  static const uint8_t kMagic[] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  if (bytes.size() < 10 ||
      std::memcmp(bytes.data(), kMagic, sizeof(kMagic)) != 0) {
    std::fprintf(stderr, "%s: bad npy magic\n", path.c_str());
    return false;
  }
  const size_t header_len =
      static_cast<size_t>(bytes[8]) | (static_cast<size_t>(bytes[9]) << 8);
  if (bytes.size() < 10 + header_len) {
    std::fprintf(stderr, "%s: truncated header\n", path.c_str());
    return false;
  }
  const std::string header(reinterpret_cast<const char*>(bytes.data()) + 10,
                           header_len);
  if (header.back() != '\n') {
    std::fprintf(stderr, "%s: header is not newline terminated\n", path.c_str());
    return false;
  }
  // numpy requires the total header length to be a multiple of 64.
  if ((10 + header_len) % 64 != 0) {
    std::fprintf(stderr, "%s: header length %zu is not 64-byte aligned\n",
                 path.c_str(), 10 + header_len);
    return false;
  }
  if (header.find("'descr': '" + expected_descr + "'") == std::string::npos) {
    std::fprintf(stderr, "%s: expected descr '%s' in header: %s\n", path.c_str(),
                 expected_descr.c_str(), header.c_str());
    return false;
  }
  if (header.find("'shape': " + expected_shape) == std::string::npos) {
    std::fprintf(stderr, "%s: expected shape %s in header: %s\n", path.c_str(),
                 expected_shape.c_str(), header.c_str());
    return false;
  }
  const uint8_t* payload = bytes.data() + 10 + header_len;
  const size_t payload_size = bytes.size() - 10 - header_len;
  if (payload_size != expected_payload.size() ||
      std::memcmp(payload, expected_payload.data(), payload_size) != 0) {
    std::fprintf(stderr, "%s: payload mismatch (%zu vs %zu bytes)\n",
                 path.c_str(), payload_size, expected_payload.size());
    return false;
  }
  return true;
}

std::string ReadText(const std::string& path) {
  std::vector<uint8_t> bytes;
  if (!coralnpu::ReadFileBytes(path, &bytes)) return std::string();
  return std::string(bytes.begin(), bytes.end());
}

}  // namespace

int main() {
  const std::string work = MakeTempDir();

  // The stub needs to know where to capture and what to emit.
  const std::string capture = work + "/capture";
  ::mkdir(capture.c_str(), 0755);
  const std::vector<uint8_t> payload = {0,  1,  2,  3,  4,  5,  6,  7,
                                        8,  9,  10, 11, 12, 13, 14, 15,
                                        16, 17, 18, 19, 20, 21, 22, 23};
  const std::string payload_path = work + "/payload.bin";
  CORALNPU_EXPECT(coralnpu::WriteFileBytes(payload_path, payload.data(),
                                          payload.size()));

  const std::string tool = work + "/iree-run-module";
  CORALNPU_EXPECT(WriteScript(tool, StubToolScript()));

  ::setenv("CORALNPU_RUN_MODULE", tool.c_str(), /*overwrite=*/1);
  ::setenv("CORALNPU_DEVICE", "local-task", /*overwrite=*/1);
  ::setenv("CAPTURE", capture.c_str(), /*overwrite=*/1);
  ::setenv("PAYLOAD", payload_path.c_str(), /*overwrite=*/1);
  ::unsetenv("CORALNPU_DISPATCH_KEEP_TEMP");
  ::unsetenv("CORALNPU_RUN_MODULE_EXTRA_FLAGS");

  // 2x3 float32 input and output, 24 bytes each.
  const std::vector<int32_t> dims = {2, 3};
  const LiteRtRankedTensorType float_type =
      MakeTensorType(dims, kLiteRtElementTypeFloat32);

  std::vector<float> input_values = {1.5f, 2.5f, 3.5f, -1.0f, -2.0f, 0.25f};
  std::vector<float> output_values(6, 0.0f);
  g_host_pointers[1] = input_values.data();
  g_host_pointers[2] = output_values.data();

  LiteRtRuntimeContext runtime_context{};
  runtime_context.abi_header.struct_size = sizeof(LiteRtRuntimeContext);
  runtime_context.abi_header.major_version = 1;
  runtime_context.get_tensor_buffer_host_memory =
      FakeGetTensorBufferHostMemory;

  coralnpu::TensorArg input;
  input.buffer = reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(1));
  input.type = float_type;
  input.has_type = true;

  coralnpu::TensorArg output;
  output.buffer = reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(2));
  output.type = float_type;
  output.has_type = true;

  const std::vector<uint8_t> module = {'V', 'M', 'F', 'B', '-', 'S', 'T', 'U',
                                       'B'};
  LiteRtMemBuffer bytecode{};
  bytecode.fd = -1;
  bytecode.base_addr = module.data();
  bytecode.offset = 0;
  bytecode.size = module.size();

  // --- happy path -----------------------------------------------------------
  {
    const LiteRtStatus status = coralnpu::GetDispatchExecutor()->Invoke(
        bytecode, "partition_0", {input}, {output}, &runtime_context);
    CORALNPU_EXPECT_EQ(status, kLiteRtStatusOk);

    // The tool saw the device and function the ABI supplied.
    const std::string args_seen = ReadText(capture + "/args.txt");
    CORALNPU_EXPECT_STREQ(args_seen.c_str(),
                          "local-task|partition_0|module.vmfb\n");

    // The staged input is a well-formed npy holding the input bytes.
    const std::vector<uint8_t> expected_input_bytes(
        reinterpret_cast<const uint8_t*>(input_values.data()),
        reinterpret_cast<const uint8_t*>(input_values.data()) +
            input_values.size() * sizeof(float));
    CORALNPU_EXPECT(CheckNpy(capture + "/in0.npy", "=f4", "(2, 3)",
                             expected_input_bytes));

    // The output buffer received exactly the tool's payload.
    std::vector<uint8_t> got(reinterpret_cast<const uint8_t*>(output_values.data()),
                             reinterpret_cast<const uint8_t*>(output_values.data()) +
                                 output_values.size() * sizeof(float));
    CORALNPU_EXPECT(got == payload);
  }

  // --- output size mismatch is refused --------------------------------------
  {
    // Claim a bigger output than the tool produced; the executor must not
    // memcpy past the tool's data.
    const std::vector<int32_t> bigger = {2, 6};
    coralnpu::TensorArg wrong = output;
    wrong.type = MakeTensorType(bigger, kLiteRtElementTypeFloat32);

    std::vector<float> scratch(12, -7.0f);
    g_host_pointers[3] = scratch.data();
    wrong.buffer =
        reinterpret_cast<LiteRtTensorBuffer>(static_cast<uintptr_t>(3));

    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "partition_0", {input}, {wrong},
                           &runtime_context),
                       kLiteRtStatusErrorRuntimeFailure);
    // Nothing was written, so the buffer still holds its sentinel values.
    CORALNPU_EXPECT_EQ(scratch[0], -7.0f);
    CORALNPU_EXPECT_EQ(scratch[11], -7.0f);
  }

  // --- missing tensor type is refused ---------------------------------------
  {
    coralnpu::TensorArg untyped = input;
    untyped.has_type = false;
    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "partition_0", {untyped}, {output},
                           &runtime_context),
                       kLiteRtStatusErrorInvalidArgument);
  }

  // --- element type numpy cannot express ------------------------------------
  {
    coralnpu::TensorArg int4 = input;
    int4.type = MakeTensorType({3}, kLiteRtElementTypeInt4);
    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "partition_0", {int4}, {output},
                           &runtime_context),
                       kLiteRtStatusErrorUnsupported);
  }

  // --- failing tool ----------------------------------------------------------
  {
    const std::string failing = work + "/fail-tool";
    CORALNPU_EXPECT(WriteScript(failing, "#!/bin/sh\nexit 7\n"));
    ::setenv("CORALNPU_RUN_MODULE", failing.c_str(), /*overwrite=*/1);
    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "partition_0", {input}, {output},
                           &runtime_context),
                       kLiteRtStatusErrorRuntimeFailure);
    ::setenv("CORALNPU_RUN_MODULE", tool.c_str(), /*overwrite=*/1);
  }

  // --- missing tool ----------------------------------------------------------
  {
    ::setenv("CORALNPU_RUN_MODULE", (work + "/does-not-exist").c_str(),
             /*overwrite=*/1);
    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "partition_0", {input}, {output},
                           &runtime_context),
                       kLiteRtStatusErrorNotFound);
  }

  // --- bad arguments ----------------------------------------------------------
  {
    ::unsetenv("CORALNPU_RUN_MODULE");
    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "", {input}, {output}, &runtime_context),
                       kLiteRtStatusErrorInvalidArgument);
    CORALNPU_EXPECT_EQ(coralnpu::GetDispatchExecutor()->Invoke(
                           bytecode, "partition_0", {input}, {output}, nullptr),
                       kLiteRtStatusErrorUnsupported);
  }

  return CORALNPU_TEST_MAIN_RESULT();
}
