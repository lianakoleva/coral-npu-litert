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

// Tests for the AOT compile backend that drives scripts/compile_litert.py.
//
// The driver is replaced by a shell script that writes a recognisable VMFB.
// That keeps the test hermetic: it verifies argument construction, scratch-file
// handling, error propagation, and call-info generation without needing Python,
// IREE, or a Coral NPU toolchain.

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"
#include "litert/vendors/coralnpu/compiler_plugin/coralnpu_compile_backend.h"
#include "litert/vendors/coralnpu/tests/coralnpu_test_util.h"

namespace {

constexpr char kVmfbPayload[] = "VMFB-STUB-PAYLOAD";

std::string MakeTempDir() {
  const char* tmpdir = std::getenv("TMPDIR");
  std::string tmpl = tmpdir != nullptr && tmpdir[0] != '\0' ? tmpdir : "/tmp";
  if (!tmpl.empty() && tmpl.back() != '/') {
    tmpl.push_back('/');
  }
  tmpl += "coralnpu-backend-test-XXXXXX";
  std::vector<char> buffer(tmpl.begin(), tmpl.end());
  buffer.push_back('\0');
  if (::mkdtemp(buffer.data()) == nullptr) {
    std::perror("mkdtemp");
    std::exit(2);
  }
  return std::string(buffer.data());
}

// Writes `contents` to `path` and marks it executable.
bool WriteScript(const std::string& path, const std::string& contents) {
  {
    std::ofstream out(path, std::ios::trunc);
    if (!out) return false;
    out << contents;
  }
  return ::chmod(path.c_str(), 0755) == 0;
}

// A driver that behaves like compile_litert.py: it honours --out-dir and drops
// a model-stem VMFB next to the input.
std::string SuccessfulDriverScript() {
  return R"(#!/bin/sh
out_dir=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--out-dir" ]; then out_dir="$arg"; fi
  prev="$arg"
done
if [ -z "$out_dir" ]; then
  echo "stub driver: missing --out-dir" >&2
  exit 2
fi
printf 'VMFB-STUB-PAYLOAD' > "$out_dir/partitions.vmfb"
exit 0
)";
}

std::vector<uint8_t> ToBytes(const char* text) {
  const char* end = text + std::string(text).size();
  return std::vector<uint8_t>(text, end);
}

void SetDriver(const std::string& path) {
  ::setenv("CORALNPU_AOT_DRIVER", path.c_str(), /*overwrite=*/1);
  // Run the stub through sh so the test does not depend on a usable shebang.
  ::setenv("CORALNPU_PYTHON", "/bin/sh", /*overwrite=*/1);
  ::unsetenv("CORALNPU_AOT_SEQ_LEN");
  ::unsetenv("CORALNPU_AOT_EXTRA_FLAGS");
  ::unsetenv("CORALNPU_AOT_KEEP_TEMP");
}

}  // namespace

int main() {
  const std::string work_dir = MakeTempDir();

  // --- missing configuration ------------------------------------------------
  {
    ::unsetenv("CORALNPU_AOT_DRIVER");
    coralnpu::CompileRequest request;
    const std::vector<uint8_t> model = ToBytes("model");
    request.model_data = model.data();
    request.model_size = model.size();
    request.num_partitions = 1;

    coralnpu::CompileArtifact artifact;
    CORALNPU_EXPECT_EQ(coralnpu::GetCompileBackend()->Compile(request, &artifact),
                       kLiteRtStatusErrorUnsupported);
    CORALNPU_EXPECT(artifact.byte_code.empty());
  }

  // --- invalid arguments ----------------------------------------------------
  {
    coralnpu::CompileArtifact artifact;
    coralnpu::CompileRequest request;
    request.num_partitions = 1;
    CORALNPU_EXPECT_EQ(
        coralnpu::GetCompileBackend()->Compile(request, &artifact),
        kLiteRtStatusErrorInvalidArgument);
    CORALNPU_EXPECT_EQ(coralnpu::GetCompileBackend()->Compile(request, nullptr),
                       kLiteRtStatusErrorInvalidArgument);
  }

  // --- successful compile ---------------------------------------------------
  {
    const std::string driver = work_dir + "/good_driver.sh";
    CORALNPU_EXPECT(WriteScript(driver, SuccessfulDriverScript()));
    SetDriver(driver);

    const std::vector<uint8_t> model = ToBytes("TFL3-serialized-partitions");
    coralnpu::CompileRequest request;
    request.soc_model = "coralnpu";
    request.model_data = model.data();
    request.model_size = model.size();
    request.num_partitions = 3;

    coralnpu::CompileArtifact artifact;
    CORALNPU_EXPECT_EQ(
        coralnpu::GetCompileBackend()->Compile(request, &artifact),
        kLiteRtStatusOk);

    CORALNPU_EXPECT_EQ(artifact.byte_code.size(),
                       std::string(kVmfbPayload).size());
    CORALNPU_EXPECT(std::string(artifact.byte_code.begin(),
                                artifact.byte_code.end()) == kVmfbPayload);
    CORALNPU_EXPECT_EQ(artifact.call_infos.size(), static_cast<size_t>(3));
    CORALNPU_EXPECT_STREQ(artifact.call_infos[0].c_str(), "partition_0");
    CORALNPU_EXPECT_STREQ(artifact.call_infos[1].c_str(), "partition_1");
    CORALNPU_EXPECT_STREQ(artifact.call_infos[2].c_str(), "partition_2");
  }

  // --- a populated cache serves the artifact with no driver ----------------
  {
    // First run: a driver plus a writable CORALNPU_AOT_CACHE, which stores
    // the compiled artifact under the content-hash key.
    const std::string driver = work_dir + "/cache_driver.sh";
    const std::string cache_dir = work_dir + "/aot-cache";
    CORALNPU_EXPECT(WriteScript(driver, SuccessfulDriverScript()));
    SetDriver(driver);
    ::setenv("CORALNPU_AOT_CACHE", cache_dir.c_str(), /*overwrite=*/1);

    const std::vector<uint8_t> model = ToBytes("cacheable-partitions");
    coralnpu::CompileRequest request;
    request.model_data = model.data();
    request.model_size = model.size();
    request.num_partitions = 2;

    coralnpu::CompileArtifact artifact;
    CORALNPU_EXPECT_EQ(
        coralnpu::GetCompileBackend()->Compile(request, &artifact),
        kLiteRtStatusOk);
    CORALNPU_EXPECT_EQ(static_cast<int>(artifact.call_infos.size()), 2);

    // Second run: no driver at all.  The cache hit must come before the
    // driver check — a device with a pushed artifact never needs the
    // toolchain configuration.
    ::unsetenv("CORALNPU_AOT_DRIVER");
    ::unsetenv("CORALNPU_PYTHON");
    coralnpu::CompileArtifact from_cache;
    CORALNPU_EXPECT_EQ(
        coralnpu::GetCompileBackend()->Compile(request, &from_cache),
        kLiteRtStatusOk);
    CORALNPU_EXPECT_EQ(from_cache.byte_code.size(),
                       artifact.byte_code.size());
    CORALNPU_EXPECT(from_cache.byte_code == artifact.byte_code);
    CORALNPU_EXPECT_EQ(from_cache.call_infos.size(),
                       artifact.call_infos.size());
    ::unsetenv("CORALNPU_AOT_CACHE");
  }

  // --- failing driver -------------------------------------------------------
  {
    const std::string driver = work_dir + "/bad_driver.sh";
    CORALNPU_EXPECT(WriteScript(driver, "#!/bin/sh\nexit 3\n"));
    SetDriver(driver);

    const std::vector<uint8_t> model = ToBytes("model");
    coralnpu::CompileRequest request;
    request.model_data = model.data();
    request.model_size = model.size();
    request.num_partitions = 1;

    coralnpu::CompileArtifact artifact;
    CORALNPU_EXPECT_EQ(coralnpu::GetCompileBackend()->Compile(request, &artifact),
                       kLiteRtStatusErrorCompilation);
    CORALNPU_EXPECT(artifact.byte_code.empty());
  }

  // --- driver succeeds but produces nothing ---------------------------------
  {
    const std::string driver = work_dir + "/silent_driver.sh";
    CORALNPU_EXPECT(WriteScript(driver, "#!/bin/sh\nexit 0\n"));
    SetDriver(driver);

    const std::vector<uint8_t> model = ToBytes("model");
    coralnpu::CompileRequest request;
    request.model_data = model.data();
    request.model_size = model.size();
    request.num_partitions = 1;

    coralnpu::CompileArtifact artifact;
    CORALNPU_EXPECT_EQ(coralnpu::GetCompileBackend()->Compile(request, &artifact),
                       kLiteRtStatusErrorCompilation);
  }

  // --- the configured profile reaches the driver ----------------------------
  {
    const std::string driver = work_dir + "/args_driver.sh";
    const std::string args_file = work_dir + "/args.txt";
    CORALNPU_EXPECT(WriteScript(
        driver, "#!/bin/sh\nprintf '%s\\n' \"$@\" > " + args_file + "\nexit 3\n"));
    SetDriver(driver);
    ::setenv("CORALNPU_AOT_PROFILE", "coralnpu", /*overwrite=*/1);
    ::setenv("CORALNPU_AOT_SEQ_LEN", "128", /*overwrite=*/1);

    const std::vector<uint8_t> model = ToBytes("model");
    coralnpu::CompileRequest request;
    request.model_data = model.data();
    request.model_size = model.size();
    request.num_partitions = 1;

    coralnpu::CompileArtifact artifact;
    CORALNPU_EXPECT_EQ(coralnpu::GetCompileBackend()->Compile(request, &artifact),
                       kLiteRtStatusErrorCompilation);

    std::ifstream in(args_file);
    std::string joined;
    for (std::string line; std::getline(in, line);) {
      joined += line + "\n";
    }
    CORALNPU_EXPECT(joined.find("--out-dir\n") != std::string::npos);
    CORALNPU_EXPECT(joined.find("--profile\ncoralnpu\n") != std::string::npos);
    CORALNPU_EXPECT(joined.find("--seq-len\n128\n") != std::string::npos);
    CORALNPU_EXPECT(joined.find("--allow-untiled-attention\n") !=
                    std::string::npos);
    CORALNPU_EXPECT(joined.find("partitions.tflite\n") != std::string::npos);

    ::unsetenv("CORALNPU_AOT_SEQ_LEN");
  }

  return CORALNPU_TEST_MAIN_RESULT();
}
