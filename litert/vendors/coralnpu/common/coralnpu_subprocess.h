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

// POSIX helpers for the two Coral NPU backends that shell out.
//
// Both the AOT compile backend (scripts/compile_litert.py) and the command
// executor (iree-run-module) need the same three things: a private scratch
// directory, a file write that reports failure, and a child process whose exit
// status is visible.  Keeping one implementation means one place where fork,
// exec, EINTR, and partial writes are handled.
//
// Windows is not supported by either caller; the header compiles to nothing
// useful there and the callers report kLiteRtStatusErrorUnsupported.

#ifndef ODML_LITERT_LITERT_VENDORS_CORALNPU_COMMON_CORALNPU_SUBPROCESS_H_
#define ODML_LITERT_LITERT_VENDORS_CORALNPU_COMMON_CORALNPU_SUBPROCESS_H_

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "litert/vendors/coralnpu/common/coralnpu_common.h"

#if !defined(_WIN32)
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace coralnpu {

// Returns the value of `name`, or `fallback` when unset/empty.
inline std::string EnvOr(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return value;
}

// True when `name` is set to a non-empty value.
inline bool FlagSet(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0';
}

// Splits on whitespace.  Arguments that may contain spaces (paths) must be
// passed as their own environment variable instead of going through this.
inline std::vector<std::string> SplitWhitespace(const std::string& text) {
  std::vector<std::string> parts;
  size_t i = 0;
  while (i < text.size()) {
    while (i < text.size() &&
           std::isspace(static_cast<unsigned char>(text[i]))) {
      ++i;
    }
    const size_t start = i;
    while (i < text.size() &&
           !std::isspace(static_cast<unsigned char>(text[i]))) {
      ++i;
    }
    if (i > start) {
      parts.emplace_back(text.substr(start, i - start));
    }
  }
  return parts;
}

// Creates `path` including every missing parent, in the manner of
// `mkdir -p`.  True when the directory exists afterwards.  The AOT cache and
// the executor's staging points are caller-configured paths, so a missing
// component of one must not turn every later artifact write into a
// mysterious failure.
inline bool MakeDirectories(const std::string& path) {
  if (path.empty()) {
    return false;
  }
  size_t start = path[0] == '/' ? 1 : 0;
  while (start <= path.size()) {
    size_t end = path.find('/', start);
    if (end == std::string::npos) {
      end = path.size();
    }
    const std::string component = path.substr(0, end);
    if (!component.empty() && component != "/") {
      if (::mkdir(component.c_str(), 0777) != 0 && errno != EEXIST) {
        // A non-directory file standing in the path is fatal, like mkdir -p.
        return false;
      }
    }
    start = end + 1;
  }
  struct stat st;
  return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// Creates an empty directory under $TMPDIR (or /tmp) named `prefix` + random
// suffix.  Returns false if the directory could not be created.
inline bool MakeScratchDir(const char* prefix, std::string* out_dir) {
  if (out_dir == nullptr) {
    return false;
  }
  std::string templ = EnvOr("TMPDIR", "/tmp");
  if (!templ.empty() && templ.back() != '/') {
    templ.push_back('/');
  }
  templ += prefix;
  templ += "XXXXXX";
  std::vector<char> buffer(templ.begin(), templ.end());
  buffer.push_back('\0');
  if (::mkdtemp(buffer.data()) == nullptr) {
    return false;
  }
  *out_dir = buffer.data();
  return true;
}

// Removes `dir` and every file in `names` (entries without a path separator are
// taken relative to `dir`).  Best effort: a partial cleanup is not an error.
inline void RemoveScratchDir(const std::string& dir,
                             const std::vector<std::string>& names) {
  for (const std::string& name : names) {
    ::unlink((dir + "/" + name).c_str());
  }
  ::rmdir(dir.c_str());
}

inline bool WriteFileBytes(const std::string& path, const uint8_t* data,
                           size_t size) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    return false;
  }
  if (size > 0) {
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(size));
  }
  return out.good();
}

inline bool WriteFileText(const std::string& path, const std::string& text) {
  return WriteFileBytes(path, reinterpret_cast<const uint8_t*>(text.data()),
                        text.size());
}

// Reads the whole file into `out`.  Returns false if it cannot be opened.
inline bool ReadFileBytes(const std::string& path, std::vector<uint8_t>* out) {
  if (out == nullptr) {
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  in.seekg(0, std::ios::end);
  const std::streamoff size = in.tellg();
  if (size < 0) {
    return false;
  }
  in.seekg(0, std::ios::beg);
  out->resize(static_cast<size_t>(size));
  if (size > 0) {
    in.read(reinterpret_cast<char*>(out->data()), size);
  }
  return in.good() || in.eof();
}

#if !defined(_WIN32)

// Runs `argv` to completion and returns its exit status, or -1 if the process
// could not be started or reaped.  The child inherits stdin/stdout/stderr, so
// diagnostics from the tool reach the user unchanged.
inline int RunProcess(const std::vector<std::string>& argv) {
  if (argv.empty()) {
    return -1;
  }
  std::vector<char*> raw;
  raw.reserve(argv.size() + 1);
  for (const std::string& arg : argv) {
    raw.push_back(const_cast<char*>(arg.c_str()));
  }
  raw.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) {
    CORALNPU_ERROR("fork failed while starting %s", argv[0].c_str());
    return -1;
  }
  if (pid == 0) {
    ::execvp(raw[0], raw.data());
    // execvp only returns on failure.  _exit avoids running the parent's
    // atexit handlers and flushing its buffers twice.
    std::fprintf(stderr, "CoralNPU error: cannot execute %s: %s\n", raw[0],
                 std::strerror(errno));
    ::_exit(127);
  }

  int status = 0;
  while (::waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      CORALNPU_ERROR("waitpid failed while running %s", argv[0].c_str());
      return -1;
    }
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    CORALNPU_ERROR("%s terminated by signal %d", argv[0].c_str(),
                   WTERMSIG(status));
    return -1;
  }
  return -1;
}

// True when `name` names an executable that can be found and run.  Uses
// access(X_OK) on the resolved path rather than running it.
inline bool IsExecutableFile(const std::string& path) {
  return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

// True when `path` is a readable regular file.
inline bool IsReadableFile(const std::string& path) {
  return !path.empty() && ::access(path.c_str(), R_OK) == 0;
}

// Finds `name` on $PATH.  Returns an empty string when it is not there.
inline std::string FindOnPath(const char* name) {
  const char* path_env = std::getenv("PATH");
  if (path_env == nullptr) {
    return std::string();
  }
  std::string paths = path_env;
  size_t start = 0;
  while (start <= paths.size()) {
    const size_t end = paths.find(':', start);
    const std::string dir =
        end == std::string::npos ? paths.substr(start)
                                 : paths.substr(start, end - start);
    if (!dir.empty()) {
      const std::string candidate = dir + "/" + name;
      if (IsExecutableFile(candidate)) {
        return candidate;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return std::string();
}

#else  // defined(_WIN32)

inline int RunProcess(const std::vector<std::string>& argv) { return -1; }
inline bool IsExecutableFile(const std::string& path) { return false; }
inline bool IsReadableFile(const std::string& path) { return false; }
inline std::string FindOnPath(const char* name) { return std::string(); }

#endif  // !defined(_WIN32)

}  // namespace coralnpu

#endif  // ODML_LITERT_LITERT_VENDORS_CORALNPU_COMMON_CORALNPU_SUBPROCESS_H_
