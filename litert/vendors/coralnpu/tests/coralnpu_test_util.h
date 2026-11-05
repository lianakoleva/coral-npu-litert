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

// Tiny assertion helpers for the Coral NPU vendor tests.
//
// The vendor libraries are loaded by the stock LiteRT runtime and therefore
// cannot depend on LiteRT's C++ libraries or on absl.  Their tests follow the
// same rule: no gtest, no gmock.  Each test is a plain executable that prints
// the first failure and exits non-zero, which is enough to drive them from
// CTest, from a Bazel `sh_test`, or by hand over `adb shell`.

#ifndef ODML_LITERT_LITERT_VENDORS_CORALNPU_TESTS_CORALNPU_TEST_UTIL_H_
#define ODML_LITERT_LITERT_VENDORS_CORALNPU_TESTS_CORALNPU_TEST_UTIL_H_

#include <cstdio>
#include <cstdlib>
#include <string>

namespace coralnpu {
namespace testing {

inline int& FailureCount() {
  static int failures = 0;
  return failures;
}

}  // namespace testing
}  // namespace coralnpu

// Fails the current test case but keeps going, so one run reports every
// problem instead of only the first.
#define CORALNPU_EXPECT(condition)                                          \
  do {                                                                      \
    if (!(condition)) {                                                     \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                   #condition);                                             \
      ++::coralnpu::testing::FailureCount();                                \
    }                                                                       \
  } while (0)

#define CORALNPU_EXPECT_EQ(actual, expected)                                \
  do {                                                                      \
    const auto coralnpu_actual_ = (actual);                                 \
    const auto coralnpu_expected_ = (expected);                             \
    if (!(coralnpu_actual_ == coralnpu_expected_)) {                        \
      std::fprintf(stderr, "FAIL %s:%d: %s == %s\n", __FILE__, __LINE__,    \
                   #actual, #expected);                                     \
      ++::coralnpu::testing::FailureCount();                                \
    }                                                                       \
  } while (0)

#define CORALNPU_EXPECT_STREQ(actual, expected)                             \
  do {                                                                      \
    const char* coralnpu_actual_ = (actual);                                \
    const char* coralnpu_expected_ = (expected);                            \
    if (coralnpu_actual_ == nullptr || coralnpu_expected_ == nullptr ||     \
        std::string(coralnpu_actual_) != std::string(coralnpu_expected_)) { \
      std::fprintf(stderr, "FAIL %s:%d: \"%s\" == \"%s\"\n", __FILE__,      \
                   __LINE__, coralnpu_actual_ ? coralnpu_actual_ : "(null)",\
                   coralnpu_expected_ ? coralnpu_expected_ : "(null)");     \
      ++::coralnpu::testing::FailureCount();                                \
    }                                                                       \
  } while (0)

// Returns the process exit code for a test's main().
#define CORALNPU_TEST_MAIN_RESULT()                                        \
  (::coralnpu::testing::FailureCount() == 0                                 \
       ? 0                                                                  \
       : (std::fprintf(stderr, "%d check(s) failed\n",                      \
                       ::coralnpu::testing::FailureCount()),                \
          1))

#endif  // ODML_LITERT_LITERT_VENDORS_CORALNPU_TESTS_CORALNPU_TEST_UTIL_H_
