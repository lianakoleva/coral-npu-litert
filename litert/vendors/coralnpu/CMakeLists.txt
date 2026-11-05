# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standalone build for the Coral NPU LiteRT vendor libraries.
#
# These are the two shared objects the stock LiteRT runtime loads:
#
#   libLiteRtCompilerPlugin_CoralNpu.so   graph -> VMFB (AOT, build time)
#   libLiteRtDispatch_CoralNpu.so         VMFB -> Coral NPU (run time)
#
# Neither links libLiteRt: they speak only the versioned C interface tables in
# litert/vendors/c/.  That is what lets them be built and shipped separately
# from a LiteRT release.
#
#   cmake -S litert/vendors/coralnpu -B build/coralnpu \
#       -DLITERT_SOURCE_ROOT=/path/to/LiteRT
#   cmake --build build/coralnpu
#   ctest --test-dir build/coralnpu --output-on-failure
#
# The same file builds natively on the board (aarch64 Linux, g++ 13) with no
# extra flags; that is the configuration the ABI tests are validated against.

cmake_minimum_required(VERSION 3.20)
project(coralnpu_litert_vendors CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LITERT_SOURCE_ROOT)
  message(FATAL_ERROR
    "LITERT_SOURCE_ROOT must point at a LiteRT checkout (the directory "
    "containing litert/c/litert_common.h)")
endif()

set(CORALNPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(LITERT_ROOT "${LITERT_SOURCE_ROOT}")

# Vendor sources include each other as "litert/vendors/coralnpu/...", so the
# include root is the directory containing `litert/`.  When this package is
# checked into a LiteRT tree that is LITERT_ROOT; when it lives in a separate
# workspace it is two levels up from this file.
get_filename_component(CORALNPU_WORKSPACE_ROOT "${CORALNPU_ROOT}/../../.." ABSOLUTE)

# Exactly one executor is linked into the dispatch library.  The default
# (`off`/`off`) builds the ABI without an execution backend, which is enough to
# conformance-test the interface and to ship as a plugin skeleton.
option(CORALNPU_WITH_COMMAND_EXECUTOR
       "Run modules through iree-run-module as a subprocess (needs no IREE at build time)"
       OFF)
option(CORALNPU_WITH_IREE_RUNTIME
       "Link the IREE runtime and the Coral NPU HAL driver into the dispatch library (not implemented yet)"
       OFF)
option(CORALNPU_BUILD_TESTS "Build the vendor ABI tests" ON)

if(CORALNPU_WITH_COMMAND_EXECUTOR AND CORALNPU_WITH_IREE_RUNTIME)
  message(FATAL_ERROR
    "CORALNPU_WITH_COMMAND_EXECUTOR and CORALNPU_WITH_IREE_RUNTIME are "
    "mutually exclusive: both provide GetDispatchExecutor()")
endif()

set(CORALNPU_WARNING_FLAGS
    -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)

# ---------------------------------------------------------------------------
# Compiler plugin
# ---------------------------------------------------------------------------

add_library(LiteRtCompilerPlugin_CoralNpu SHARED
  compiler_plugin/coralnpu_compiler_plugin.cc
  compiler_plugin/aot_compile_backend.cc)
target_include_directories(LiteRtCompilerPlugin_CoralNpu PRIVATE
  "${CORALNPU_WORKSPACE_ROOT}"
  "${CORALNPU_ROOT}"
  "${LITERT_ROOT}")
target_compile_options(LiteRtCompilerPlugin_CoralNpu PRIVATE
  ${CORALNPU_WARNING_FLAGS})
# Vendor libraries are dlopen()ed into the LiteRT process; keeping them free of
# an rpath to libLiteRt avoids picking up a mismatched runtime.
set_target_properties(LiteRtCompilerPlugin_CoralNpu PROPERTIES
  OUTPUT_NAME "LiteRtCompilerPlugin_CoralNpu"
  PREFIX "lib")

# ---------------------------------------------------------------------------
# Dispatch library
# ---------------------------------------------------------------------------

set(CORALNPU_DISPATCH_EXECUTOR
    dispatch/coralnpu_executor_unavailable.cc)
if(CORALNPU_WITH_COMMAND_EXECUTOR)
  set(CORALNPU_DISPATCH_EXECUTOR dispatch/coralnpu_executor_command.cc)
endif()

add_library(LiteRtDispatch_CoralNpu SHARED
  dispatch/coralnpu_dispatch.cc
  ${CORALNPU_DISPATCH_EXECUTOR})
target_include_directories(LiteRtDispatch_CoralNpu PRIVATE
  "${CORALNPU_WORKSPACE_ROOT}"
  "${CORALNPU_ROOT}"
  "${LITERT_ROOT}")
target_compile_options(LiteRtDispatch_CoralNpu PRIVATE
  ${CORALNPU_WARNING_FLAGS})
target_compile_definitions(LiteRtDispatch_CoralNpu PRIVATE
  $<$<BOOL:${CORALNPU_WITH_COMMAND_EXECUTOR}>:CORALNPU_WITH_COMMAND_EXECUTOR=1>
  $<$<BOOL:${CORALNPU_WITH_IREE_RUNTIME}>:CORALNPU_WITH_IREE_RUNTIME=1>)
set_target_properties(LiteRtDispatch_CoralNpu PROPERTIES
  OUTPUT_NAME "LiteRtDispatch_CoralNpu"
  PREFIX "lib")

if(CORALNPU_WITH_IREE_RUNTIME)
  message(FATAL_ERROR
    "CORALNPU_WITH_IREE_RUNTIME is not implemented yet.\n"
    "The in-process executor (dispatch/coralnpu_executor_iree.cc) needs a "
    "prebuilt IREE runtime that exposes the coralnpu HAL driver, and no such "
    "build is available to test against, so it is deliberately not shipped as "
    "untested code.\n"
    "Use -DCORALNPU_WITH_COMMAND_EXECUTOR=ON to run modules through "
    "iree-run-module, which reaches the same Coral NPU via --device=torq and "
    "needs nothing at build time.")
endif()

# ---------------------------------------------------------------------------
# Tests
#
# All of these are plain executables with no gtest dependency, because the
# vendor libraries must not depend on LiteRT's C++ libraries (or absl) and
# their tests follow the same rule.  They run on the host and on the board.
# ---------------------------------------------------------------------------

if(CORALNPU_BUILD_TESTS)
  enable_testing()

  function(coralnpu_test target)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} PRIVATE
      "${CORALNPU_WORKSPACE_ROOT}" "${CORALNPU_ROOT}" "${LITERT_ROOT}")
    target_compile_options(${target} PRIVATE ${CORALNPU_WARNING_FLAGS})
  endfunction()

  # --- dispatch ABI, linked against the default (unavailable) executor -------
  coralnpu_test(coralnpu_dispatch_abi_test
    tests/coralnpu_dispatch_abi_test.cc
    dispatch/coralnpu_dispatch.cc
    dispatch/coralnpu_executor_unavailable.cc)
  add_test(NAME coralnpu_dispatch_abi COMMAND coralnpu_dispatch_abi_test)

  # --- AOT compile backend --------------------------------------------------
  coralnpu_test(coralnpu_compile_backend_test
    tests/coralnpu_compile_backend_test.cc
    compiler_plugin/aot_compile_backend.cc)
  add_test(NAME coralnpu_compile_backend COMMAND coralnpu_compile_backend_test)

  # --- compiler plugin ABI --------------------------------------------------
  coralnpu_test(coralnpu_compiler_plugin_metadata_test
    tests/coralnpu_compiler_plugin_metadata_test.cc
    compiler_plugin/coralnpu_compiler_plugin.cc
    compiler_plugin/aot_compile_backend.cc)
  add_test(NAME coralnpu_compiler_plugin_metadata
           COMMAND coralnpu_compiler_plugin_metadata_test)

  # --- command executor, always built ---------------------------------------
  # Built even when the dispatch library uses a different executor: this is the
  # test for the executor itself, not for the shipped configuration.
  coralnpu_test(coralnpu_command_executor_test
    tests/coralnpu_command_executor_test.cc
    dispatch/coralnpu_executor_command.cc)
  add_test(NAME coralnpu_command_executor
           COMMAND coralnpu_command_executor_test)

  # --- dlopen the real artefact and drive the ABI ---------------------------
  coralnpu_test(coralnpu_dlopen_smoke_test
    tests/coralnpu_dlopen_smoke_test.cc)
  target_link_libraries(coralnpu_dlopen_smoke_test PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(coralnpu_dlopen_smoke_test LiteRtDispatch_CoralNpu)
  add_test(NAME coralnpu_dlopen_smoke
           COMMAND coralnpu_dlopen_smoke_test
                   $<TARGET_FILE:LiteRtDispatch_CoralNpu>)
endif()
