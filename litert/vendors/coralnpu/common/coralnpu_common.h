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

#ifndef ODML_LITERT_LITERT_VENDORS_CORALNPU_COMMON_CORALNPU_COMMON_H_
#define ODML_LITERT_LITERT_VENDORS_CORALNPU_COMMON_CORALNPU_COMMON_H_

// Small, dependency-free helpers shared by the Coral NPU Compiler Plugin and
// Dispatch libraries.  These libraries are loaded by the stock LiteRT runtime,
// so they must not depend on LiteRT internal C++ libraries (absl, litert::cc).
// Only the public C ABI headers are used.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

#include "litert/c/litert_common.h"
#include "litert/c/litert_layout.h"
#include "litert/c/litert_model_types.h"

#define CORALNPU_RETURN_IF_ERROR(expr)             \
  do {                                             \
    ::LiteRtStatus coralnpu_status_ = (expr);      \
    if (coralnpu_status_ != kLiteRtStatusOk) {     \
      return coralnpu_status_;                     \
    }                                              \
  } while (0)

// Vendor libraries should use the LiteRT logger so that `min_logger_severity`
// is honored.  That helper lives in a C++ internal header which is not part of
// the vendor ABI surface, so keep this to stderr.  It is intentionally quiet
// unless the library was built with CORALNPU_VERBOSE_LOGGING, except for
// errors.
#if defined(CORALNPU_VERBOSE_LOGGING)
#define CORALNPU_INFO(...)                                        \
  do {                                                            \
    std::fprintf(stderr, "CoralNPU info %s:%d: ", __FILE__,       \
                 __LINE__);                                       \
    std::fprintf(stderr, __VA_ARGS__);                            \
    std::fprintf(stderr, "\n");                                   \
  } while (0)
#else
#define CORALNPU_INFO(...) \
  do {                     \
  } while (0)
#endif

#define CORALNPU_ERROR(...)                                       \
  do {                                                            \
    std::fprintf(stderr, "CoralNPU error %s:%d: ", __FILE__,      \
                 __LINE__);                                       \
    std::fprintf(stderr, __VA_ARGS__);                            \
    std::fprintf(stderr, "\n");                                   \
  } while (0)

namespace coralnpu {

// Returns the number of bits in one element of `element_type`, or 0 if the
// type cannot be represented as a flat byte buffer (strings, resources,
// variants).
inline int ElementTypeBits(LiteRtElementType element_type) {
  switch (element_type) {
    case kLiteRtElementTypeBool:
      // TFLite bool is one byte.
      return 8;
    case kLiteRtElementTypeInt2:
      return 2;
    case kLiteRtElementTypeInt4:
    case kLiteRtElementTypeUInt4:
      return 4;
    case kLiteRtElementTypeInt8:
    case kLiteRtElementTypeUInt8:
    case kLiteRtElementTypeFloat8E4M3FN:
    case kLiteRtElementTypeFloat8E5M2:
      return 8;
    case kLiteRtElementTypeInt16:
    case kLiteRtElementTypeUInt16:
    case kLiteRtElementTypeFloat16:
    case kLiteRtElementTypeBFloat16:
      return 16;
    case kLiteRtElementTypeInt32:
    case kLiteRtElementTypeUInt32:
    case kLiteRtElementTypeFloat32:
      return 32;
    case kLiteRtElementTypeInt64:
    case kLiteRtElementTypeUInt64:
    case kLiteRtElementTypeFloat64:
    case kLiteRtElementTypeComplex64:
      return 64;
    case kLiteRtElementTypeComplex128:
      return 128;
    default:
      return 0;
  }
}

// Computes the tightly packed byte size of a static ranked tensor.
//
// Coral NPU requires fully static shapes, so dynamic dimensions and explicit
// strides are rejected here rather than silently materialized.
inline LiteRtStatus TensorByteSize(const LiteRtRankedTensorType& tensor_type,
                                   size_t* out_size) {
  if (out_size == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const LiteRtLayout& layout = tensor_type.layout;
  if (layout.rank > LITERT_TENSOR_MAX_RANK) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (layout.has_strides) {
    // Host buffers handed to the dispatch library are contiguous.
    return kLiteRtStatusErrorUnsupported;
  }
  const int element_bits = ElementTypeBits(tensor_type.element_type);
  if (element_bits == 0) {
    return kLiteRtStatusErrorUnsupported;
  }

  size_t num_elements = 1;
  for (unsigned int i = 0; i < layout.rank; ++i) {
    const int32_t dim = layout.dimensions[i];
    if (dim < 0) {
      // Dynamic dimension: the compiler plugin is responsible for binding it
      // before dispatch ever sees the tensor.
      return kLiteRtStatusErrorUnsupported;
    }
    if (dim == 0) {
      *out_size = 0;
      return kLiteRtStatusOk;
    }
    if (num_elements >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    num_elements *= static_cast<size_t>(dim);
  }

  const size_t num_bits = num_elements * static_cast<size_t>(element_bits);
  *out_size = (num_bits + 7) / 8;
  return kLiteRtStatusOk;
}

// Alignment requested for host-visible buffers.  IREE's HAL allocator and the
// Coral NPU command buffer both prefer cache-line aligned imports.
constexpr size_t kHostBufferAlignment = 64;

}  // namespace coralnpu

#endif  // ODML_LITERT_LITERT_VENDORS_CORALNPU_COMMON_CORALNPU_COMMON_H_
