/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef CPU_AARCH64_CPU_ISA_TRAITS_HPP
#define CPU_AARCH64_CPU_ISA_TRAITS_HPP

#include <type_traits>

#include "dnnl_types.h"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR

#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64.h"
#include "cpu/aarch64/xbyak_aarch64/xbyak_aarch64_util.h"
#include "cpu/aarch64/xbyak_translator_aarch64/translator/include/xbyak_translator_aarch64/xbyak.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

enum cpu_isa_bit_t : unsigned {
    sse41_bit = 1u << 0,
    avx_bit = 1u << 1,
    avx2_bit = 1u << 2,
    avx512_common_bit = 1u << 3,
    avx512_mic_bit = 1u << 4,
    avx512_mic_4ops_bit = 1u << 5,
    avx512_core_bit = 1u << 6,
    avx512_core_vnni_bit = 1u << 7,
    avx512_core_bf16_bit = 1u << 8,
    amx_tile_bit = 1u << 9,
    amx_int8_bit = 1u << 10,
    amx_bf16_bit = 1u << 11,
    avx_vnni_bit = 1u << 12,

    asimd_bit = 1u << 13,
    sve_128_bit = 1u << 14,
    sve_256_bit = 1u << 15,
    sve_384_bit = 1u << 16,
    sve_512_bit = 1u << 17,
};

enum cpu_isa_t : unsigned {
    isa_any = 0u,
    sse41 = sse41_bit,
    avx = avx_bit | sse41,
    avx2 = avx2_bit | avx,
    avx_vnni = avx_vnni_bit | avx_bit,
    avx2_vnni = avx_vnni | avx2,
    avx512_common = avx512_common_bit | avx2,
    avx512_mic = avx512_mic_bit | avx512_common,
    avx512_mic_4ops = avx512_mic_4ops_bit | avx512_mic,
    avx512_core = avx512_core_bit | avx512_common,
    avx512_core_vnni = avx512_core_vnni_bit | avx512_core,
    avx512_core_bf16 = avx512_core_bf16_bit | avx512_core_vnni,
    amx_tile = amx_tile_bit,
    amx_int8 = amx_int8_bit | amx_tile,
    amx_bf16 = amx_bf16_bit | amx_tile,
    avx512_core_bf16_amx_int8 = avx512_core_bf16 | amx_int8,
    avx512_core_bf16_amx_bf16 = avx512_core_bf16 | amx_bf16,
    avx512_core_amx = avx512_core_bf16 | amx_int8 | amx_bf16,

    asimd = asimd_bit,
    sve_128 = sve_128_bit | asimd,
    sve_256 = sve_256_bit | asimd,
    sve_384 = sve_384_bit | asimd,
    sve_512 = sve_512_bit | asimd,

    // NOTE: Intel AMX is under initial support and turned off by default
    isa_all = ~0u & ~amx_tile_bit & ~amx_int8_bit & ~amx_bf16_bit,
};

enum class cpu_isa_cmp_t {
    // List of infix comparison relations between two cpu_isa_t
    // where we take isa_1 and isa_2 to be two cpu_isa_t instances.

    // isa_1 SUBSET isa_2 if all feature flags supported by isa_1
    // are supported by isa_2 as well (equality allowed)
    SUBSET,

    // isa_1 SUPERSET isa_2 if all feature flags supported by isa_2
    // are supported by isa_1 as well (equality allowed)
    SUPERSET,

    // Few more options that (depending upon need) can be enabled in future

    // 1. PROPER_SUBSET: isa_1 SUBSET isa_2 and isa_1 != isa_2
    // 2. PROPER_SUPERSET: isa_1 SUPERSET isa_2 and isa_1 != isa_2
};

const char *get_isa_info();

static inline bool compare_isa(
        cpu_isa_t isa_1, cpu_isa_cmp_t cmp, cpu_isa_t isa_2) {
    unsigned mask_1 = static_cast<unsigned>(isa_1);
    unsigned mask_2 = static_cast<unsigned>(isa_2);
    unsigned mask_min = mask_1 & mask_2;

    switch (cmp) {
        case cpu_isa_cmp_t::SUBSET: return mask_1 == mask_min;
        case cpu_isa_cmp_t::SUPERSET: return mask_2 == mask_min;
        default: assert(!"unsupported comparison of isa"); return false;
    }
}

static inline bool is_subset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUBSET, isa_2);
}

static inline bool is_superset(cpu_isa_t isa_1, cpu_isa_t isa_2) {
    return compare_isa(isa_1, cpu_isa_cmp_t::SUPERSET, isa_2);
}

cpu_isa_t DNNL_API get_max_cpu_isa_mask(bool soft = false);
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
dnnl_cpu_isa_t get_effective_cpu_isa();

template <cpu_isa_t>
struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <>
struct cpu_isa_traits<isa_all> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_all;
    static constexpr const char *user_option_env = "ALL";
};

template <>
struct cpu_isa_traits<sse41> {
    typedef Xbyak::Xmm Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_sse41;
    static constexpr const char *user_option_env = "SSE41";
};

template <>
struct cpu_isa_traits<avx> {
    typedef Xbyak::Ymm Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx;
    static constexpr const char *user_option_env = "AVX";
};

template <>
struct cpu_isa_traits<avx2> : public cpu_isa_traits<avx> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2;
    static constexpr const char *user_option_env = "AVX2";
};

template <>
struct cpu_isa_traits<avx2_vnni> : public cpu_isa_traits<avx2> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx2_vnni;
    static constexpr const char *user_option_env = "AVX2_VNNI";
};

template <>
struct cpu_isa_traits<avx512_common> {
    typedef Xbyak::Zmm Vmm;
    typedef Xbyak_aarch64::ZReg TReg;
    typedef Xbyak_aarch64::ZRegB TRegB;
    typedef Xbyak_aarch64::ZRegH TRegH;
    typedef Xbyak_aarch64::ZRegS TRegS;
    typedef Xbyak_aarch64::ZRegD TRegD;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_512);
    static constexpr const char *user_option_env = "SVE_512";
};

template <>
struct cpu_isa_traits<avx512_core> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_core;
    static constexpr const char *user_option_env = "AVX512_CORE";
};

template <>
struct cpu_isa_traits<avx512_mic> : public cpu_isa_traits<avx512_common> {
    static constexpr dnnl_cpu_isa_t user_option_val = dnnl_cpu_isa_avx512_mic;
    static constexpr const char *user_option_env = "AVX512_MIC";
};

template <>
struct cpu_isa_traits<avx512_mic_4ops> : public cpu_isa_traits<avx512_mic> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_mic_4ops;
    static constexpr const char *user_option_env = "AVX512_MIC_4OPS";
};

template <>
struct cpu_isa_traits<avx512_core_vnni> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_vnni;
    static constexpr const char *user_option_env = "AVX512_CORE_VNNI";
};

template <>
struct cpu_isa_traits<avx512_core_bf16> : public cpu_isa_traits<avx512_core> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_bf16;
    static constexpr const char *user_option_env = "AVX512_CORE_BF16";
};

template <>
struct cpu_isa_traits<avx512_core_amx> {
    static constexpr dnnl_cpu_isa_t user_option_val
            = dnnl_cpu_isa_avx512_core_amx;
    static constexpr const char *user_option_env = "AVX512_CORE_AMX";
};

template <>
struct cpu_isa_traits<asimd> {
    typedef Xbyak_aarch64::VReg TReg;
    typedef Xbyak_aarch64::VReg16B TRegB;
    typedef Xbyak_aarch64::VReg8H TRegH;
    typedef Xbyak_aarch64::VReg4S TRegS;
    typedef Xbyak_aarch64::VReg2D TRegD;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_asimd);
    static constexpr const char *user_option_env = "ADVANCED_SIMD";
};

template <>
struct cpu_isa_traits<sve_128> {
    typedef Xbyak_aarch64::VReg TReg;
    typedef Xbyak_aarch64::VReg16B TRegB;
    typedef Xbyak_aarch64::VReg8H TRegH;
    typedef Xbyak_aarch64::VReg4S TRegS;
    typedef Xbyak_aarch64::VReg2D TRegD;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_128);
    static constexpr const char *user_option_env = "SVE_128";
};

template <>
struct cpu_isa_traits<sve_256> {
    typedef Xbyak_aarch64::VReg TReg;
    typedef Xbyak_aarch64::VReg16B TRegB;
    typedef Xbyak_aarch64::VReg8H TRegH;
    typedef Xbyak_aarch64::VReg4S TRegS;
    typedef Xbyak_aarch64::VReg2D TRegD;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_256);
    static constexpr const char *user_option_env = "SVE_256";
};

template <>
struct cpu_isa_traits<sve_512> {
    typedef Xbyak_aarch64::ZReg TReg;
    typedef Xbyak_aarch64::ZRegB TRegB;
    typedef Xbyak_aarch64::ZRegH TRegH;
    typedef Xbyak_aarch64::ZRegS TRegS;
    typedef Xbyak_aarch64::ZRegD TRegD;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
    static constexpr dnnl_cpu_isa_t user_option_val
            = static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_512);
    static constexpr const char *user_option_env = "SVE_512";
};

inline const Xbyak_aarch64::util::Cpu &cpu() {
    const static Xbyak_aarch64::util::Cpu cpu_;
    return cpu_;
}

namespace {

static inline bool mayiuse(const cpu_isa_t cpu_isa, bool soft = false) {
    using namespace Xbyak_aarch64::util;

    unsigned cpu_isa_mask = aarch64::get_max_cpu_isa_mask(soft);
    if ((cpu_isa_mask & cpu_isa) != cpu_isa) return false;

    switch (cpu_isa) {
        case asimd:
            /* Advanced SIMD and floating-point instructions are
         mondatory for AArch64. */
            return true;
        case sve_128:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_128;
        case sve_256:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_256;
        case sve_384:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_384;
        case sve_512:
            return cpu().has(Cpu::tSVE) && cpu().getSveLen() == SVE_512;
        case sse41:
        case avx2:
        case avx512_common: return true;
        case isa_any: return true;
        case isa_all: return false;
    }
    return false;
}

static inline bool mayiuse_atomic() {
    using namespace Xbyak_aarch64::util;
    return cpu().isAtomicSupported();
}

inline bool isa_has_bf16(cpu_isa_t isa) {
    return false;
}

} // namespace

/* whatever is required to generate string literals... */
#include "common/z_magic.hpp"
/* clang-format off */
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    ((isa) == isa_any ? prefix STRINGIFY(any) : \
    ((isa) == asimd ? prefix STRINGIFY(asimd) : \
    ((isa) == sve_512 ? prefix STRINGIFY(sve_512) : \
    prefix suffix_if_any)))
/* clang-format on */

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
