/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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
#include "cpu/aarch64/jit_uni_i8i8_pooling.hpp"
#include <math.h>

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

static inline dim_t get_offset(
        const memory_desc_wrapper &mdw, int n, int c, int d, int h, int w) {
    switch (mdw.ndims()) {
        case 3: return mdw.blk_off(n, c, w);
        case 4: return mdw.blk_off(n, c, h, w);
        case 5: return mdw.blk_off(n, c, d, h, w);
        default: assert(!"Invalid tensor dimension in pooling");
    }
    return 0;
}

using namespace Xbyak_aarch64;

using namespace dnnl::impl::utils;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;
using namespace alg_kind;

#define GET_OFF(field) offsetof(call_params_t, field)

struct call_params_t {
    const char *src_i8;
    const char *dst_i8;
    const void *post_ops_binary_rhs_arg_vec;
    size_t kd_range;
    size_t kh_range;
    size_t kw_range;
    float idivider;
    const char *src_safe_access;
    const char *dst_safe_access;
};

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_pooling_fwd_ker_t)

    using TReg = typename cpu_isa_traits<isa>::TReg;

    VReg xreg(int idx) const { return VReg(idx); }
    ZReg yreg(int idx) const { return ZReg(xreg(idx).getIdx()); }
    TReg vreg(int idx) const { return TReg(xreg(idx).getIdx()); }
    // In case of avx2 with data type i8 we need to use
    // maskmovdqu and maskmovq instructions which has its destination hardcoded in rdi.
    // Windows ABI: abi_param1 is rcx - nothing to do else
    // Unix ABI: abi_param1 is rdi - copy it to rcx and use it as abi_param1
    XReg reg_param = XReg(3); // Our "unified abi_param1"
    XReg reg_ptr_src_i8 = XReg(4);
    XReg reg_ptr_dst_i8 = XReg(5);
    XReg reg_ptr_maskmovdqu_dst = XReg(0); // store destination - must be rdi

    XReg reg_kd_index = XReg(
            0); // shared with reg_ptr_maskmovdqu_dst; only used before store
    XReg reg_kh_index = XReg(11);
    XReg reg_kw_index = XReg(10);
    XReg reg_kd = XReg(14);
    XReg reg_kh = XReg(13);
    XReg reg_kw = XReg(12);
    XReg c_iter = XReg(15); // shared with reg_mask; only used after mask init

    XReg aux_reg_src_d = XReg(
            2); // shared with reg_tmp; loaded before each accum loop, unused during store
    XReg aux_reg_src_h = XReg(7);
    XReg aux_reg_src_w = XReg(1);

    XReg reg_tmp = XReg(2); // only used during mask init and store
    XReg reg_src_safe_access = XReg(9);
    XReg reg_dst_safe_access = XReg(1);

    XReg reg_mask = XReg(15); // only used during mask init

    XReg X_TRANSLATOR_STACK = XReg(22);
    XReg x_tmp_addr = XReg(28);
    XReg x_tmp_0 = XReg(23);

    PReg k_cmp_mask = PReg(7);

    PReg mask(int idx) { return PReg(6 - idx); } /* 6, 5, 4, 3 */

    PReg p_256 = PReg(1);
    PReg p_512 = PReg(2);
    PReg p_tmp0 = PReg(8);
    PReg p_128 = PReg(0);
    PReg p_lsb = PReg(2);
    PReg p_tmp1 = PReg(11);
    PReg p_tmp2 = PReg(12);
    PReg P_MSB_256 = PReg(13);
    PReg P_MSB_384 = PReg(14);
    PReg P_ALL_ONE = PReg(15);

    // ref to any of XYZ-regs via xreg/yreg/vreg functions
    VReg xmm_tmp = xreg(0); // temp to init vreg_tmp
    TReg vreg_tmp = vreg(0); // max pooling : holds minimum values for data_type
    TReg vreg_zeros = vreg(1);
    TReg vreg_tail = vreg(4);

    // only in case of <isa> == avx2
    TReg vreg_mask = vreg(2); // full byte-mask
    VReg xreg_mask_lo = xreg(
            2); // low 128-bits part of byte-mask (alias for xmm part of vreg_mask)
    VReg xreg_mask_hi = xreg(
            3); // "max" - high 128-bits part of byte-mask (stored separately)

    // vreg_mask shifted left (aligned left) to be used in tail processing.
    // Example:       idx [31..0]
    //          vreg_mask = [0,0,0,0,0,.....,0,x,x,x,x,x] ; x => byte mask (msb set)
    //          vreg_mask_2 = [x,x,x,x,x,0,0,0,0,0,.....,0]
    TReg vreg_mask_2 = vreg(5);
    VReg xreg_mask_2_lo = xreg(5); // similar to xreg_mask_lo
    VReg xreg_mask_2_hi = xreg(6); // similar to xreg_mask_hi

    TReg vreg_mask_q = vreg(3); // "avg" - 1/4 part for non-zero tails

    ZReg z_tmp0 = ZReg(24);
    ZReg z_tmp1 = ZReg(25);
    ZReg z_tmp2 = ZReg(26);
    ZReg z_tmp3 = ZReg(27);

    int post_op_tail_opmask_idx_ = -1;
    jit_pool_conf_t jpp;
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;

    enum : int { max_vidx_base = utils::one_of(isa, asimd, sve_256) ? 7 : 2 };
    //"avg" pool uses more registers for unrolling.
    enum : int { avg_vidx_base = utils::one_of(isa, asimd, sve_256) ? 4 : 2 };

    TReg max_base_vr(int idx) const { return vreg(max_vidx_base + idx); }
    TReg avg_base_vr(int idx) const { return vreg(avg_vidx_base + idx); }

    size_t sizeof_src_dt() const { return data_type_size(jpp.src_dt); }
    size_t sizeof_dst_dt() const { return data_type_size(jpp.dst_dt); }

    /* max pooling */
    TReg vreg_src(int idx) const {
        return max_base_vr(idx);
    } // [0    .. ur_c-1]
    TReg vreg_dst(int idx) const {
        return max_base_vr(jpp.ur_c + idx);
    } // [ur_c .. 2*ur_c-1]

    /* avg pooling */
    // s32 used for processing of s8/u8 data
    // thus we need to take into account ratio of sizes s32/i8 = 4
    static constexpr data_type_t avg_proc_dt = data_type::s32;
    enum : int {
        s32_to_i8_ratio = sizeof(typename prec_traits<avg_proc_dt>::type)
                / sizeof(typename prec_traits<data_type::u8>::type),
        max_num_ll = s32_to_i8_ratio,
        mmx_msk_base_reg = 3
    };

    TReg vreg_src_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 0 * max_num_ll);
    } // ll: 0..4 [0..3]

    TReg vreg_dst_s32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 1 * max_num_ll);
    } // ll: 0..4 [4..7]

    TReg vreg_dst_f32(int jj, int ll) {
        return avg_base_vr(3 * max_num_ll * jj + ll + 2 * max_num_ll);
    } // ll: 0..4 [8..11]

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    void init_tmp_reg();
    void init_mask();

    void load_vreg_mask_q(int ll) {};

    void load_src_max_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src_avg_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src(int jj, int ll, int c_tail);

    void store_dst_max_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst_avg_op(
            int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_op(const int jj);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate() override;

    static status_t init_conf(jit_pool_conf_t &jpp, const pooling_pd_t *ppd);

    jit_uni_i8i8_pooling_fwd_ker_t(
            const jit_pool_conf_t &jpp_, const memory_desc_t *dst_md)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
        , jpp(jpp_)
        , postops_injector_(nullptr) {

        if (jpp.with_postops) {

            const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
            const std::size_t c_tail_elems = jpp.c % simd_w;
            post_op_tail_opmask_idx_ = 0;
            if (c_tail_elems) {
                for (int ll = max_num_ll - 1; ll >= 0; ll--) {
                    if (jpp.tail[ll] != 0) {
                        post_op_tail_opmask_idx_ = ll;
                        break;
                    }
                }
            };

            static constexpr bool use_per_oc_spatial_strategy = false;
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            static constexpr std::size_t tmp_vmm_injector = 0u;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, XReg(7), XReg(14), preserve_gpr,
                    preserve_vmm, GET_OFF(post_ops_binary_rhs_arg_vec),
                    memory_desc_wrapper(*dst_md), c_tail_elems,
                    mask(post_op_tail_opmask_idx_),
                    use_exact_tail_scalar_bcast};
            const binary_injector::static_params_t bsp {
                    reg_param, use_per_oc_spatial_strategy, rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<isa>>(
                    this, jpp.post_ops, bsp);
        }
    }
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<asimd>::load_src_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_256>::load_src_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::load_src_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32) {
            add_imm(x_tmp_addr, aux_reg_src_w, offset, x_tmp_0);
            pfalse(p9.b);
            zip1(p1.b, mask(0).b, p9.b);
            zip1(p1.h, p1.h, p9.h);
            ld1w(z_tmp0.s, p1 / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_tmp_addr));
            xa_->mov(ZRegS(IDX(vreg_src(jj))), p1 / T_m, z_tmp0.s);
        } else {
            add_imm(x_tmp_addr, aux_reg_src_w, offset, x_tmp_0);
            ld1b(z_tmp0.b, mask(0) / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_tmp_addr));
            xa_->mov(ZRegB(IDX(vreg_src(jj))), mask(0) / T_m, z_tmp0.b);
        }
    } else {
        add_imm(x_tmp_addr, aux_reg_src_w, offset, x_tmp_0);
        ldr(ZReg(IDX(vreg_src(jj))), Xbyak_aarch64::ptr(x_tmp_addr));
    }
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<asimd>::load_src_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_256>::load_src_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::load_src_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    const TReg &vr_src = vreg_src_s32(jj, ll);

    switch (jpp.src_dt) {
        case s32:
            add_imm(x_tmp_addr, aux_reg_src_w, offset * data_type_size(s32),
                    x_tmp_0);
            if (masked) {
                pfalse(p9.b);
                zip1(p1.b, mask(ll).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                ld1w(z_tmp0.s, p1 / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(x_tmp_addr));
                xa_->mov(ZRegS(IDX(vr_src)), p1 / T_m, z_tmp0.s);
            } else {
                ldr(ZReg(IDX(vr_src)), Xbyak_aarch64::ptr(x_tmp_addr));
            }
            break;
        case data_type::s8:
            add_imm(x_tmp_addr, aux_reg_src_w, offset, x_tmp_0);
            if (masked) {
                pfalse(p9.b);
                zip1(p1.b, mask(ll).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                // use p_tmp, uzp1 can be eliminate.
                ld1b(z_tmp0.s, p1 / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(x_tmp_addr));
                sxtb(ZReg(IDX(vr_src)).s, p1 / T_m, z_tmp0.s);
            } else {
                ld1b(z_tmp0.s, p_512 / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(x_tmp_addr));
                sxtb(ZReg(IDX(vr_src)).s, p_512 / T_m, z_tmp0.s);
            }
            break;
        case u8:
            add_imm(x_tmp_addr, aux_reg_src_w, offset, x_tmp_0);
            if (masked) {
                pfalse(p9.b);
                zip1(p1.b, mask(ll).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                // use p_tmp, uzp1 can be eliminate.
                ld1b(z_tmp0.s, p1 / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(x_tmp_addr));
                uxtb(ZReg(IDX(vr_src)).s, p1 / T_m, z_tmp0.s);
            } else {
                ldr(QReg(z_tmp0.getIdx()), Xbyak_aarch64::ptr(x_tmp_addr));
                zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                uxtb(ZReg(IDX(vr_src)).s, p_512 / T_m, z_tmp0.s);
            }
            break;
        default: assert(!"unsupported src data type");
    }
};

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::load_src(int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_max_op(jj, ll, offset, masked, jpp.tail[0]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<asimd>::store_dst_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_256>::store_dst_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::store_dst_max_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        switch (jpp.src_dt) {
            case s32:
                add_imm(x_tmp_addr, reg_ptr_dst_i8, offset, x_tmp_0);
                pfalse(p9.b);
                zip1(p1.b, mask(0).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                st1w(ZRegS(IDX(vreg_dst(jj))), p1,
                        Xbyak_aarch64::ptr(x_tmp_addr));
                break;
            case data_type::s8:
            case u8:
                add_imm(x_tmp_addr, reg_ptr_dst_i8, offset, x_tmp_0);
                st1b(ZRegB(IDX(vreg_dst(jj))), mask(0),
                        Xbyak_aarch64::ptr(x_tmp_addr));
                break;
            default: assert(!"unsupported src data type");
        }
    } else {
        add_imm(x_tmp_addr, reg_ptr_dst_i8, offset, x_tmp_0);
        str(ZReg(IDX(vreg_dst(jj))), Xbyak_aarch64::ptr(x_tmp_addr));
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<asimd>::store_dst_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_256>::store_dst_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::store_dst_avg_op(
        int jj, int ll, size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk) return;

    const TReg &vr_dst = vreg_dst_s32(jj, ll);
    switch (jpp.dst_dt) {
        case s32:
            add_imm(x_tmp_addr, reg_ptr_dst_i8, offset, x_tmp_0);
            if (masked) {
                pfalse(p9.b);
                zip1(p1.b, mask(ll).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                st1w(ZRegS(IDX(vr_dst)), p1, Xbyak_aarch64::ptr(x_tmp_addr));
            } else {
                str(ZReg(IDX(vr_dst)), Xbyak_aarch64::ptr(x_tmp_addr));
            }
            break;
        case data_type::s8:
            add_imm(x_tmp_addr, reg_ptr_dst_i8, offset, x_tmp_0);
            if (masked) {
                xa_->mov(z_tmp0.d, ZRegD(IDX(vr_dst)));
                smin(z_tmp0.s, 127);
                smax(z_tmp0.s, -128);
                pfalse(p9.b);
                zip1(p1.b, mask(ll).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                st1b(z_tmp0.s, p1, Xbyak_aarch64::ptr(x_tmp_addr));
            } else {
                xa_->mov(z_tmp0.d, ZRegD(IDX(vr_dst)));
                smin(z_tmp0.s, 127);
                smax(z_tmp0.s, -128);
                st1b(z_tmp0.s, p_512, Xbyak_aarch64::ptr(x_tmp_addr));
            }
            break;
        case u8:
            add_imm(x_tmp_addr, reg_ptr_dst_i8, offset, x_tmp_0);
            if (masked) {
                xa_->mov(z_tmp0.d, ZRegD(IDX(vr_dst)));
                umin(z_tmp0.s, 255);
                pfalse(p9.b);
                zip1(p1.b, mask(ll).b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                st1b(z_tmp0.s, p1, Xbyak_aarch64::ptr(x_tmp_addr));
            } else {
                xa_->mov(z_tmp0.d, ZRegD(IDX(vr_dst)));
                umin(z_tmp0.s, 255);
                st1b(z_tmp0.s, p_512, Xbyak_aarch64::ptr(x_tmp_addr));
            }
            break;
        default: assert(!"unsupported dst data_type");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::store_dst(
        int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj * c_block * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_max_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll * (c_block / max_num_ll) + jj * c_block)
                    * sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<asimd>::compute_max_op(const int jj) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_256>::compute_max_op(const int jj) {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::compute_max_op(const int jj) {
    using namespace data_type;

    // Compare
    switch (jpp.src_dt) {
        case s32:
            switch (_cmp_lt_os) {
                case 0:
                    cmpeq(k_cmp_mask.s, p_512 / Xbyak_aarch64::T_z,
                            ZRegS(IDX(vreg_dst(jj))), ZRegS(IDX(vreg_src(jj))));
                    break; //EQ
                case 1:
                    cmplt(k_cmp_mask.s, p_512 / Xbyak_aarch64::T_z,
                            ZRegS(IDX(vreg_dst(jj))), ZRegS(IDX(vreg_src(jj))));
                    break; //LT
                case 2:
                    cmple(k_cmp_mask.s, p_512 / Xbyak_aarch64::T_z,
                            ZRegS(IDX(vreg_dst(jj))), ZRegS(IDX(vreg_src(jj))));
                    break; //LE
                case 4:
                    cmpne(k_cmp_mask.s, p_512 / Xbyak_aarch64::T_z,
                            ZRegS(IDX(vreg_dst(jj))), ZRegS(IDX(vreg_src(jj))));
                    break; //NEQ
                case 5:
                    cmpge(k_cmp_mask.s, p_512 / Xbyak_aarch64::T_z,
                            ZRegS(IDX(vreg_dst(jj))), ZRegS(IDX(vreg_src(jj))));
                    break; //NLT
                case 6:
                    cmpgt(k_cmp_mask.s, p_512 / Xbyak_aarch64::T_z,
                            ZRegS(IDX(vreg_dst(jj))), ZRegS(IDX(vreg_src(jj))));
                    break; //NLE
                case 3:
                case 7:
                default: assert(!"unreachable"); break;
            }
            break;
        case data_type::s8:
            switch (_cmp_lt_os) {
                case 0:
                    cmpeq(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //EQ
                case 1:
                    cmplt(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //LT
                case 2:
                    cmple(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //LE
                case 4:
                    cmpne(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //NEQ
                case 5:
                    cmplt(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    xa_->not_(k_cmp_mask.b, p_512, k_cmp_mask.b);
                    break; //NLT
                case 6:
                    cmple(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    xa_->not_(k_cmp_mask.b, p_512, k_cmp_mask.b);
                    break; //NLE
                case 3:
                case 7:
                default: assert(!"unreachable"); break;
            }
            break;
        case u8:
            switch (_cmp_lt_os) {
                case 0:
                    cmpeq(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //EQ
                case 1:
                    cmpls(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //LT
                case 2:
                    cmplo(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //LE
                case 4:
                    cmpne(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    break; //NEQ
                case 5:
                    cmplo(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    xa_->not_(k_cmp_mask.b, p_512, k_cmp_mask.b);
                    break; //NLT
                case 6:
                    cmpls(k_cmp_mask.b, p_512 / Xbyak_aarch64::T_z,
                            ZRegB(IDX(vreg_dst(jj))), ZRegB(IDX(vreg_src(jj))));
                    xa_->not_(k_cmp_mask.b, p_512, k_cmp_mask.b);
                    break; //NLE
                case 3:
                case 7:
                default: assert(!"unreachable"); break;
            }
            break;
        default: assert(!"unsupported src data type");
    }

    // move max values into vreg_dst
    if (jpp.src_dt == s32) {
        sel(ZRegS(IDX(vreg_dst(jj))), k_cmp_mask / T_m,
                ZRegS(IDX(vreg_src(jj))), ZRegS(IDX(vreg_dst(jj))));
    } else {
        sel(ZRegB(IDX(vreg_dst(jj))), k_cmp_mask / T_m,
                ZRegB(IDX(vreg_src(jj))), ZRegB(IDX(vreg_dst(jj))));
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_max_step(
        int ur_c, int c_tail) {
    Label l_kd, l_kh, l_kw;

    int ih = jpp.ih;
    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++) {
        int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) {
            xa_->mov(ZRegD(IDX(vreg_dst(jj))), ZRegD(IDX(vreg_tmp)));
        } else if (vlen == 32) {
            xa_->mov(ZRegD(IDX(vreg_dst(jj))), ZRegD(IDX(vreg_tmp)));
            xa_->mov(ZRegS(IDX(vreg_dst(jj))), P_MSB_256 / T_m, 0);
        } else if (vlen == 16) {
            xa_->mov(VReg16B(IDX(vreg_dst(jj))), VReg16B(IDX(vreg_tmp)));
        } else {
            assert(!"unreachable");
        }
    }

    xa_->mov(aux_reg_src_d, reg_ptr_src_i8);
    eor(reg_kd_index, reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        xa_->mov(aux_reg_src_h, aux_reg_src_d);
        eor(reg_kh_index, reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            xa_->mov(aux_reg_src_w, aux_reg_src_h);
            eor(reg_kw_index, reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    load_src(jj, 0, c_tail);
                    compute_max_op(jj);
                }
                xa_->add(aux_reg_src_w, aux_reg_src_w, c * sizeof_src_dt());
                adds(reg_kw_index, reg_kw_index, 1);
                xa_->cmp(reg_kw_index, reg_kw);
                b(LT, l_kw);
            }
            add_imm(aux_reg_src_h, aux_reg_src_h, iw * c * sizeof_src_dt(),
                    x_tmp_0);
            adds(reg_kh_index, reg_kh_index, 1);
            xa_->cmp(reg_kh_index, reg_kh);
            b(LT, l_kh);
        }
        add_imm(aux_reg_src_d, aux_reg_src_d, ih * iw * c * sizeof_src_dt(),
                x_tmp_0);
        adds(reg_kd_index, reg_kd_index, 1);
        xa_->cmp(reg_kd_index, reg_kd);
        b(LT, l_kd);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_avg_step(
        int ur_c, int c_tail) {
    using namespace data_type;

    Label l_kd, l_kh, l_kw;

    int ih = jpp.ih;
    int iw = jpp.iw;
    int c = jpp.c;

    const int num_ll = data_type_size(avg_proc_dt) / data_type_size(jpp.src_dt);

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            bool masked = jj == ur_c - 1 && c_tail;
            size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                // Clearing of src reg is not needed as they are written before read
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    eor(ZReg(IDX(vreg_dst_s32(jj, ll))).d,
                            ZReg(IDX(vreg_dst_s32(jj, ll))).d,
                            ZReg(IDX(vreg_dst_s32(jj, ll))).d);
                } else if (vlen == 32) {
                    eor(ZRegD(IDX(vreg_dst_s32(jj, ll))),
                            ZRegD(IDX(vreg_dst_s32(jj, ll))),
                            ZRegD(IDX(vreg_dst_s32(jj, ll))));
                    xa_->mov(ZRegS(IDX(vreg_dst_s32(jj, ll))), P_MSB_256 / T_m,
                            0);
                } else if (vlen == 16) {
                    eor(VReg16B(IDX(vreg_dst_s32(jj, ll))),
                            VReg16B(IDX(vreg_dst_s32(jj, ll))),
                            VReg16B(IDX(vreg_dst_s32(jj, ll))));
                } else {
                    assert(!"unreachable");
                }
            }
        }
    }

    xa_->mov(aux_reg_src_d, reg_ptr_src_i8);
    eor(reg_kd_index, reg_kd_index, reg_kd_index);
    L(l_kd);
    {
        xa_->mov(aux_reg_src_h, aux_reg_src_d);
        eor(reg_kh_index, reg_kh_index, reg_kh_index);
        L(l_kh);
        {
            xa_->mov(aux_reg_src_w, aux_reg_src_h);
            eor(reg_kw_index, reg_kw_index, reg_kw_index);
            L(l_kw);
            {
                for (int jj = 0; jj < ur_c; jj++) {
                    for (int ll = 0; ll < num_ll; ll++) {
                        bool masked = jj == ur_c - 1 && c_tail;
                        size_t msk = jpp.tail[ll];
                        if (!(masked && !msk)) {
                            load_src(jj, ll, c_tail);
                            int vlen = cpu_isa_traits<isa>::vlen;
                            if (vlen == 64) {
                                xa_->add(ZReg(IDX(vreg_dst_s32(jj, ll))).s,
                                        ZReg(IDX(vreg_dst_s32(jj, ll))).s,
                                        ZReg(IDX(vreg_src_s32(jj, ll))).s);
                            } else if (vlen == 32) {
                                xa_->add(ZReg(IDX(vreg_dst_s32(jj, ll))).s,
                                        ZReg(IDX(vreg_dst_s32(jj, ll))).s,
                                        ZReg(IDX(vreg_src_s32(jj, ll))).s);
                            } else if (vlen == 16) {
                                xa_->add(VReg(IDX(vreg_dst_s32(jj, ll))).s4,
                                        VReg(IDX(vreg_dst_s32(jj, ll))).s4,
                                        VReg(IDX(vreg_src_s32(jj, ll))).s4);
                                xa_->mov(ZReg(IDX(vreg_dst_s32(jj, ll))).s,
                                        P_MSB_256 / T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    }
                }
                xa_->add(aux_reg_src_w, aux_reg_src_w, c * sizeof_src_dt());
                adds(reg_kw_index, reg_kw_index, 1);
                xa_->cmp(reg_kw_index, reg_kw);
                b(LT, l_kw);
            }
            add_imm(aux_reg_src_h, aux_reg_src_h, iw * c * sizeof_src_dt(),
                    x_tmp_0);
            adds(reg_kh_index, reg_kh_index, 1);
            xa_->cmp(reg_kh_index, reg_kh);
            b(LT, l_kh);
        }
        add_imm(aux_reg_src_d, aux_reg_src_d, ih * iw * c * sizeof_src_dt(),
                x_tmp_0);
        adds(reg_kd_index, reg_kd_index, 1);
        xa_->cmp(reg_kd_index, reg_kd);
        b(LT, l_kd);
    }

    static constexpr int vlen_size_elem
            = cpu_isa_traits<isa>::vlen / sizeof(float);
    const auto reg_tmp_postops = XReg(15);
    const injector_utils::register_preserve_guard_t reg_guard(this,
            jpp.with_binary ? std::initializer_list<XReg> {reg_tmp_postops}
                            : std::initializer_list<XReg> {},
            {});
    if (jpp.with_binary) {
        xa_->mov_imm(x_tmp_0,
                static_cast<int64_t>(
                        static_cast<int8_t>(ur_c * num_ll * vlen_size_elem)));
        xa_->mul(reg_tmp_postops, c_iter, x_tmp_0);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            const bool masked = jj == ur_c - 1 && c_tail;
            const size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                const auto &reg_dst_f32 = vreg_dst_f32(jj, ll);
                const auto &reg_dst_s32 = vreg_dst_s32(jj, ll);
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    scvtf(ZReg(IDX(reg_dst_f32)).s, p_512 / T_m,
                            ZReg(IDX(reg_dst_s32)).s);
                    fmad(ZRegS(IDX(reg_dst_f32)), p_512 / T_m,
                            ZRegS(IDX(vreg_tmp)), ZRegS(IDX(vreg_zeros)));
                } else if (vlen == 32) {
                    scvtf(ZReg(IDX(reg_dst_f32)).s, p_512 / T_m,
                            ZReg(IDX(reg_dst_s32)).s);
                    fmad(ZRegS(IDX(reg_dst_f32)), p_512 / T_m,
                            ZRegS(IDX(vreg_tmp)), ZRegS(IDX(vreg_zeros)));
                    xa_->mov(ZReg(IDX(reg_dst_f32)).s, P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    scvtf(VReg(IDX(reg_dst_f32)).s4, VReg(IDX(reg_dst_s32)).s4);
                    fmad(ZRegS(IDX(reg_dst_f32)), p_512 / T_m,
                            ZRegS(IDX(vreg_tmp)), ZRegS(IDX(vreg_zeros)));
                    xa_->mov(ZReg(IDX(reg_dst_f32)).s, P_MSB_384 / T_m, 0);
                } else {
                    assert(!"unreachable");
                }

                if (jpp.with_postops) {
                    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
                    if (jpp.with_binary) {
                        rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                                reg_dst_f32.getIdx(), reg_tmp_postops);
                        rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                                reg_dst_f32.getIdx(),
                                ll * vlen_size_elem + jj * vlen_size_elem);
                        rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                                reg_dst_f32.getIdx(), reg_tmp_postops);
                        rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                                reg_dst_f32.getIdx(),
                                ll * vlen_size_elem + jj * vlen_size_elem);
                        const bool tail = ll == post_op_tail_opmask_idx_;
                        if (tail && masked)
                            rhs_arg_params.vmm_tail_idx_.emplace(
                                    reg_dst_f32.getIdx());
                    }
                    postops_injector_->compute_vector(
                            reg_dst_f32.getIdx(), rhs_arg_params);
                }
                if (vlen == 64) {
                    frinti(ZRegS(IDX(reg_dst_s32)), p_512 / T_m,
                            ZRegS(IDX(reg_dst_f32)));
                    fcvtzs(ZRegS(IDX(reg_dst_s32)), p_512 / T_m,
                            ZRegS(IDX(reg_dst_s32)));
                } else if (vlen == 32) {
                    frinti(ZRegS(IDX(reg_dst_s32)), p_512 / T_m,
                            ZRegS(IDX(reg_dst_f32)));
                    fcvtzs(ZRegS(IDX(reg_dst_s32)), p_512 / T_m,
                            ZRegS(IDX(reg_dst_s32)));
                    xa_->mov(ZReg(IDX(reg_dst_f32)).s, P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    frinti(VReg4S(IDX(reg_dst_s32)), VReg4S(IDX(reg_dst_f32)));
                    fcvtzs(VReg4S(IDX(reg_dst_s32)), VReg4S(IDX(reg_dst_s32)));
                } else {
                    assert(!"unreachable");
                }

                if (jpp.with_postops)
                    if (jpp.dst_dt == u8) {
                        int vlen = cpu_isa_traits<isa>::vlen;
                        if (vlen == 64) {
                            cmple(p_tmp0.s, p_512 / Xbyak_aarch64::T_z,
                                    ZReg(IDX(reg_dst_s32)).s,
                                    ZReg(IDX(vreg_zeros)).s);
                            cmpgt(p_tmp1.s, p_512 / Xbyak_aarch64::T_z,
                                    ZReg(IDX(reg_dst_s32)).s,
                                    ZReg(IDX(vreg_zeros)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, p_tmp0 / T_m,
                                    ZReg(IDX(vreg_zeros)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, p_tmp1 / T_m,
                                    ZReg(IDX(reg_dst_s32)).s);
                        } else if (vlen == 32) {
                            cmple(p_tmp0.s, p_512 / Xbyak_aarch64::T_z,
                                    ZReg(IDX(reg_dst_s32)).s,
                                    ZReg(IDX(vreg_zeros)).s);
                            cmpgt(p_tmp1.s, p_512 / Xbyak_aarch64::T_z,
                                    ZReg(IDX(reg_dst_s32)).s,
                                    ZReg(IDX(vreg_zeros)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, p_tmp0 / T_m,
                                    ZReg(IDX(vreg_zeros)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, p_tmp1 / T_m,
                                    ZReg(IDX(reg_dst_s32)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, P_MSB_256 / T_m,
                                    0);
                        } else if (vlen == 16) {
                            cmple(p_tmp0.s, p_512 / Xbyak_aarch64::T_z,
                                    ZReg(IDX(reg_dst_s32)).s,
                                    ZReg(IDX(vreg_zeros)).s);
                            cmpgt(p_tmp1.s, p_512 / Xbyak_aarch64::T_z,
                                    ZReg(IDX(reg_dst_s32)).s,
                                    ZReg(IDX(vreg_zeros)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, p_tmp0 / T_m,
                                    ZReg(IDX(vreg_zeros)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, p_tmp1 / T_m,
                                    ZReg(IDX(reg_dst_s32)).s);
                            xa_->mov(ZReg(IDX(reg_dst_s32)).s, P_MSB_384 / T_m,
                                    0);
                        } else {
                            assert(!"unreachable");
                        }
                    }
                store_dst(jj, ll, c_tail);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max: compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_c_block() {
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    eor(c_iter, c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop);
        {
            compute_step(ur_c, 0);
            xa_->add(reg_ptr_src_i8, reg_ptr_src_i8,
                    ur_c * c_block * sizeof_src_dt());
            xa_->add(reg_ptr_dst_i8, reg_ptr_dst_i8,
                    ur_c * c_block * sizeof_dst_dt());
            adds(c_iter, c_iter, 1);
            xa_->mov_imm(x_tmp_0, c_steps);
            xa_->cmp(c_iter, x_tmp_0);
            b(LT, l_main_loop);
        }
    }

    if (ur_c_tail != 0) { compute_step(ur_c_tail, c_tail); }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<asimd>::init_mask() {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_256>::init_mask() {
    assert(false /*function should not be used*/);
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<sve_512>::init_mask() {
    using namespace data_type;

    xa_->sub(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8 * max_num_ll);

    for (int ll = 0; ll < max_num_ll; ll++) {
        xa_->mov_imm(reg_mask, jpp.tail[ll]);
        str(reg_mask, Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 8 * ll));
    }
    for (int ll = 0; ll < max_num_ll; ll++) {
        ldr(PReg(mask(ll)), Xbyak_aarch64::ptr(X_TRANSLATOR_STACK));
        xa_->add(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_tmp_reg() {
    using namespace data_type;

    int vlen = cpu_isa_traits<isa>::vlen;
    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, idivider),
                    x_tmp_0);
            ldr(reg_tmp, Xbyak_aarch64::ptr(x_tmp_addr));
            bic(VReg(IDX(xmm_tmp)).b16, VReg(IDX(xmm_tmp)).b16,
                    VReg(IDX(xmm_tmp)).b16);
            xa_->mov(VReg(IDX(xmm_tmp)).d[0], reg_tmp);
            if (vlen == 64) {
                dup(ZRegS(IDX(vreg_tmp)), ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                dup(ZRegS(IDX(vreg_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                xa_->mov(ZRegS(IDX(vreg_tmp)), P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                dup(VReg4S(IDX(vreg_tmp)), VReg4S(IDX(xmm_tmp))[0]);
                xa_->mov(ZRegS(IDX(vreg_tmp)), P_MSB_384 / T_m, 0);
            } else {
                assert(!"unreachable");
            }
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    xa_->mov_imm(
                            reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case data_type::s8:
                    xa_->mov_imm(
                            reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    xa_->mov(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            bic(VReg(IDX(xmm_tmp)).b16, VReg(IDX(xmm_tmp)).b16,
                    VReg(IDX(xmm_tmp)).b16);
            xa_->mov(VReg(IDX(xmm_tmp)).d[0], reg_tmp);
            if (jpp.src_dt == s32) {
                if (vlen == 64) {
                    dup(ZRegS(IDX(vreg_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                } else if (vlen == 32) {
                    dup(ZRegS(IDX(vreg_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                    xa_->mov(ZRegS(IDX(vreg_tmp)), P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    dup(VReg4S(IDX(vreg_tmp)), VReg4S(IDX(xmm_tmp))[0]);
                    xa_->mov(ZRegS(IDX(vreg_tmp)), P_MSB_384 / T_m, 0);
                } else {
                    assert(!"unreachable");
                }
            } else if (mayiuse(sve_512)) {
                if (vlen == 64) {
                    dup(ZRegB(IDX(vreg_tmp)), ZRegB(IDX(xmm_tmp))[0]);
                } else if (vlen == 32) {
                    dup(ZRegB(IDX(vreg_tmp)), ZRegB(IDX(xmm_tmp))[0]);
                    xa_->mov(ZRegS(IDX(vreg_tmp)), P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    dup(VReg16B(IDX(vreg_tmp)), VReg16B(IDX(xmm_tmp))[0]);
                    xa_->mov(ZRegS(IDX(vreg_tmp)), P_MSB_384 / T_m, 0);
                } else {
                    assert(!"unreachable");
                }

            } else {
                assert(!"unreachable");
            }
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::generate() {
    preamble();

    ptrue(p_512.b);
    ptrue(p_256.b, VL32);
    ptrue(p_128.b, VL16);
    if (cpu_isa_traits<isa>::vlen == 32) {
        p_lsb = p_256;
    } else if (cpu_isa_traits<isa>::vlen == 16) {
        p_lsb = p_128;
    }

#if !defined(_WIN32)
    // Always use rcx as abi_param1 -
    // see the note about maskmovdqu/maskmovq near reg_param.
    xa_->mov(XReg(3), XReg(0));
#endif
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, src_i8), x_tmp_0);
    ldr(reg_ptr_src_i8, Xbyak_aarch64::ptr(x_tmp_addr));
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, dst_i8), x_tmp_0);
    ldr(reg_ptr_dst_i8, Xbyak_aarch64::ptr(x_tmp_addr));
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, kd_range), x_tmp_0);
    ldr(reg_kd, Xbyak_aarch64::ptr(x_tmp_addr));
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, kh_range), x_tmp_0);
    ldr(reg_kh, Xbyak_aarch64::ptr(x_tmp_addr));
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, kw_range), x_tmp_0);
    ldr(reg_kw, Xbyak_aarch64::ptr(x_tmp_addr));
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, src_safe_access),
            x_tmp_0);
    ldr(reg_src_safe_access, Xbyak_aarch64::ptr(x_tmp_addr));
    add_imm(x_tmp_addr, reg_param, offsetof(call_params_t, dst_safe_access),
            x_tmp_0);
    ldr(reg_dst_safe_access, Xbyak_aarch64::ptr(x_tmp_addr));

    eor(VReg16B(IDX(vreg_zeros)), VReg16B(IDX(vreg_zeros)),
            VReg16B(IDX(vreg_zeros)));

    init_mask();

    init_tmp_reg();

    compute_c_block();

    postamble();

    if (jpp.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(
        jit_pool_conf_t &jpp, const pooling_pd_t *ppd) {
    if (!mayiuse(isa)) return status::unimplemented;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];

    jpp.id = is_3d ? src_d.dims()[ndims - 3] : 1;
    jpp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];

    jpp.od = is_3d ? dst_d.dims()[ndims - 3] : 1;
    jpp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jpp.ow = dst_d.dims()[ndims - 1];

    jpp.stride_d = is_3d ? pd.strides[ndims - 5] : 1;
    jpp.stride_h = is_1d ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];

    jpp.kd = is_3d ? pd.kernel[ndims - 5] : 1;
    jpp.kh = is_1d ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = is_3d ? pd.padding[0][ndims - 5] : 0;
    jpp.t_pad = is_1d ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    // data_type items per one vreg on the <isa>
    //     isa == sve_512 : 64 bytes -> 64 for s8/u8, 16 for s32
    int simd_w = cpu_isa_traits<isa>::vlen / data_type_size(jpp.src_dt);

    /* Verify that vlen-sized memory access happens within the tensor's
     * size, otherwise load/store will always spill outside the memory
     * boundary.*/
    bool safe_load_n_store = IMPLICATION(utils::one_of(isa, sve_512),
            jpp.mb * jpp.c * nstl::min(jpp.id, jpp.od)
                            * nstl::min(jpp.ih, jpp.oh)
                            * nstl::min(jpp.iw, jpp.ow)
                    >= simd_w);
    if (!safe_load_n_store) return status::unimplemented;

    jpp.c_block = simd_w;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.c_tail != 0;

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    /* If channel_size is bigger than vlen, we can safely assume there is no
     * underflow of memory boundary, so always perform c_tail and save
     * a couple of compute cycles*/
    jpp.safe_c_tail = jpp.c_tail > 0 && jpp.c >= simd_w;

    switch (jpp.alg) {
        case pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            // avg_proc_dt (s32) defines granularity (because u8/s8 processed as s32)
            // sve_512 : 16
            const size_t msk_gran
                    = cpu_isa_traits<isa>::vlen / data_type_size(avg_proc_dt);
            const size_t msk_msk = (1ULL << msk_gran) - 1;
            size_t m = tail_mask;
            for (size_t ll = 0; ll < max_num_ll; ll++) {
                jpp.tail[ll] = m & msk_msk;
                m = m >> msk_gran;
            }
            break;
        }
        default: return status::unimplemented;
    }

    if (!post_ops_ok(jpp, *ppd->attr(), dst_d)) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
bool jit_uni_i8i8_pooling_fwd_ker_t<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (entries.empty()) return true;

    for (const auto &entry : entries) {
        if (entry.is_eltwise()) {
            jpp.with_eltwise = true;
        } else if (entry.is_binary()) {
            if (isa != sve_512
                    && entry.binary.src1_desc.data_type == data_type::bf16)
                return false;
            jpp.with_binary = true;
        } else
            return false;
    }

    jpp.with_postops = jpp.with_eltwise || jpp.with_binary;
    jpp.post_ops = post_ops;

    /*
     * TODO Currently eltwise/binary injectors assumes that data in vmm has f32 dt.
     * In max pooling data remains in i8 data type.
     */
    return IMPLICATION(jpp.with_postops, jpp.alg != pooling_max)
            && binary_injector::binary_args_broadcast_supported(
                    post_ops, dst_d);
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::pd_t::jit_conf() {
    return jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(jpp_, this);
}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::jit_uni_i8i8_pooling_fwd_t(const pd_t *apd)
    : primitive_t(apd), ker_(nullptr) {}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::~jit_uni_i8i8_pooling_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            new jit_uni_i8i8_pooling_fwd_ker_t<isa>(
                    pd()->jpp_, pd()->invariant_dst_md())));
    return ker_->create_kernel();
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_i8 = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst_i8 = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &jpp = pd()->jpp_;
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jpp.post_ops, ctx);
    /* Calculate when the memory-access will happen outisde of the memory
     * boundary, if so, compute a safe memory access. */
    const auto src_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(src_i8 + src_d.size() - 1)
            - (cpu_isa_traits<isa>::vlen - 1));

    const auto dst_safe_access = reinterpret_cast<char *>(
            reinterpret_cast<ptrdiff_t>(dst_i8 + dst_d.size() - 1)
            - (cpu_isa_traits<isa>::vlen - 1));

    parallel_nd(
            jpp.mb, jpp.od, jpp.oh, jpp.ow, [&](int n, int od, int oh, int ow) {
                const int id = nstl::max(od * jpp.stride_d - jpp.f_pad, 0);
                const int ih = nstl::max(oh * jpp.stride_h - jpp.t_pad, 0);
                const int iw = nstl::max(ow * jpp.stride_w - jpp.l_pad, 0);

                const int kd_start
                        = nstl::max(0, jpp.f_pad - od * jpp.stride_d);
                const int kd_end = nstl::min(
                        jpp.kd, jpp.id + jpp.f_pad - od * jpp.stride_d);
                const int kh_start
                        = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
                const int kh_end = nstl::min(
                        jpp.kh, jpp.ih + jpp.t_pad - oh * jpp.stride_h);
                const int kw_start
                        = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
                const int kw_end = nstl::min(
                        jpp.kw, jpp.iw + jpp.l_pad - ow * jpp.stride_w);

                auto p = call_params_t();
                p.src_i8 = &src_i8[get_offset(src_d, n, 0, id, ih, iw)
                        * src_d.data_type_size()];
                p.dst_i8 = &dst_i8[get_offset(dst_d, n, 0, od, oh, ow)
                        * dst_d.data_type_size()];
                p.kd_range = (size_t)(kd_end - kd_start);
                p.kh_range = (size_t)(kh_end - kh_start);
                p.kw_range = (size_t)(kw_end - kw_start);
                p.idivider = 1.0f
                        / ((jpp.alg == pooling_avg_exclude_padding)
                                        ? p.kd_range * p.kh_range * p.kw_range
                                        : jpp.kd * jpp.kh * jpp.kw);
                p.src_safe_access = src_safe_access;
                p.dst_safe_access = dst_safe_access;
                p.post_ops_binary_rhs_arg_vec
                        = post_ops_binary_rhs_arg_vec.data();
                (*ker_)(&p);
            });
}

// Explicit instantiation only for supported <isa> values.
//
template struct jit_uni_i8i8_pooling_fwd_ker_t<sve_512>;
template struct jit_uni_i8i8_pooling_fwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
