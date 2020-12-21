/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
* Copyright 2018 YANDEX LLC
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_pooling_pd.hpp"

#include "cpu/aarch64/jit_uni_pool_kernel.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::~jit_uni_pool_kernel() = default;

template <cpu_isa_t isa>
jit_uni_pool_kernel<isa>::jit_uni_pool_kernel(
        const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md)
    : jpp(ajpp), bf16_emu_(nullptr) {
    if (jpp.is_bf16 && !isa_has_bf16(jpp.isa))
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                bf16_emu_reserv_4, bf16_emu_reserv_5);

    if (jpp.with_postops) {

        static constexpr bool use_per_oc_spatial_strategy = false;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        static constexpr int sse41_single_block_size
                = cpu_isa_traits<asimd>::vlen / sizeof(float);
        size_t postop_tail = static_cast<size_t>(jpp.c_tail);
        const bool high_half_block_empty = isa == asimd
                && static_cast<size_t>(jpp.c_tail) > sse41_single_block_size;
        if (high_half_block_empty) postop_tail -= sse41_single_block_size;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<std::size_t>(VReg(4).getIdx()), XReg(7), XReg(2),
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(*dst_md), postop_tail, k_c_tail_mask,
                use_exact_tail_scalar_bcast};

        const binary_injector::static_params_t bsp {
                reg_param, use_per_oc_spatial_strategy, rhs_sp};

        postops_injector_
                = utils::make_unique<injector::jit_uni_postops_injector_t<isa>>(
                        this, jpp.post_ops, bsp);
    }
}

template <cpu_isa_t isa>
status_t jit_uni_pool_kernel<isa>::init_conf(jit_pool_conf_t &jpp,
        memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
        int nthreads) {

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(
            ppd->is_fwd() ? ppd->src_md() : ppd->diff_src_md());
    const memory_desc_wrapper dst_d(
            ppd->is_fwd() ? ppd->dst_md() : ppd->diff_dst_md());

    const int ndims = src_d.ndims();

    jpp.is_training = pd.prop_kind == prop_kind::forward_training;
    jpp.is_backward = pd.prop_kind == prop_kind::backward_data;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];
    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    const bool is_avx512 = utils::one_of(isa, sve_512);
    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = is_avx512 ? 16 : 8;

    jpp.alg = pd.alg_kind;

    using namespace format_tag;
    const auto blocked_fmt_tag = utils::one_of(isa, sve_512)
            ? utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    auto ncsp_fmt_tag = format_tag::undef;

    const unsigned int L3_cache_size_per_core
            = platform::get_per_core_cache_size(3);
    const size_t block_size
            = ((size_t)jpp.id * jpp.ih * jpp.iw + jpp.od * jpp.oh * jpp.ow)
            * jpp.c_block * types::data_type_size(src_d.data_type());

    const bool forward_ncsp_allowed = !jpp.is_backward
            && jpp.c_without_padding > 3
            && ((jpp.ih > 1 && jpp.iw > 1
                        && block_size <= L3_cache_size_per_core)
                    || src_d.data_type() == data_type::bf16);

    const bool backward_ncsp_allowed = jpp.is_backward
            && ((jpp.ih > 1 && jpp.iw > 1 && jpp.c_without_padding > 1
                        && block_size <= L3_cache_size_per_core)
                    || (src_d.data_type() == data_type::bf16
                            && !(jpp.alg == pooling_max
                                    && block_size > L3_cache_size_per_core)));

    ncsp_fmt_tag = ((forward_ncsp_allowed || backward_ncsp_allowed)
                           && isa == sve_512 && ndims <= 5)
            ? utils::pick(ndims - 3, ncw, nchw, ncdhw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    if (fmt_tag == ncsp_fmt_tag) {
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jit_memory_tag_kind_t::ncsp;
    } else {
        jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                && dst_d.data_type() == data_type::bf16);
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag)
                ? jit_memory_tag_kind_t::nspc
                : jit_memory_tag_kind_t::blocked;
    }

    jpp.isa = isa;

    const bool args_ok = true && mayiuse(isa) && (fmt_tag != format_tag::undef)
            && IMPLICATION(jpp.is_bf16, mayiuse(sve_512))
            && utils::one_of(pd.alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    jpp.c = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            ? utils::rnd_up(jpp.c_without_padding, jpp.c_block)
            : jpp.c_without_padding;
    if (jpp.tag_kind == jit_memory_tag_kind_t::blocked)
        assert(src_d.padded_dims()[1] == jpp.c);
    jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
    jpp.c_tail = jpp.c_without_padding % jpp.c_block;
    jpp.is_c_padded = jpp.tag_kind == jit_memory_tag_kind_t::blocked
            && src_d.padded_dims()[1] != jpp.c_without_padding;

    jpp.stride_d = (ndims == 5) ? pd.strides[0] : 1;
    jpp.stride_h = (ndims == 3) ? 1 : pd.strides[ndims - 4];
    jpp.stride_w = pd.strides[ndims - 3];
    jpp.kd = (ndims == 5) ? pd.kernel[0] : 1;
    jpp.kh = (ndims == 3) ? 1 : pd.kernel[ndims - 4];
    jpp.kw = pd.kernel[ndims - 3];

    jpp.f_pad = (ndims == 5) ? pd.padding[0][0] : 0;
    jpp.t_pad = (ndims == 3) ? 0 : pd.padding[0][ndims - 4];
    jpp.l_pad = pd.padding[0][ndims - 3];

    const int back_pad = calculate_end_padding(
            jpp.f_pad, jpp.od, jpp.id, jpp.stride_d, jpp.kd);
    const int bottom_pad = calculate_end_padding(
            jpp.t_pad, jpp.oh, jpp.ih, jpp.stride_h, jpp.kh);
    const int right_pad = calculate_end_padding(
            jpp.l_pad, jpp.ow, jpp.iw, jpp.stride_w, jpp.kw);

    if (jpp.f_pad >= jpp.kd || jpp.t_pad >= jpp.kh || jpp.l_pad >= jpp.kw
            || back_pad >= jpp.kd || bottom_pad >= jpp.kh
            || right_pad >= jpp.kw)
        return status::unimplemented;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = is_avx512 ? 16 : 4;

        if ((isa == sve_128 || isa == sve_256) && jpp.c_tail > 0)
            // Additional register needed for tail mask
            jpp.ur -= 1;

        if (jpp.is_training)
            jpp.ur = is_avx512 ? 9 : 3;
        else if (jpp.is_backward)
            jpp.ur = is_avx512 ? 6 : 3;
    } else {
        if (jpp.is_backward)
            jpp.ur = is_avx512 ? 12 : 6;
        else
            jpp.ur = is_avx512 ? 24 : 12;
    }
    if (jpp.is_bf16) {
        jpp.ur = (!isa_has_bf16(jpp.isa))
                ? jpp.ur - 4 // Free registers for AVX512 emulation
                : jpp.ur - 1; // Free register for cvt from bf16 to f32
    }

    // select jpp.ur_bc
    if (jpp.tag_kind == jit_memory_tag_kind_t::nspc) {
        auto min_ur_w = nstl::max(1, utils::div_up(jpp.l_pad, jpp.stride_w));
        int min_ur_w1 = utils::div_up(right_pad, jpp.stride_w);
        if (min_ur_w < min_ur_w1) { min_ur_w = min_ur_w1; }
        jpp.ur_bc = nstl::min(jpp.nb_c, nstl::max(1, jpp.ur / min_ur_w));
        //take into account threading - to have enough work for parallelization
        float best_eff = 0;
        for (int ur_bc = jpp.ur_bc; ur_bc > 0; ur_bc--) {

            const auto nb2_c = utils::div_up(jpp.nb_c, ur_bc);
            auto work = jpp.is_backward
                    ? (ndims == 5 && jpp.simple_alg ? jpp.od : 1)
                    : (ndims == 5 ? jpp.od : jpp.oh);
            work *= jpp.mb * nb2_c;
            auto eff = (float)work / utils::rnd_up(work, nthreads);
            if (eff > best_eff) {

                best_eff = eff;
                jpp.ur_bc = ur_bc;
            }
            if (eff > 0.9) break; // Heuristic threshold
        }

        //take into account cache re-usage after zeroing on backward
        if (jpp.is_backward && ndims < 5) {
            //unsupported
        }

        jpp.ur_bc_tail = jpp.nb_c % jpp.ur_bc;
    } else {
        jpp.ur_bc = 1;
        jpp.ur_bc_tail = 0;
    }
    auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
    if (utils::div_up(jpp.l_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;
    if (utils::div_up(right_pad, jpp.stride_w) > ur_w)
        return status::unimplemented;

    // scratchpad for c_block slice of input and/or output
    using namespace memory_tracking::names;
    const int nscr = nstl::min(dnnl_get_max_threads(), jpp.mb * jpp.nb_c);
    if (jpp.tag_kind == jit_memory_tag_kind_t::ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                jpp.c_block * jpp.id * jpp.ih * jpp.iw * nscr, jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr, jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr);
    }

    const auto attr = *ppd->attr();
    if (!post_ops_ok(jpp, attr, dst_d)) return status::unimplemented;

    jpp.post_ops = attr.post_ops_;

    return status::success;
}

static int reg_ind(int shift, int bc, int j, int ur_bc, int ur_w) noexcept {
    return shift * ur_bc * ur_w + bc * ur_w + j;
};

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::prepare_tail_mask() {
    if (is_superset(isa, sve_512)) {
        size_t c_tail_mask = (1ULL << jpp.c_tail) - 1ULL;
        /* PRegS(IDX(k_c_tail_mask)) keeps flags in the context
           of 8-bit elements. */
        xa_->mov_imm(X_TMP_0, c_tail_mask);
        xa_->sub(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
        str(X_TMP_0, Xbyak_aarch64::ptr(X_TRANSLATOR_STACK));
        ldr(PReg(k_c_tail_mask), Xbyak_aarch64::ptr(X_TRANSLATOR_STACK));
        xa_->add(X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 8);
    } else if (isa == sve_128 || isa == sve_256) {
        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};
        xa_->mov_imm(tmp_gpr, reinterpret_cast<size_t>(&mask[8 - jpp.c_tail]));
        ld1w(ZRegS(IDX(vmm_c_tail_mask)), p_lsb / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(tmp_gpr));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::put_one_in_vmm() {
    xa_->mov_imm(tmp_gpr, 1);
    uni_broadcast_reg_val(tmp_gpr.getIdx(), vmm_one.getIdx());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::uni_broadcast_reg_val(
        const int reg_idx, const int vmm_idx) {
    ptrue(p_tmp0.d, VL2);
    xa_->mov(ZRegD(vmm_idx), p_tmp0 / T_m, 0);
    ptrue(p_tmp0.d, VL1);
    xa_->mov(ZRegD(vmm_idx), p_tmp0 / T_m, XReg(reg_idx));

    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        dup(ZRegS(vmm_idx), ZRegS(vmm_idx)[0]);
    } else if (vlen == 32) {
        dup(ZRegS(vmm_idx), ZRegS(vmm_idx)[0]);
        xa_->mov(ZRegS(vmm_idx), P_MSB_256 / T_m, 0);
    } else if (vlen == 16) {
        dup(VReg4S(vmm_idx), VReg4S(vmm_idx)[0]);
        xa_->mov(ZRegS(vmm_idx), P_MSB_384 / T_m, 0);
    } else {
        assert(!"unreachable");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::push_vmm_val(const int idx) {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    TReg val_to_store(idx);
    sub_imm(XReg(idx), XReg(idx), val_to_store.getBit(), x_tmp_0);

    int vlen = cpu_isa_traits<isa>::vlen;

    if (vlen == 64) {
        str(ZReg(IDX(val_to_store)), Xbyak_aarch64::ptr(X_SP));
    } else if (vlen == 32) {
        st1w(ZRegS(IDX(val_to_store)), p_lsb, Xbyak_aarch64::ptr(X_SP));
    } else if (vlen == 16) {
        str(QReg(IDX(val_to_store)), Xbyak_aarch64::ptr(X_SP));
    } else {
        assert(!"unreachable");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::pop_vmm_val(const int idx) {
    using TReg = typename cpu_isa_traits<isa>::TReg;
    TReg val_to_load(idx);

    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        ldr(ZReg(IDX(val_to_load)), Xbyak_aarch64::ptr(X_SP));
    } else if (vlen == 32) {
        ld1w(ZRegS(IDX(val_to_load)), p_lsb / Xbyak_aarch64::T_z,
                Xbyak_aarch64::ptr(X_SP));
    } else if (vlen == 16) {
        ldr(QReg(z_tmp0.getIdx()), Xbyak_aarch64::ptr(X_SP));
        xa_->mov(ZRegD(IDX(val_to_load)), p_lsb / T_m, z_tmp0.d);
    } else {
        assert(!"unreachable");
    }

    add_imm(XReg(9), XReg(9), val_to_load.getBit(), x_tmp_0);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::load(const int idx, const xreg_t &reg_ptr,
        const int offset, const bool is_c_tail_proccessing) {
    if (jpp.is_bf16) {
        /*TODO: maybe use vpmovzxwd + vpslld,
             * in order to free up vmm_idx() register */
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
                if (is_c_tail_proccessing) {
                    assert(!"unreachable");
                } else {
                    ldr(z_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
                    zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);

                    uxth(ZReg(idx).s, p_512 / T_m, z_tmp0.s);

                    lsl(ZReg(idx).s, ZReg(idx).s, 16);
                }
            } else if (vlen == 32) {
                add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
                if (is_c_tail_proccessing) {
                    assert(!"unreachable");
                } else {
                    ldr(z_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
                    zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);

                    uxth(ZReg(idx).s, p_512 / T_m, z_tmp0.s);
                    xa_->mov(ZReg(idx).s, P_MSB_256 / T_m, 0);

                    lsl(ZReg(idx).s, ZReg(idx).s, 16);
                }
            } else if (vlen == 16) {
                add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
                if (is_c_tail_proccessing) {
                    assert(!"unreachable");
                } else {
                    ldr(z_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
                    zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);

                    uxth(ZReg(idx).s, p_512 / T_m, z_tmp0.s);
                    xa_->mov(ZReg(idx).s, P_MSB_384 / T_m, 0);

                    lsl(ZReg(idx).s, ZReg(idx).s, 16);
                }
            } else {
                assert(!"unreachable");
            }
        } else {
            add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);

            ld1w(ZRegS(idx), p_256 / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_tmp_addr));
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                xa_->mov(z_tmp0.h, 31);
                xa_->and_(z_tmp0.b, p_512, ZRegB(reg_idx()));
                for (int i = 0; i < 16; i++) {
                    cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    dup(z_tmp2.h, ZRegH(idx)[i]);
                    xa_->mov(z_tmp3.h, p_tmp1 / T_m, z_tmp2.h);
                }
                xa_->sub(z_tmp0.h, 16);
                for (int i = 0; i < 16; i++) {
                    cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    dup(z_tmp2.h, ZRegH(idx)[16 + i]);
                    xa_->mov(z_tmp3.h, p_tmp1 / T_m, z_tmp2.h);
                }
                xa_->mov(ZRegH(idx), 0);
                xa_->mov(ZRegH(idx), k_mask_cvt / T_m, z_tmp3.h);
            } else if (vlen == 32) {
                xa_->mov(z_tmp0.h, 15);
                xa_->and_(z_tmp0.b, p_512, ZRegB(reg_idx()));
                for (int i = 0; i < 16; i++) {
                    cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    dup(z_tmp2.h, ZRegH(idx)[i]);
                    xa_->mov(z_tmp3.h, p_tmp1 / T_m, z_tmp2.h);
                }
                xa_->mov(ZRegH(idx), 0);
                xa_->mov(ZRegH(idx), k_mask_cvt / T_m, z_tmp3.h);
                xa_->mov(ZRegH(idx), P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                xa_->mov(z_tmp0.h, 15);
                xa_->and_(z_tmp0.b, p_512, ZRegB(reg_idx()));
                for (int i = 0; i < 16; i++) {
                    cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    dup(z_tmp2.h, ZRegH(idx)[i]);
                    xa_->mov(z_tmp3.h, p_tmp1 / T_m, z_tmp2.h);
                }
                xa_->mov(ZRegH(idx), 0);
                xa_->mov(ZRegH(idx), k_mask_cvt / T_m, z_tmp3.h);
                xa_->mov(ZRegH(idx), P_MSB_384 / T_m, 0);
            } else {
                assert(!"unreachable");
            }
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == asimd) {
                assert(!"unsupported");
            } else if (isa == sve_128 || isa == sve_256) {
                assert(!"unsupported");
            } else {
                add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
                pfalse(p9.b);
                zip1(p1.b, k_c_tail_mask.b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                ld1w(ZRegS(idx), p1 / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(x_tmp_addr));
            }
        } else {
            add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
            ld1w(ZRegS(idx), p_lsb / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(x_tmp_addr));
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::store(const int idx,
        const xreg_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    if (jpp.is_bf16) {
        add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                assert(!"unsupported");
                // under construction
                pfalse(p9.b);
                zip1(p1.b, k_c_tail_mask.b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                st1h(ZRegH(idx), k_c_tail_mask, Xbyak_aarch64::ptr(x_tmp_addr));
            } else if (vlen == 32) {
                bic(p_tmp0.b, P_ALL_ONE / Xbyak_aarch64::T_z, k_c_tail_mask.b,
                        P_MSB_256.b);
                st1h(ZRegH(idx), p_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
            } else if (vlen == 16) {
                bic(p_tmp0.b, P_ALL_ONE / Xbyak_aarch64::T_z, k_c_tail_mask.b,
                        P_MSB_384.b);
                st1h(ZRegH(idx), p_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
            } else {
                assert(!"unreachable");
            }
        } else {
            add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
            st1w(ZRegS(idx), p_lsb, Xbyak_aarch64::ptr(x_tmp_addr));
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == asimd) {
                assert(!"unsupported");
            } else if (isa == sve_128 || isa == sve_256) {
                assert(!"unsupported");
            } else {
                add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
                pfalse(p9.b);
                zip1(p1.b, k_c_tail_mask.b, p9.b);
                zip1(p1.h, p1.h, p9.h);
                st1w(ZRegS(idx), p1, Xbyak_aarch64::ptr(x_tmp_addr));
            }
        } else {
            add_imm(x_tmp_addr, reg_ptr, offset, x_tmp_0);
            st1w(ZRegS(idx), p_lsb, Xbyak_aarch64::ptr(x_tmp_addr));
        }
    }
}
template <cpu_isa_t isa>
bool jit_uni_pool_kernel<isa>::post_ops_ok(jit_pool_conf_t &jpp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d) {
    const auto &post_ops = attr.post_ops_;
    const auto &entries = post_ops.entry_;
    jpp.with_postops = false;
    jpp.with_eltwise = false;
    jpp.with_binary = false;

    if (!jpp.is_backward) {
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
    }

    return binary_injector::binary_args_broadcast_supported(post_ops, dst_d);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::apply_postops(int ur_bc, int ur_w, int c_block,
        const std::function<bool(int)> &is_tail_predicate) {

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    const int end_idx = vmm_idx_upper_bound() + 1;
    const int start_idx = end_idx - (ur_bc * ur_w);
    const bool sse41_postops_disabled
            = isa == asimd && disable_postops_when_sse_high_half_processed_;

    if (jpp.with_binary && !sse41_postops_disabled) {

        static constexpr int sse41_simd_w
                = cpu_isa_traits<asimd>::vlen / sizeof(float);
        const int sse_elem_off = sse_high_half ? sse41_simd_w : 0;
        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto vmm_idx
                        = vreg(reg_ind(0, bci, jj, ur_bc, ur_w)).getIdx();

                add_imm(reg_adrimm, reg_param, GET_OFF(c_elem_off), x_tmp_0);

                rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
                        vmm_idx, Xbyak_aarch64::ptr(reg_adrimm));
                rhs_arg_params.vmm_idx_to_oc_elem_off_val.emplace(
                        vmm_idx, bci * c_block + sse_elem_off);
                if (is_tail_predicate && is_tail_predicate(bci))
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }
    postops_injector_->compute_vector_range(start_idx, end_idx, rhs_arg_params);
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::maybe_recalculate_divisor(
        int jj, int ur_w, int pad_l, int pad_r, bool with_c_tail_proccessing) {
    if (jpp.alg == pooling_avg_exclude_padding) {
        int kw = jpp.kw;
        int stride_w = jpp.stride_w;

        int non_zero_kw = kw;
        non_zero_kw -= nstl::max(0, pad_l - jj * stride_w);
        non_zero_kw -= nstl::max(0, pad_r - (ur_w - 1 - jj) * stride_w);

        if (non_zero_kw != prev_kw) {
            xa_->mov_imm(tmp_gpr, float2int((float)non_zero_kw));

            ptrue(p_tmp0.d, VL2);
            xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
            ptrue(p_tmp0.d, VL1);
            xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, tmp_gpr);

            const int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_384 / T_m, 0);
            } else {
                assert(!"unreachable");
            }
            if (with_c_tail_proccessing && (isa == sve_128 || isa == sve_256)) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                uni_broadcast_reg_val(
                        reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
            }

            if (vlen == 64) {
                xa_->fmul(ZReg(IDX(vmm_tmp)).s, ZReg(IDX(vmm_tmp)).s,
                        ZReg(IDX(vmm_ker_area_h)).s);
            } else if (vlen == 32) {
                xa_->fmul(ZReg(IDX(vmm_tmp)).s, ZReg(IDX(vmm_tmp)).s,
                        ZReg(IDX(vmm_ker_area_h)).s);
                xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                xa_->fmul(VReg(IDX(vmm_tmp)).s4, VReg(IDX(vmm_tmp)).s4,
                        VReg(IDX(vmm_ker_area_h)).s4);
            } else {
                assert(!"unreachable");
            }
            if (with_c_tail_proccessing && (isa == sve_128 || isa == sve_256)) {
                pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
            prev_kw = non_zero_kw;
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::avg_step(int ur_w, int ur_bc, int pad_l,
        int pad_r, bool with_c_tail_proccessing) {

    auto iw = jpp.iw;
    auto kw = jpp.kw;
    auto stride_w = jpp.stride_w;
    auto c_block = jpp.c_block;
    auto dt_size = jpp.dt_size;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        if (isa == asimd && !jpp.is_c_padded) {
            assert(!"unsupported");
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward)
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
        for (int bci = 0; bci < ur_bc; bci++) {
            const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
            auto accvr = vreg(accr_i);
            if (jpp.is_backward) {
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
                load(accvr.getIdx(), xreg_output, output_offset,
                        is_tail_processing(bci));
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    xa_->fdiv(ZRegS(IDX(accvr)), p_512, ZRegS(IDX(vmm_tmp)));
                } else if (vlen == 32) {
                    xa_->fdiv(ZRegS(IDX(accvr)), p_512, ZRegS(IDX(vmm_tmp)));
                    xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    xa_->fdiv(VReg(IDX(accvr)).s4, VReg(IDX(accvr)).s4,
                            VReg(IDX(vmm_tmp)).s4);
                } else {
                    assert(!"unreachable");
                }
            } else {
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    eor(ZReg(IDX(accvr)).d, ZReg(IDX(accvr)).d,
                            ZReg(IDX(accvr)).d);
                } else if (vlen == 32) {
                    eor(ZRegD(IDX(accvr)), ZRegD(IDX(accvr)),
                            ZRegD(IDX(accvr)));
                    xa_->mov(ZRegS(IDX(accvr)), P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    eor(VReg16B(IDX(accvr)), VReg16B(IDX(accvr)),
                            VReg16B(IDX(accvr)));
                } else {
                    assert(!"unreachable");
                }
            }
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        str(reg_input, pre_ptr(X_TRANSLATOR_STACK, -8));

        str(reg_output, pre_ptr(X_TRANSLATOR_STACK, -8));

        xa_->mov(aux_reg_input_d, reg_input);

        add_imm(x_tmp_addr, reg_param, GET_OFF(kd_padding), x_tmp_0);
        ldr(ki, Xbyak_aarch64::ptr(x_tmp_addr));
        L(kd_label);

        xa_->mov(aux_reg_input, aux_reg_input_d);
    } else {
        xa_->mov(aux_reg_input, reg_input);
    }

    eor(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);

            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                auto inpvr = vreg(inpr_i);
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = dt_size * aux_input_offset;
                if (jpp.is_backward) {
                    auto inpyr = yreg(inpr_i);
                    load(reg_idx(inpr_i), aux_xreg_input, input_offset,
                            is_tail_processing(bci));

                    int vlen = cpu_isa_traits<isa>::vlen;

                    if (vlen == 64) {
                        xa_->fadd(ZReg(IDX(inpvr)).s, ZReg(IDX(inpvr)).s,
                                ZReg(IDX(accvr)).s);
                    } else if (vlen == 32) {
                        xa_->fadd(ZReg(IDX(inpvr)).s, ZReg(IDX(inpvr)).s,
                                ZReg(IDX(accvr)).s);
                        xa_->mov(ZReg(IDX(inpvr)).s, P_MSB_256 / T_m, 0);
                    } else if (vlen == 16) {
                        xa_->fadd(VReg(IDX(inpvr)).s4, VReg(IDX(inpvr)).s4,
                                VReg(IDX(accvr)).s4);
                    } else {
                        assert(!"unreachable");
                    }
                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa)) {
                            bf16_emu_->vcvtneps2bf16(inpyr, zreg(inpr_i));
                        } else {
                            assert(!"unreachable");
                        }
                    }
                    store(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                } else {
                    if (jpp.is_bf16 || is_tail_processing(bci)
                            || (isa == asimd
                                    && c_off % (jpp.c_block / 2) != 0)) {
                        load(vmm_tmp_1.getIdx(), aux_xreg_input, input_offset,
                                is_tail_processing(bci));

                        int vlen = cpu_isa_traits<isa>::vlen;

                        if (vlen == 64) {
                            xa_->fadd(ZReg(IDX(accvr)).s, ZReg(IDX(accvr)).s,
                                    ZReg(IDX(vmm_tmp_1)).s);
                        } else if (vlen == 32) {
                            xa_->fadd(ZReg(IDX(accvr)).s, ZReg(IDX(accvr)).s,
                                    ZReg(IDX(vmm_tmp_1)).s);
                            xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                        } else if (vlen == 16) {
                            xa_->fadd(VReg(IDX(accvr)).s4, VReg(IDX(accvr)).s4,
                                    VReg(IDX(vmm_tmp_1)).s4);
                        } else {
                            assert(!"unreachable");
                        }
                    } else {
                        int vlen = cpu_isa_traits<isa>::vlen;
                        add_imm(x_tmp_addr, aux_reg_input, input_offset,
                                x_tmp_0);

                        if (vlen == 64) {
                            ldr(z_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
                            xa_->fadd(ZReg(IDX(accvr)).s, ZReg(IDX(accvr)).s,
                                    z_tmp0.s);
                        } else if (vlen == 32) {
                            ldr(z_tmp0, Xbyak_aarch64::ptr(x_tmp_addr));
                            xa_->fadd(ZReg(IDX(accvr)).s, ZReg(IDX(accvr)).s,
                                    z_tmp0.s);
                            xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                        } else if (vlen == 16) {
                            ld1(VReg(z_tmp0.getIdx()).s4,
                                    Xbyak_aarch64::ptr(x_tmp_addr));
                            xa_->fadd(VReg(IDX(accvr)).s4, VReg(IDX(accvr)).s4,
                                    VReg(z_tmp0.getIdx()).s4);
                        } else {
                            assert(!"unreachable");
                        }
                    }
                }
            }
        }
        add_imm(aux_reg_input, aux_reg_input, (jpp.dt_size * iw * c_off),
                x_tmp_0);

        adds(kj, kj, 1);

        xa_->cmp(kj, reg_kh);

        b(LT, kh_label);
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d,
                (jpp.dt_size * jpp.ih * iw * c_off), x_tmp_0);

        subs(ki, ki, 1);

        xa_->mov_imm(x_tmp_0, 0);
        xa_->cmp(ki, x_tmp_0);

        b(GT, kd_label);

        ldr(reg_output, post_ptr(X_TRANSLATOR_STACK, 8));

        ldr(reg_input, post_ptr(X_TRANSLATOR_STACK, 8));
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto accvr = vreg(accr_i);
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    xa_->fdiv(ZRegS(IDX(accvr)), p_512, ZRegS(IDX(vmm_tmp)));
                } else if (vlen == 32) {
                    xa_->fdiv(ZRegS(IDX(accvr)), p_512, ZRegS(IDX(vmm_tmp)));
                    xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                } else if (vlen == 16) {
                    xa_->fdiv(VReg(IDX(accvr)).s4, VReg(IDX(accvr)).s4,
                            VReg(IDX(vmm_tmp)).s4);
                } else {
                    assert(!"unreachable");
                }
            }
        }

        if (jpp.with_postops) {
            apply_postops(ur_bc, ur_w, c_block, is_tail_processing);
        }

        for (int jj = 0; jj < ur_w; jj++) {
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
                const auto output_offset
                        = dt_size * (jj * c_off + bci * c_block);
                if (jpp.is_bf16) {
                    const auto acczr = zreg(accr_i);
                    const auto accyr = yreg(accr_i);
                    if (!isa_has_bf16(jpp.isa)) {
                        bf16_emu_->vcvtneps2bf16(accyr, acczr);
                    } else {
                        assert(!"unsupported");
                    }
                }
                store(reg_idx(accr_i), xreg_output, output_offset,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_fwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {
    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto is_tail_processing = [&](int bc) {
        if (isa == asimd && !jpp.is_c_padded) {
            assert(!"unsupported");
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    xa_->mov_imm(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));

    ptrue(p_tmp0.d, VL2);
    xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
    ptrue(p_tmp0.d, VL1);
    xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, tmp_gpr);

    const int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
    } else if (vlen == 32) {
        dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
        xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_256 / T_m, 0);
    } else if (vlen == 16) {
        dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
        xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_384 / T_m, 0);
    } else {
        assert(!"unreachable");
    }

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
        int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) {
            xa_->mov(ZRegD(IDX(accvr)), ZRegD(IDX(vmm_tmp)));
        } else if (vlen == 32) {
            xa_->mov(ZRegD(IDX(accvr)), ZRegD(IDX(vmm_tmp)));
            xa_->mov(ZRegS(IDX(accvr)), P_MSB_256 / T_m, 0);
        } else if (vlen == 16) {
            xa_->mov(VReg16B(IDX(accvr)), VReg16B(IDX(vmm_tmp)));
        } else {
            assert(!"unreachable");
        }
        if (jpp.is_training) {
            const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w));
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                eor(ZReg(IDX(indvr)).d, ZReg(IDX(indvr)).d, ZReg(IDX(indvr)).d);
            } else if (vlen == 32) {
                eor(ZRegD(IDX(indvr)), ZRegD(IDX(indvr)), ZRegD(IDX(indvr)));
                xa_->mov(ZRegS(IDX(indvr)), P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                eor(VReg16B(IDX(indvr)), VReg16B(IDX(indvr)),
                        VReg16B(IDX(indvr)));
            } else {
                assert(!"unreachable");
            }
        }
    }
    if (jpp.is_training) {

        ptrue(p_tmp0.d, VL2);
        xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
        ptrue(p_tmp0.d, VL1);
        xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, reg_k_shift);

        int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) {
            dup(ZRegS(IDX(vmm_k_offset)), ZRegS(IDX(xmm_tmp))[0]);
        } else if (vlen == 32) {
            dup(ZRegS(IDX(vmm_k_offset)), ZRegS(IDX(xmm_tmp))[0]);
            xa_->mov(ZRegS(IDX(vmm_k_offset)), P_MSB_256 / T_m, 0);
        } else if (vlen == 16) {
            dup(VReg4S(IDX(vmm_k_offset)), VReg4S(IDX(xmm_tmp))[0]);
            xa_->mov(ZRegS(IDX(vmm_k_offset)), P_MSB_384 / T_m, 0);
        } else {
            assert(!"unreachable");
        }
    }
    if (jpp.ndims == 5) {

        str(reg_input, pre_ptr(X_TRANSLATOR_STACK, -8));

        str(reg_output, pre_ptr(X_TRANSLATOR_STACK, -8));

        xa_->mov(aux_reg_input_d, reg_input);

        add_imm(x_tmp_addr, reg_param, GET_OFF(kd_padding), x_tmp_0);
        ldr(ki, Xbyak_aarch64::ptr(x_tmp_addr));
        L(kd_label);

        xa_->mov(aux_reg_input, aux_reg_input_d);
    } else {

        xa_->mov(aux_reg_input, reg_input);
    }

    eor(kj, kj, kj);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start
                    = nstl::max(0, utils::div_up(pad_l - ki, stride_w)); //test
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto accvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i);
                const auto indvr = vreg(reg_ind(2, bci, jj, ur_bc, ur_w));
                const auto cvtvr = vreg(reg_ind(3, bci, jj, ur_bc, ur_w));
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = jpp.dt_size * aux_input_offset;
                load(reg_idx(inpr_i), aux_xreg_input, input_offset,
                        is_tail_processing(bci));
                if (isa == asimd) {
                    assert(!"unsupported");
                } else if (isa == sve_128 || isa == sve_256) {
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        uint cmpDstIdx = IDX(cvtvr);
                        uint cmpMaskIdx = p_512.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OQ
                            case 3: //UNORD_Q
                            case 7: //ORD_Q
                            case 11: //FALSE_OQ
                            case 15: //TRUE_UQ
                            case 19: //UNORD_S
                            case 23: //ORD_S
                            case 27: //FALSE_OS
                            case 31: //TRUE_US
                            default: assert(!"unreachable"); break;
                        }
                        lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                        cmpgt(p10.s, p_256 / Xbyak_aarch64::T_z, z_tmp0.s, 0);
                        xa_->mov(ZReg(IDX(accvr)).s, p10 / T_m,
                                ZReg(IDX(inpvr)).s);
                        xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                        xa_->mov(ZReg(IDX(accvr)).s, p_tmp0 / T_m,
                                ZReg(IDX(accvr)).s);
                        xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                        if (jpp.is_training) {
                            if (vlen == 64) {
                                assert(!"unreachable");
                            } else if (vlen == 32) {
                                lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                                cmpgt(p10.s, p_256 / Xbyak_aarch64::T_z,
                                        z_tmp0.s, 0);
                                xa_->mov(ZReg(IDX(indvr)).s, p10 / T_m,
                                        ZReg(IDX(vmm_k_offset)).s);
                                xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                xa_->mov(ZReg(IDX(indvr)).s, p_tmp0 / T_m,
                                        ZReg(IDX(indvr)).s);
                                xa_->mov(
                                        ZReg(IDX(indvr)).s, P_MSB_256 / T_m, 0);
                            } else if (vlen == 16) {
                                lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                                cmpgt(p10.s, p_128 / Xbyak_aarch64::T_z,
                                        z_tmp0.s, 0);
                                xa_->mov(ZReg(IDX(indvr)).s, p10 / T_m,
                                        ZReg(IDX(vmm_k_offset)).s);
                                xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                xa_->mov(ZReg(IDX(indvr)).s, p_tmp0 / T_m,
                                        ZReg(IDX(indvr)).s);
                                xa_->mov(
                                        ZReg(IDX(indvr)).s, P_MSB_384 / T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    } else if (vlen == 32) {
                        xa_->mov(p_tmp0.b, P_ALL_ONE / Xbyak_aarch64::T_z,
                                P_MSB_256.b);
                        uint cmpDstIdx = IDX(cvtvr);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OQ
                            case 3: //UNORD_Q
                            case 7: //ORD_Q
                            case 11: //FALSE_OQ
                            case 15: //TRUE_UQ
                            case 19: //UNORD_S
                            case 23: //ORD_S
                            case 27: //FALSE_OS
                            case 31: //TRUE_US
                            default: assert(!"unreachable"); break;
                        }
                        lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                        cmpgt(p10.s, p_256 / Xbyak_aarch64::T_z, z_tmp0.s, 0);
                        xa_->mov(ZReg(IDX(accvr)).s, p10 / T_m,
                                ZReg(IDX(inpvr)).s);
                        xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                        xa_->mov(ZReg(IDX(accvr)).s, p_tmp0 / T_m,
                                ZReg(IDX(accvr)).s);
                        xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                        if (jpp.is_training) {
                            if (vlen == 64) {
                                assert(!"unreachable");
                            } else if (vlen == 32) {
                                lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                                cmpgt(p10.s, p_256 / Xbyak_aarch64::T_z,
                                        z_tmp0.s, 0);
                                xa_->mov(ZReg(IDX(indvr)).s, p10 / T_m,
                                        ZReg(IDX(vmm_k_offset)).s);
                                xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                xa_->mov(ZReg(IDX(indvr)).s, p_tmp0 / T_m,
                                        ZReg(IDX(indvr)).s);
                                xa_->mov(
                                        ZReg(IDX(indvr)).s, P_MSB_256 / T_m, 0);
                            } else if (vlen == 16) {
                                lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                                cmpgt(p10.s, p_128 / Xbyak_aarch64::T_z,
                                        z_tmp0.s, 0);
                                xa_->mov(ZReg(IDX(indvr)).s, p10 / T_m,
                                        ZReg(IDX(vmm_k_offset)).s);
                                xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                xa_->mov(ZReg(IDX(indvr)).s, p_tmp0 / T_m,
                                        ZReg(IDX(indvr)).s);
                                xa_->mov(
                                        ZReg(IDX(indvr)).s, P_MSB_384 / T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    } else if (vlen == 16) {
                        xa_->mov(p_tmp0.b, P_ALL_ONE / Xbyak_aarch64::T_z,
                                P_MSB_256.b);
                        uint cmpDstIdx = IDX(cvtvr);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OQ
                            case 3: //UNORD_Q
                            case 7: //ORD_Q
                            case 11: //FALSE_OQ
                            case 15: //TRUE_UQ
                            case 19: //UNORD_S
                            case 23: //ORD_S
                            case 27: //FALSE_OS
                            case 31: //TRUE_US
                            default: assert(!"unreachable"); break;
                        }
                        lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                        cmpgt(p10.s, p_128 / Xbyak_aarch64::T_z, z_tmp0.s, 0);
                        xa_->mov(ZReg(IDX(accvr)).s, p10 / T_m,
                                ZReg(IDX(inpvr)).s);
                        xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                        xa_->mov(ZReg(IDX(accvr)).s, p_tmp0 / T_m,
                                ZReg(IDX(accvr)).s);
                        xa_->mov(ZReg(IDX(accvr)).s, P_MSB_384 / T_m, 0);
                        if (jpp.is_training) {
                            if (vlen == 64) {
                                assert(!"unreachable");
                            } else if (vlen == 32) {
                                lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                                cmpgt(p10.s, p_256 / Xbyak_aarch64::T_z,
                                        z_tmp0.s, 0);
                                xa_->mov(ZReg(IDX(indvr)).s, p10 / T_m,
                                        ZReg(IDX(vmm_k_offset)).s);
                                xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                xa_->mov(ZReg(IDX(indvr)).s, p_tmp0 / T_m,
                                        ZReg(IDX(indvr)).s);
                                xa_->mov(
                                        ZReg(IDX(indvr)).s, P_MSB_256 / T_m, 0);
                            } else if (vlen == 16) {
                                lsr(z_tmp0.s, ZReg(IDX(cvtvr)).s, 31);
                                cmpgt(p10.s, p_128 / Xbyak_aarch64::T_z,
                                        z_tmp0.s, 0);
                                xa_->mov(ZReg(IDX(indvr)).s, p10 / T_m,
                                        ZReg(IDX(vmm_k_offset)).s);
                                xa_->not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                xa_->mov(ZReg(IDX(indvr)).s, p_tmp0 / T_m,
                                        ZReg(IDX(indvr)).s);
                                xa_->mov(
                                        ZReg(IDX(indvr)).s, P_MSB_384 / T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    } else {
                        assert(!"unreachable");
                    }
                } else {
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        uint cmpDstIdx = IDX(k_store_mask);
                        uint cmpMaskIdx = p_512.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OQ
                            case 3: //UNORD_Q
                            case 7: //ORD_Q
                            case 11: //FALSE_OQ
                            case 15: //TRUE_UQ
                            case 19: //UNORD_S
                            case 23: //ORD_S
                            case 27: //FALSE_OS
                            case 31: //TRUE_US
                            default: assert(!"unreachable"); break;
                        }
                    } else if (vlen == 32) {
                        xa_->mov(p_tmp0.b, P_ALL_ONE / Xbyak_aarch64::T_z,
                                P_MSB_256.b);
                        uint cmpDstIdx = IDX(k_store_mask);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OQ
                            case 3: //UNORD_Q
                            case 7: //ORD_Q
                            case 11: //FALSE_OQ
                            case 15: //TRUE_UQ
                            case 19: //UNORD_S
                            case 23: //ORD_S
                            case 27: //FALSE_OS
                            case 31: //TRUE_US
                            default: assert(!"unreachable"); break;
                        }
                    } else if (vlen == 16) {
                        xa_->mov(p_tmp0.b, P_ALL_ONE / Xbyak_aarch64::T_z,
                                P_MSB_384.b);
                        uint cmpDstIdx = IDX(k_store_mask);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                fcmeq(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                fcmlt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                fcmle(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                fcmne(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                fcmge(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                fcmgt(PRegS(cmpDstIdx),
                                        PReg(cmpMaskIdx) / Xbyak_aarch64::T_z,
                                        ZRegS(cmpSrcIdx), ZRegS(cmpSrc2Idx));
                                break; //GT_OQ
                            case 3: //UNORD_Q
                            case 7: //ORD_Q
                            case 11: //FALSE_OQ
                            case 15: //TRUE_UQ
                            case 19: //UNORD_S
                            case 23: //ORD_S
                            case 27: //FALSE_OS
                            case 31: //TRUE_US
                            default: assert(!"unreachable"); break;
                        }
                    } else {
                        assert(!"unreachable");
                    }
                    if (vlen == 64) {
                        sel(ZRegS(IDX(accvr)), k_store_mask / T_m,
                                ZRegS(IDX(inpvr)), ZRegS(IDX(accvr)));
                    } else if (vlen == 32) {
                        sel(ZRegS(IDX(accvr)), k_store_mask / T_m,
                                ZRegS(IDX(inpvr)), ZRegS(IDX(accvr)));
                        xa_->mov(ZReg(IDX(accvr)).s, P_MSB_256 / T_m, 0);
                    } else if (vlen == 16) {
                        sel(ZRegS(IDX(accvr)), k_store_mask / T_m,
                                ZRegS(IDX(inpvr)), ZRegS(IDX(accvr)));
                        xa_->mov(ZReg(IDX(accvr)).s, P_MSB_384 / T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
                    if (jpp.is_training) {
                        if (vlen == 64) {
                            sel(ZRegS(IDX(indvr)), k_store_mask / T_m,
                                    ZRegS(IDX(vmm_k_offset)),
                                    ZRegS(IDX(indvr)));
                        } else if (vlen == 32) {
                            sel(ZRegS(IDX(indvr)), k_store_mask / T_m,
                                    ZRegS(IDX(vmm_k_offset)),
                                    ZRegS(IDX(indvr)));
                            xa_->mov(ZReg(IDX(indvr)).s, P_MSB_256 / T_m, 0);
                        } else if (vlen == 16) {
                            sel(ZRegS(IDX(indvr)), k_store_mask / T_m,
                                    ZRegS(IDX(vmm_k_offset)),
                                    ZRegS(IDX(indvr)));
                            xa_->mov(ZReg(IDX(indvr)).s, P_MSB_384 / T_m, 0);
                        } else {
                            assert(!"unreachable");
                        }
                    }
                }
            }
            if (jpp.is_training) {
                if (with_c_tail_proccessing && isa == sve_128) {
                    push_vmm_val(vmm_c_tail_mask.getIdx());
                    put_one_in_vmm();
                }

                if (isa == sve_128 && !mayiuse(sve_256)) {
                    assert(!"unsupported");
                } else {
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        xa_->add(ZReg(IDX(vmm_k_offset)).s,
                                ZReg(IDX(vmm_k_offset)).s,
                                ZReg(IDX(vmm_one)).s);
                    } else if (vlen == 32) {
                        xa_->add(ZReg(IDX(vmm_k_offset)).s,
                                ZReg(IDX(vmm_k_offset)).s,
                                ZReg(IDX(vmm_one)).s);
                    } else if (vlen == 16) {
                        xa_->add(VReg(IDX(vmm_k_offset)).s4,
                                VReg(IDX(vmm_k_offset)).s4,
                                VReg(IDX(vmm_one)).s4);
                        xa_->mov(ZReg(IDX(vmm_k_offset)).s, P_MSB_256 / T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
                }

                if (with_c_tail_proccessing && isa == sve_128)
                    pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
        }

        add_imm(aux_reg_input, aux_reg_input, (jpp.dt_size * iw * c_off),
                x_tmp_0);

        adds(kj, kj, 1);

        xa_->cmp(kj, reg_kh);

        b(LT, kh_label);
    }

    if (jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d,
                (jpp.dt_size * jpp.ih * iw * c_off), x_tmp_0);
        if (jpp.is_training) {
            add_imm(x_tmp_addr, reg_param, GET_OFF(kd_padding_shift), x_tmp_0);
            ldr(tmp_gpr, Xbyak_aarch64::ptr(x_tmp_addr));

            ptrue(p_tmp0.d, VL2);
            xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
            ptrue(p_tmp0.d, VL1);
            xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, tmp_gpr);

            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                xa_->mov(ZRegS(IDX(vmm_tmp)), P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                dup(VReg4S(IDX(vmm_tmp)), VReg4S(IDX(xmm_tmp))[0]);
                xa_->mov(ZRegS(IDX(vmm_tmp)), P_MSB_384 / T_m, 0);
            } else {
                assert(!"unreachable");
            }
            if (isa == sve_128 && !mayiuse(sve_256)) {
                assert(!"unsupported");
            } else {
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    xa_->add(ZReg(IDX(vmm_k_offset)).s,
                            ZReg(IDX(vmm_k_offset)).s, ZReg(IDX(vmm_tmp)).s);
                } else if (vlen == 32) {
                    xa_->add(ZReg(IDX(vmm_k_offset)).s,
                            ZReg(IDX(vmm_k_offset)).s, ZReg(IDX(vmm_tmp)).s);
                } else if (vlen == 16) {
                    xa_->add(VReg(IDX(vmm_k_offset)).s4,
                            VReg(IDX(vmm_k_offset)).s4, VReg(IDX(vmm_tmp)).s4);
                    xa_->mov(ZReg(IDX(vmm_k_offset)).s, P_MSB_384 / T_m, 0);
                } else {
                    assert(!"unreachable");
                }
            }
        }

        subs(ki, ki, 1);

        xa_->mov_imm(x_tmp_0, 0);
        xa_->cmp(ki, x_tmp_0);

        b(GT, kd_label);

        ldr(reg_output, post_ptr(X_TRANSLATOR_STACK, 8));

        ldr(reg_input, post_ptr(X_TRANSLATOR_STACK, 8));
    }

    if (with_c_tail_proccessing && jpp.is_c_padded && isa == asimd) {
        xa_->mov_imm(tmp_gpr, 0);
    }

    if (jpp.with_postops) {
        apply_postops(ur_bc, ur_w, c_block, is_tail_processing);
    }

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto accr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        const auto output_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        if (jpp.is_bf16) {
            auto acczr = zreg(accr_i);
            auto accyr = yreg(accr_i);
            if (!isa_has_bf16(jpp.isa)) {
                bf16_emu_->vcvtneps2bf16(accyr, acczr);
            } else {
                assert(!"unsupported");
            }
        }
        store(reg_idx(accr_i), xreg_output, output_offset,
                is_tail_processing(bci));

        if (jpp.is_training) {
            const size_t step_index = (jj * c_off + bci * c_block)
                    * types::data_type_size(jpp.ind_dt);

            const auto indr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
            auto vr = vreg(indr_i);
            if (jpp.ind_dt == data_type::u8) {
                auto xr = xreg(indr_i);
                if (isa == asimd) {
                    assert(!"unsupported");
                } else if (isa == sve_128 || isa == sve_256) {
                    auto yr = yreg(indr_i);
                    if (is_tail_processing(bci) && !jpp.is_c_padded) {
                        assert(!"unsupported");
                    } else {
                        if (is_tail_processing(bci)) {
                            assert(jpp.is_c_padded);
                            if (isa == sve_128) {
                                xa_->and_(VReg16B(IDX(yr)), VReg16B(IDX(yr)),
                                        VReg16B(IDX(vmm_c_tail_mask)));
                            } else if (isa == sve_256) {
                                xa_->and_(ZRegD(IDX(yr)), ZRegD(IDX(yr)),
                                        ZRegD(IDX(vmm_c_tail_mask)));
                                xa_->mov(ZRegS(IDX(yr)), P_MSB_256 / T_m, 0);
                            } else {
                                assert(!"unsupported");
                            }
                        }
                        if (jj == 0) {
                            bic(VReg(IDX(xmm_tmp)).s4, 0);
                            fmov(SReg(IDX(xmm_tmp)), WReg(IDX(reg_shuf_mask)));

                            int vlen = cpu_isa_traits<isa>::vlen;
                            if (vlen == 64) {
                                dup(ZRegS(IDX(vmm_tmp)),
                                        ZRegS(IDX(xmm_tmp))[0]);
                            } else if (vlen == 32) {
                                dup(ZRegS(IDX(vmm_tmp)),
                                        ZRegS(IDX(xmm_tmp))[0]);
                                xa_->mov(ZRegS(IDX(vmm_tmp)), P_MSB_256 / T_m,
                                        0);
                            } else if (vlen == 16) {
                                dup(VReg4S(IDX(vmm_tmp)),
                                        VReg4S(IDX(xmm_tmp))[0]);
                                xa_->mov(ZRegS(IDX(vmm_tmp)), P_MSB_384 / T_m,
                                        0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                        if (mayiuse(sve_256)) {
                            xa_->mov(z_tmp0.b, 15);
                            xa_->and_(z_tmp0.b, p_512, ZRegB(IDX(vmm_tmp)));
                            for (int i = 0; i < 16; i++) {
                                cmpeq(p_tmp1.b, p_512, z_tmp0.b, i);
                                dup(z_tmp1.b, ZRegB(IDX(yr))[i]);
                                xa_->mov(z_tmp2.b, p_tmp1 / T_m, z_tmp1.b);
                            }
                            for (int i = 16; i < 32; i++) {
                                cmpeq(p_tmp1.b, p_512, z_tmp0.b, i - 16);
                                xa_->and_(
                                        p_tmp1.b, p_512, p_tmp1.b, P_MSB_384.b);
                                dup(z_tmp1.b, ZRegB(IDX(yr))[i]);
                                xa_->mov(z_tmp2.b, p_tmp1 / T_m, z_tmp1.b);
                            }
                            cmplt(p_tmp1.b, p_512, ZRegB(IDX(vmm_tmp)), 0);
                            xa_->mov(ZRegD(IDX(yr)), z_tmp2.d);
                            orr(p_tmp0.b, p_512, p_tmp1.b, P_MSB_256.b);
                            xa_->mov(ZRegB(IDX(yr)), p_tmp0 / T_m, 0);

                            add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                            str(SReg(IDX(xr)), Xbyak_aarch64::ptr(x_tmp_addr));

                            ptrue(p_tmp0.d, VL2);
                            xa_->mov(z_tmp0.q, ZReg(IDX(yr)).q[1]);
                            xa_->mov(z_tmp1.q, QReg(IDX(yr)));
                            sel(ZRegD(IDX(yr)), p_tmp0, z_tmp0.d, z_tmp1.d);
                            xa_->mov(ZRegD(IDX(yr)), P_MSB_256 / T_m, 0);

                            add_imm(x_tmp_addr, reg_index,
                                    step_index + (jpp.c_block / 2), x_tmp_0);
                            str(SReg(IDX(xr)), Xbyak_aarch64::ptr(x_tmp_addr));
                        } else {
                            VReg t(vmm_mask.getIdx());

                            xa_->mov(VReg(IDX(t)).b16, VReg(IDX(yr)).b16);

                            xa_->mov(z_tmp0.b, 15);
                            xa_->and_(z_tmp0.b, p_512, ZRegB(IDX(xmm_tmp)));
                            for (int i = 0; i < 16; i++) {
                                cmpeq(p_tmp1.b, p_512, z_tmp0.b, i);
                                dup(z_tmp1.b, ZRegB(IDX(t))[i]);
                                xa_->mov(z_tmp2.b, p_tmp1 / T_m, z_tmp1.b);
                            }
                            for (int i = 16; i < 32; i++) {
                                cmpeq(p_tmp1.b, p_512, z_tmp0.b, i - 16);
                                xa_->and_(
                                        p_tmp1.b, p_512, p_tmp1.b, P_MSB_384.b);
                                dup(z_tmp1.b, ZRegB(IDX(t))[i]);
                                xa_->mov(z_tmp2.b, p_tmp1 / T_m, z_tmp1.b);
                            }
                            cmplt(p_tmp1.b, p_512, ZRegB(IDX(xmm_tmp)), 0);
                            xa_->mov(ZRegD(IDX(t)), z_tmp2.d);
                            orr(p_tmp0.b, p_512, p_tmp1.b, P_MSB_256.b);
                            xa_->mov(ZRegB(IDX(t)), p_tmp0 / T_m, 0);

                            add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                            add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                            str(SReg(IDX(t)), Xbyak_aarch64::ptr(x_tmp_addr));

                            xa_->mov(z_tmp0.d, ZRegD(IDX(yr)));
                            ext(z_tmp0.b, ZRegB(IDX(yr)), 16);
                            xa_->mov(VReg(IDX(t)).b16, VReg(IDX(z_tmp0)).b16);

                            add_imm(x_tmp_addr, reg_index,
                                    step_index + (jpp.c_block / 2), x_tmp_0);
                            str(SReg(IDX(t)), Xbyak_aarch64::ptr(x_tmp_addr));
                        }
                    }
                } else {
                    if (is_tail_processing(bci)) {
                        if (jpp.is_c_padded) {
                            add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                            int vlen = cpu_isa_traits<isa>::vlen;
                            if (vlen == 64) {
                                xa_->mov(z_tmp0.d, ZRegD(IDX(vr)));
                                umin(z_tmp0.s, 255);
                                st1b(z_tmp0.s, p_512,
                                        Xbyak_aarch64::ptr(x_tmp_addr));
                            } else if (vlen == 32) {
                                assert(!"unsupported");
                            } else if (vlen == 16) {
                                assert(!"unsupported");
                            } else {
                                assert(!"unsupported");
                            }
                        } else {
                            add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                            int vlen = cpu_isa_traits<isa>::vlen;
                            if (vlen == 64) {
                                xa_->mov(z_tmp0.d, ZRegD(IDX(vr)));
                                umin(z_tmp0.s, 255);
                                pfalse(p9.b);
                                zip1(p1.b, k_c_tail_mask.b, p9.b);
                                zip1(p1.h, p1.h, p9.h);
                                st1b(z_tmp0.s, p1,
                                        Xbyak_aarch64::ptr(x_tmp_addr));
                            } else if (vlen == 32) {
                                assert(!"unsupported");
                            } else if (vlen == 16) {
                                assert(!"unsupported");
                            } else {
                                assert(!"unsupported");
                            }
                        }
                    } else {
                        add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                        int vlen = cpu_isa_traits<isa>::vlen;
                        if (vlen == 64) {
                            xa_->mov(z_tmp0.d, ZRegD(IDX(vr)));
                            umin(z_tmp0.s, 255);
                            st1b(z_tmp0.s, p_512,
                                    Xbyak_aarch64::ptr(x_tmp_addr));
                        } else if (vlen == 32) {
                            assert(!"unsupported");
                        } else if (vlen == 16) {
                            assert(!"unsupported");
                        } else {
                            assert(!"unsupported");
                        }
                    }
                }
            } else {
                store(vr.getIdx(), xreg_index, step_index,
                        is_tail_processing(bci));
            }
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::max_step_bwd(int ur_w, int ur_bc,
        int pad_l, int pad_r, bool with_c_tail_proccessing) {

    int iw = jpp.iw;
    int kw = jpp.kw;
    int stride_w = jpp.stride_w;
    int c_block = jpp.c_block;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    const auto is_tail_processing = [&](int bc) {
        if (isa == asimd) {
            assert(!"unsupported");
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        const auto outr_i = reg_ind(0, bci, jj, ur_bc, ur_w);
        auto out_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        load(reg_idx(outr_i), xreg_output, out_offset, is_tail_processing(bci));
        const size_t step_index = (jj * c_off + bci * c_block)
                * types::data_type_size(jpp.ind_dt);

        const auto indr_i = reg_ind(1, bci, jj, ur_bc, ur_w);
        auto indvr = vreg(indr_i);
        if (jpp.ind_dt == data_type::u8) {
            auto indxr = xreg(indr_i);
            if (isa == asimd) {
                assert(!"unsupported");
            } else if (isa == sve_128 || isa == sve_256) {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    assert(!"unsupported");
                } else {
                    add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                    ldr(DReg(IDX(indxr)), Xbyak_aarch64::ptr(x_tmp_addr));
                }
                if (!mayiuse(sve_256)) {
                    assert(!"unsupported");
                } else {
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        zip1(z_tmp0.b, ZRegB(IDX(indxr)), ZRegB(IDX(indxr)));
                        zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        uxtb(ZRegS(IDX(indvr)), p_512 / T_m, z_tmp0.s);
                    } else if (vlen == 32) {
                        zip1(z_tmp0.b, ZRegB(IDX(indxr)), ZRegB(IDX(indxr)));
                        zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        uxtb(ZRegS(IDX(indvr)), p_512 / T_m, z_tmp0.s);
                        xa_->mov(ZRegS(IDX(indvr)), P_MSB_256 / T_m, 0);
                    } else if (vlen == 16) {
                        zip1(z_tmp0.b, ZRegB(IDX(indxr)), ZRegB(IDX(indxr)));
                        zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        uxtb(ZRegS(IDX(indvr)), p_512 / T_m, z_tmp0.s);
                        xa_->mov(ZRegS(IDX(indvr)), P_MSB_384 / T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
                }
            } else {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {

                    ZReg z_indvr(IDX(indvr));

                    add_imm(X_DEFAULT_ADDR, reg_index, step_index, X_TMP_0);
                    ld1b(z_indvr.b, k_c_tail_mask / Xbyak_aarch64::T_z,
                            Xbyak_aarch64::ptr(X_DEFAULT_ADDR));
                    zip1(z_indvr.b, z_indvr.b, z_tmp0.b);
                    zip1(z_indvr.h, z_indvr.h, z_tmp0.h);
                    pfalse(p9.b);
                    zip1(p1.b, k_c_tail_mask.b, p9.b);
                    zip1(p1.h, p1.h, p9.h);
                    uxtb(ZRegS(IDX(indvr)), p1 / T_m, z_indvr.s);
                } else {
                    add_imm(x_tmp_addr, reg_index, step_index, x_tmp_0);
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        ldr(QReg(IDX(z_tmp0)), Xbyak_aarch64::ptr(x_tmp_addr));
                        zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                        zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        uxtb(ZReg(IDX(indvr)).s, p_512 / T_m, z_tmp0.s);
                    } else if (vlen == 32) {
                        ldr(QReg(IDX(z_tmp0)), Xbyak_aarch64::ptr(x_tmp_addr));
                        zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                        zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        uxtb(ZReg(IDX(indvr)).s, p_512 / T_m, z_tmp0.s);
                        xa_->mov(ZReg(IDX(indvr)).s, P_MSB_256 / T_m, 0);
                    } else if (vlen == 16) {
                        ldr(QReg(IDX(z_tmp0)), Xbyak_aarch64::ptr(x_tmp_addr));
                        zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                        zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        uxtb(ZReg(IDX(indvr)).s, p_512 / T_m, z_tmp0.s);
                        xa_->mov(ZReg(IDX(indvr)).s, P_MSB_384 / T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
                }
            }
        } else {
            load(indvr.getIdx(), xreg_index, step_index,
                    is_tail_processing(bci));
        }
    }
    ptrue(p_tmp0.d, VL2);
    xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
    ptrue(p_tmp0.d, VL1);
    xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, reg_k_shift);
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        dup(ZRegS(IDX(vmm_k_offset)), ZRegS(IDX(xmm_tmp))[0]);
    } else if (vlen == 32) {
        dup(ZRegS(IDX(vmm_k_offset)), ZRegS(IDX(xmm_tmp))[0]);
        xa_->mov(ZRegS(IDX(vmm_k_offset)), P_MSB_256 / T_m, 0);
    } else if (vlen == 16) {
        dup(VReg4S(IDX(vmm_k_offset)), VReg4S(IDX(xmm_tmp))[0]);
        xa_->mov(ZRegS(IDX(vmm_k_offset)), P_MSB_384 / T_m, 0);
    } else {
        assert(!"unreachable");
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
        str(reg_input, pre_ptr(X_TRANSLATOR_STACK, -8));
        str(reg_output, pre_ptr(X_TRANSLATOR_STACK, -8));
        if (isa == asimd) { assert(!"unsupported"); }
        xa_->mov(aux_reg_input_d, reg_input);

        add_imm(x_tmp_addr, reg_param, GET_OFF(kd_padding), x_tmp_0);
        ldr(ki, Xbyak_aarch64::ptr(x_tmp_addr));

        add_imm(x_tmp_addr, reg_param, GET_OFF(kd_padding_shift), x_tmp_0);
        ldr(reg_kd_pad_shift, Xbyak_aarch64::ptr(x_tmp_addr));
        L(kd_label);

        xa_->mov(aux_reg_input, aux_reg_input_d);
    } else {

        xa_->mov(aux_reg_input, reg_input);
    }

    eor(kj, kj, kj);

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                const auto outvr = vreg(reg_ind(0, bci, jj, ur_bc, ur_w));
                const auto indvr = vreg(reg_ind(1, bci, jj, ur_bc, ur_w));
                const auto inpr_i = reg_ind(2, bci, jj, ur_bc, ur_w);
                const auto inpvr = vreg(inpr_i);
                const auto cvtvr = vreg(reg_ind(3, bci, jj, ur_bc, ur_w));
                int aux_inp_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_inp_offset >= iw * c_off) continue;
                int inp_offset = jpp.dt_size * aux_inp_offset;
                load(reg_idx(inpr_i), aux_xreg_input, inp_offset,
                        is_tail_processing(bci));
                if (isa == asimd) {
                    assert(!"unsupported");
                } else if (isa == sve_128 || isa == sve_256) {
                    assert(!"unsupported");
                } else {
                    auto indzr = zreg(inpr_i);
                    auto indyr = yreg(inpr_i);

                    cmpeq(k_store_mask.s, p_lsb / Xbyak_aarch64::T_z,
                            ZRegS(IDX(indvr)), ZRegS(IDX(vmm_k_offset)));

                    xa_->not_(p_tmp0.b, P_ALL_ONE.b, k_store_mask.b);
                    int vlen = cpu_isa_traits<isa>::vlen;

                    if (vlen == 64) {
                        xa_->mov(ZRegD(IDX(vmm_tmp)), ZRegD(IDX(outvr)));
                        xa_->mov(ZReg(IDX(vmm_tmp)).s, p_tmp0 / T_m, 0);
                        xa_->fadd(ZReg(IDX(inpvr)).s, ZReg(IDX(inpvr)).s,
                                ZReg(IDX(vmm_tmp)).s);
                    } else if (vlen == 32) {
                        xa_->mov(ZRegD(IDX(vmm_tmp)), ZRegD(IDX(outvr)));
                        xa_->mov(ZReg(IDX(vmm_tmp)).s, p_tmp0 / T_m, 0);
                        xa_->fadd(ZReg(IDX(inpvr)).s, ZReg(IDX(inpvr)).s,
                                ZReg(IDX(vmm_tmp)).s);
                        xa_->mov(ZReg(IDX(inpvr)).s, P_MSB_256 / T_m, 0);
                    } else if (vlen == 16) {
                        xa_->mov(VReg16B(IDX(vmm_tmp)), VReg16B(IDX(outvr)));
                        xa_->mov(ZReg(IDX(vmm_tmp)).s, p_tmp0 / T_m, 0);
                        xa_->fadd(VReg(IDX(inpvr)).s4, VReg(IDX(inpvr)).s4,
                                VReg(IDX(vmm_tmp)).s4);
                    } else {
                        assert(!"unreachable");
                    }

                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa)) {
                            bf16_emu_->vcvtneps2bf16(indyr, indzr);
                        } else {
                            assert(!"unsupported");
                        }
                    }
                    store(inpvr.getIdx(), aux_xreg_input, inp_offset,
                            is_tail_processing(bci));
                }
            }

            if (with_c_tail_proccessing && (isa == sve_128 || isa == sve_256)) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                put_one_in_vmm();
            }

            if (isa == sve_128 && !mayiuse(sve_256)) {
                assert(!"unsupported");
            } else {
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    xa_->add(ZReg(IDX(vmm_k_offset)).s,
                            ZReg(IDX(vmm_k_offset)).s, ZReg(IDX(vmm_one)).s);
                } else if (vlen == 32) {
                    xa_->add(ZReg(IDX(vmm_k_offset)).s,
                            ZReg(IDX(vmm_k_offset)).s, ZReg(IDX(vmm_one)).s);
                } else if (vlen == 16) {
                    xa_->add(VReg(IDX(vmm_k_offset)).s4,
                            VReg(IDX(vmm_k_offset)).s4, VReg(IDX(vmm_one)).s4);
                    xa_->mov(ZReg(IDX(vmm_k_offset)).s, P_MSB_384 / T_m, 0);
                } else {
                    assert(!"unreachable");
                }
            }

            if (with_c_tail_proccessing && (isa == sve_128 || isa == sve_256))
                pop_vmm_val(vmm_c_tail_mask.getIdx());
        }
        add_imm(aux_reg_input, aux_reg_input, (jpp.dt_size * iw * c_off),
                x_tmp_0);

        adds(kj, kj, 1);

        xa_->cmp(kj, reg_kh);

        b(LT, kh_label);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        add_imm(aux_reg_input_d, aux_reg_input_d,
                (jpp.dt_size * jpp.ih * iw * c_off), x_tmp_0);

        xa_->mov(tmp_gpr, reg_kd_pad_shift);

        ptrue(p_tmp0.d, VL2);
        xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
        ptrue(p_tmp0.d, VL1);
        xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, tmp_gpr);

        const int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) {
            dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
        } else if (vlen == 32) {
            dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
            xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_256 / T_m, 0);
        } else if (vlen == 16) {
            dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
            xa_->mov(ZReg(IDX(vmm_tmp)).s, P_MSB_384 / T_m, 0);
        } else {
            assert(!"unreachable");
        }
        if (isa == sve_128 && !mayiuse(sve_256)) {
            assert(!"unsupported");
        } else {
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                xa_->add(ZReg(IDX(vmm_k_offset)).s, ZReg(IDX(vmm_k_offset)).s,
                        ZReg(IDX(vmm_tmp)).s);
            } else if (vlen == 32) {
                xa_->add(ZReg(IDX(vmm_k_offset)).s, ZReg(IDX(vmm_k_offset)).s,
                        ZReg(IDX(vmm_tmp)).s);
            } else if (vlen == 16) {
                xa_->add(VReg(IDX(vmm_k_offset)).s4, VReg(IDX(vmm_k_offset)).s4,
                        VReg(IDX(vmm_tmp)).s4);
                xa_->mov(ZReg(IDX(vmm_k_offset)).s, P_MSB_384 / T_m, 0);
            } else {
                assert(!"unreachable");
            }
        }

        subs(ki, ki, 1);

        xa_->mov_imm(x_tmp_0, 0);
        xa_->cmp(ki, x_tmp_0);

        b(GT, kd_label);
        if (isa == asimd) { assert(!"unsupported"); }
        ldr(reg_output, post_ptr(X_TRANSLATOR_STACK, 8));

        ldr(reg_input, post_ptr(X_TRANSLATOR_STACK, 8));
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::zero_diff_src(
        int ur_bc, bool with_c_tail_proccessing) {
    const int c_off = (jpp.tag_kind == jit_memory_tag_kind_t::nspc)
            ? jpp.c
            : jpp.c_block;

    Label l_skip, l_ih_loop, l_id_loop;

    auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    add_imm(x_tmp_addr, reg_param, GET_OFF(zero_id), x_tmp_0);
    ldr(reg_zero_id, Xbyak_aarch64::ptr(x_tmp_addr));

    xa_->mov_imm(x_tmp_0, 0);
    xa_->cmp(reg_zero_id, x_tmp_0);

    b(EQ, l_skip);

    add_imm(x_tmp_addr, reg_param, GET_OFF(zero_ih), x_tmp_0);
    ldr(reg_zero_ih, Xbyak_aarch64::ptr(x_tmp_addr));

    xa_->mov_imm(x_tmp_0, 0);
    xa_->cmp(reg_zero_ih, x_tmp_0);

    b(EQ, l_skip);

    add_imm(x_tmp_addr, reg_param, GET_OFF(zero_ptr), x_tmp_0);
    ldr(reg_zero_ptr, Xbyak_aarch64::ptr(x_tmp_addr));

    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        using TReg = typename Xbyak_aarch64::ZReg;
    } else if (vlen == 32) {
        using TReg = typename Xbyak_aarch64::ZReg;
    } else if (vlen == 16) {
        using TReg = typename Xbyak_aarch64::VReg;
    } else {
        assert(!"unreachable");
    }
    TReg vzero = vmm_tmp;

    if (vlen == 64) {
        eor(ZReg(IDX(vzero)).d, ZReg(IDX(vzero)).d, ZReg(IDX(vzero)).d);
    } else if (vlen == 32) {
        eor(ZRegD(IDX(vzero)), ZRegD(IDX(vzero)), ZRegD(IDX(vzero)));
        xa_->mov(ZRegS(IDX(vzero)), P_MSB_256 / T_m, 0);
    } else if (vlen == 16) {
        eor(VReg16B(IDX(vzero)), VReg16B(IDX(vzero)), VReg16B(IDX(vzero)));
    } else {
        assert(!"unreachable");
    }

    const int width_size = jpp.iw * c_off * jpp.dt_size;

    auto aux_reg_zero_ptr = tmp_gpr;

    L(l_id_loop);
    {
        xa_->mov(aux_reg_zero_ptr, reg_zero_ptr);

        xa_->mov(aux_reg_zero_ih, reg_zero_ih);
        L(l_ih_loop);
        {
            const int step = c_off * jpp.dt_size;

            // TODO: maybe a big code generated here
            for_(int i = 0; i < width_size; i += step)
            for (int bci = 0; bci < ur_bc; bci++) {
                const int offs = i + bci * jpp.c_block * jpp.dt_size;
                if (isa == asimd) {
                    assert(!"unsupported");
                } else {
                    store(vzero.getIdx(), xreg_zero_ptr, offs,
                            is_tail_processing(bci));
                }
            }
            add_imm(reg_zero_ptr, reg_zero_ptr, width_size, x_tmp_0);

            subs(aux_reg_zero_ih, aux_reg_zero_ih, 1);

            b(NE, l_ih_loop);
        }
        xa_->mov(reg_zero_ptr, aux_reg_zero_ptr);

        add_imm(reg_zero_ptr, reg_zero_ptr, (width_size * jpp.ih), x_tmp_0);

        subs(reg_zero_id, reg_zero_id, 1);

        b(NE, l_id_loop);
    }

    L(l_skip);
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::generate() {

    this->preamble();

    Label idx_table;

    int ow = jpp.ow;
    int iw = jpp.iw;
    int kw = jpp.kw;
    int kh = jpp.kh;
    int c_block = jpp.c_block;
    int stride_w = jpp.stride_w;
    int l_pad = jpp.l_pad;
    const int c_off
            = (jpp.tag_kind == jit_memory_tag_kind_t::nspc) ? jpp.c : c_block;

    int vlen = cpu_isa_traits<isa>::vlen;

#if defined(_WIN32)
    // Always mimic the Unix ABI (see the note about maskmovdqu in the header
    // file).
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif
    ptrue(p_512.b);
    ptrue(p_256.b, VL32);
    ptrue(p_128.b, VL16);
    if (cpu_isa_traits<isa>::vlen == 32) {
        p_lsb = p_256;
    } else if (cpu_isa_traits<isa>::vlen == 16) {
        p_lsb = p_128;
    }

    if (!isa_has_bf16(jpp.isa) && jpp.is_bf16) bf16_emu_->init_vcvtneps2bf16();

    add_imm(x_tmp_addr, reg_param, GET_OFF(src), x_tmp_0);
    ldr(reg_input, Xbyak_aarch64::ptr(x_tmp_addr));

    add_imm(x_tmp_addr, reg_param, GET_OFF(dst), x_tmp_0);
    ldr(reg_output, Xbyak_aarch64::ptr(x_tmp_addr));
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
        add_imm(x_tmp_addr, reg_param, GET_OFF(indices), x_tmp_0);
        ldr(reg_index, Xbyak_aarch64::ptr(x_tmp_addr));
    }

    add_imm(x_tmp_addr, reg_param, GET_OFF(kh_padding), x_tmp_0);
    ldr(reg_kh, Xbyak_aarch64::ptr(x_tmp_addr));

    add_imm(x_tmp_addr, reg_param, GET_OFF(kh_padding_shift), x_tmp_0);
    ldr(reg_k_shift, Xbyak_aarch64::ptr(x_tmp_addr));

    add_imm(x_tmp_addr, reg_param, GET_OFF(ker_area_h), x_tmp_0);
    ldr(reg_ker_area_h, Xbyak_aarch64::ptr(x_tmp_addr));

    add_imm(x_tmp_addr, reg_param, GET_OFF(ur_bc), x_tmp_0);
    ldr(reg_nbc, Xbyak_aarch64::ptr(x_tmp_addr));

    if (jpp.is_bf16) { assert(!"unsupported"); }

    int r_pad
            = nstl::max(0, calculate_end_padding(l_pad, ow, iw, stride_w, kw));

    auto process_oi = [&](int ur_w, int ur_bc, int lpad, int rpad,
                              bool with_c_tail_proccessing,
                              bool inc_reg = true) {
        step(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);

        if (isa == asimd) {
            if (with_c_tail_proccessing && jpp.c_tail <= (jpp.c_block / 2)) {
                /*
                // In nspc format in case of c tail processing if c tail is
                // equal or lower than 4 we don't have to process
                // last high half block, because it doesn't exist
                if (!jpp.is_c_padded) ur_bc -= 1;
	      */
                /*
                 * In case of c_tail_processing if c_tail is equal or lower than 4
                 * applying postops never make sense. In case of blocked format it
                 * can cause overwriting zero padding or segfault because the element
                 * corresponding to the piece with padded zeros doesn't exist in binary
                 * postops arg1 tensor (nchw format) in per_oc bcast strategy.
                 */
                assert(!"unsupported");
            }
        }

        if (!inc_reg) return;

        auto dt_size = jpp.dt_size;
        auto shift = (isa == asimd) ? vlen : 0;
        add_imm(reg_input, reg_input,
                (dt_size * (ur_w * stride_w - lpad) * c_off - shift), x_tmp_0);

        add_imm(reg_output, reg_output, (dt_size * ur_w * c_off - shift),
                x_tmp_0);
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            auto ishift = (isa == asimd) ? jpp.c_block / 2 : 0;
            auto ind_dt_size = types::data_type_size(jpp.ind_dt);
            add_imm(reg_index, reg_index,
                    ((ur_w * c_off - ishift) * ind_dt_size), x_tmp_0);
        }
    };

    auto perform_ker = [&](int ur_bc, bool with_c_tail_processing) {
        prev_kw = 0; // re-initialize this value for avg steps

        if (jpp.is_backward && jpp.simple_alg)
            zero_diff_src(ur_bc, with_c_tail_processing);

        if (jpp.alg == pooling_avg_exclude_padding
                && (!with_c_tail_processing
                        || (isa != sve_128 && isa != sve_256))) {
            // vmm_ker_area_h and vmm_c_tail_mask are stored in one register
            // so when vmm_c_tail_mask is used we need to load vmm_ker_area_h
            // exactly where this information is needed with the
            // vmm_c_tail_mask information being saved first
            uni_broadcast_reg_val(
                    reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
        }

        if (jpp.alg == pooling_avg_include_padding) {
            xa_->mov_imm(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));

            ptrue(p_tmp0.d, VL2);
            xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, 0);
            ptrue(p_tmp0.d, VL1);
            xa_->mov(ZRegD(IDX(xmm_tmp)), p_tmp0 / T_m, tmp_gpr);

            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                dup(ZRegS(IDX(vmm_tmp)), ZRegS(IDX(xmm_tmp))[0]);
                xa_->mov(ZRegS(IDX(vmm_tmp)), P_MSB_256 / T_m, 0);
            } else if (vlen == 16) {
                dup(VReg4S(IDX(vmm_tmp)), VReg4S(IDX(xmm_tmp))[0]);
                xa_->mov(ZRegS(IDX(vmm_tmp)), P_MSB_384 / T_m, 0);
            } else {
                assert(!"unreachable");
            }
        }

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            if (!with_c_tail_processing || (isa != sve_128 && isa != sve_256)) {
                // The same situation as above(vmm_ker_area_h).
                put_one_in_vmm();
            }

            if (isa == sve_128 || isa == sve_256) { assert(!"unsupported"); }
        }

        auto ur_w = nstl::min(jpp.ow, jpp.ur / jpp.ur_bc);
        auto ur_w_tail = jpp.ow % ur_w;

        int n_oi = ow / ur_w;

        int r_pad1
                = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w, kw);
        if (r_pad1 > 0) n_oi--;

        if (l_pad > 0) {
            n_oi--;
            if (n_oi < 0 && r_pad1 > 0)
                process_oi(ur_w, ur_bc, l_pad, r_pad1, with_c_tail_processing);
            else
                process_oi(ur_w, ur_bc, l_pad, 0, with_c_tail_processing);
        }

        eor(oi_iter, oi_iter, oi_iter);
        if (n_oi > 0) {
            Label ow_loop;
            L(ow_loop);
            {
                process_oi(ur_w, ur_bc, 0, 0, with_c_tail_processing);

                adds(oi_iter, oi_iter, 1);

                xa_->mov_imm(x_tmp_0, n_oi);
                xa_->cmp(oi_iter, x_tmp_0);

                b(LT, ow_loop);
            }
        }

        if (r_pad1 > 0 && n_oi >= 0)
            process_oi(ur_w, ur_bc, 0, r_pad1, with_c_tail_processing);

        if (ur_w_tail != 0)
            process_oi(
                    ur_w_tail, ur_bc, 0, r_pad, with_c_tail_processing, false);
    };
    Label ur_bc_tail_label, c_tail_processing_label, finish_label;

    if (jpp.ur_bc_tail > 0) {
        xa_->mov_imm(x_tmp_0, jpp.ur_bc);
        xa_->cmp(reg_nbc, x_tmp_0);

        b(NE, ur_bc_tail_label);
    } else if (jpp.c_tail != 0) {
        // ur_bc contains number of channel blocks to processing
        // b_c contains number of channel blocks already processed
        // If reg_nbc + tmp_gpr == jpp.nb_c then this is
        // information that probably channel tail processing will be needed.
        /* get mem address */
        add_imm(x_tmp_addr, reg_param, GET_OFF(b_c), x_tmp_0);
        ldr(tmp_gpr, Xbyak_aarch64::ptr(x_tmp_addr));

        xa_->add(tmp_gpr, tmp_gpr, reg_nbc);

        xa_->mov_imm(x_tmp_0, jpp.nb_c);
        xa_->cmp(tmp_gpr, x_tmp_0);

        b(Xbyak_aarch64::EQ, c_tail_processing_label);
    }

    perform_ker(jpp.ur_bc, false);

    if (jpp.ur_bc_tail > 0) {
        xa_->bl(finish_label);

        // If ur_bc_tail exists then we know that this is
        // last set of blocks to process and we need
        // care of c tail processing if number of channels
        // is not divided by number of channels in block
        L(ur_bc_tail_label);

        if (jpp.c_tail != 0) prepare_tail_mask();
        perform_ker(jpp.ur_bc_tail, jpp.c_tail != 0);

        L(finish_label);

    } else if (jpp.c_tail != 0) {
        xa_->bl(finish_label);

        L(c_tail_processing_label);

        prepare_tail_mask();
        perform_ker(jpp.ur_bc, true);

        L(finish_label);
    }

    this->postamble();

    if (jpp.is_bf16) {
        align(64);
        L(idx_table);
        const uint16_t _idx[] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            Xbyak::CodeArray::dw(_idx[i]);

        binCommit();
    }
}

template struct jit_uni_pool_kernel<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
