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

#define CG CodeGeneratorAArch64
#define IDX(a) static_cast<uint32_t>(a.getIdx())
#ifndef DNNL_X64_IMPLEMENTATION
namespace xa = Xbyak::Xbyak_aarch64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak;
using namespace alg_kind;

#define GET_OFF(field) offsetof(jit_pool_call_s, field)

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

    jpp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];

    using namespace format_tag;
    const auto blocked_fmt_tag = utils::one_of(isa, avx512_common, avx512_core)
            ? utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);

    // src_d.data_type() is equal to dst_d.data_type(). This is checked in init
    const bool forward_ncsp_allowed = !jpp.is_backward && jpp.oh == 1;
    const auto ncsp_fmt_tag
            = ((forward_ncsp_allowed || jpp.is_backward) && isa == avx512_core
                      && ndims < 5 && src_d.data_type() == data_type::bf16)
            ? utils::pick(ndims - 3, ncw, nchw)
            : format_tag::undef;

    const auto nspc_fmt_tag = (ndims <= 5)
            ? utils::pick(ndims - 3, nwc, nhwc, ndhwc)
            : format_tag::undef;

    const auto fmt_tag = src_d.matches_one_of_tag(
            blocked_fmt_tag, ncsp_fmt_tag, nspc_fmt_tag);

    if (!dst_d.matches_tag(fmt_tag)) return status::unimplemented;

    if (fmt_tag == ncsp_fmt_tag) {
        // plain layout allowed for BWD_D only now:
        // transform input to blocked f32, call f32 jit, transform result to
        // plain output
        jpp.is_bf16 = false;
        jpp.dt_size = types::data_type_size(data_type::f32);
        jpp.tag_kind = jptg_ncsp;
    } else {
        jpp.is_bf16 = (src_d.data_type() == data_type::bf16
                && dst_d.data_type() == data_type::bf16);
        jpp.dt_size = types::data_type_size(src_d.data_type());
        jpp.tag_kind = (fmt_tag == nspc_fmt_tag) ? jptg_nspc : jptg_blocked;
    }

    jpp.isa = (jpp.is_bf16 && mayiuse(avx512_core_bf16)) ? avx512_core_bf16
                                                         : isa;

    const bool args_ok = true && mayiuse(isa) && (fmt_tag != format_tag::undef)
            && IMPLICATION(jpp.is_bf16, mayiuse(avx512_core))
            && utils::one_of(pd.alg_kind, pooling_max,
                    pooling_avg_include_padding, pooling_avg_exclude_padding);
    if (!args_ok) return status::unimplemented;

    const bool is_avx512 = utils::one_of(isa, avx512_common, avx512_core);

    jpp.ndims = ndims;
    jpp.mb = src_d.dims()[0];
    jpp.c_without_padding = src_d.dims()[1];
    jpp.c_block = is_avx512 ? 16 : 8;
    jpp.c = jpp.tag_kind == jptg_blocked
            ? utils::rnd_up(jpp.c_without_padding, jpp.c_block)
            : jpp.c_without_padding;
    if (jpp.tag_kind == jptg_blocked) assert(src_d.padded_dims()[1] == jpp.c);
    jpp.nb_c = utils::div_up(jpp.c, jpp.c_block);
    jpp.c_tail = jpp.c_without_padding % jpp.c_block;
    jpp.is_c_padded = jpp.tag_kind == jptg_blocked
            && src_d.padded_dims()[1] != jpp.c_without_padding;

    jpp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jpp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jpp.iw = src_d.dims()[ndims - 1];
    jpp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jpp.ow = dst_d.dims()[ndims - 1];

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

    jpp.alg = pd.alg_kind;

    jpp.ind_dt = ppd->workspace_md() ? ppd->workspace_md()->data_type
                                     : data_type::undef;

    jpp.simple_alg = jpp.is_training
            || IMPLICATION(jpp.is_backward, jpp.kd <= jpp.stride_d);

    jpp.ur = 0;
    if (jpp.alg == pooling_max) {
        jpp.ur = is_avx512 ? 16 : 4;

        if (isa == avx && jpp.c_tail > 0)
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
    if (jpp.tag_kind == jptg_nspc) {
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
            const int L2 = platform::get_per_core_cache_size(2)
                    / sizeof(jpp.dt_size);
            int ur_bc = nstl::max(1, L2 / (jpp.kh * jpp.iw * jpp.c_block));
            jpp.ur_bc = nstl::min(jpp.ur_bc, ur_bc);
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
    if (jpp.tag_kind == jptg_ncsp) {
        scratchpad.book(key_pool_src_plain2blocked_cvt,
                jpp.c_block * jpp.id * jpp.ih * jpp.iw * nscr, jpp.dt_size);
        scratchpad.book(key_pool_dst_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr, jpp.dt_size);
        scratchpad.book<uint32_t>(key_pool_ind_plain2blocked_cvt,
                jpp.c_block * jpp.od * jpp.oh * jpp.ow * nscr);
    }

    return status::success;
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::prepare_tail_mask() {
    if (isa >= avx512_common) {
#ifdef DNNL_X64_IMPLEMENTATION
        size_t c_tail_mask = (1ULL << jpp.c_tail) - 1ULL;
        mov(tmp_gpr.cvt32(), c_tail_mask);
        // The kmovw instrucion here can be translated correctly by translator
        kmovw(k_c_tail_mask, tmp_gpr.cvt32());
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        size_t c_tail_mask = jpp.c_tail;
        CG::mov_imm(X_TMP_0, c_tail_mask);
        CG::dup(z_tmp0.s, W_TMP_0);
        CG::index(z_tmp1.s, 0, 1);
        /* PRegS(IDX(k_c_tail_mask)) keeps flags in the context 
	   of 32-bit elements. */
        CG::cmplt(xa::PRegS(IDX(k_c_tail_mask)), p_512 / xa::T_z, z_tmp1.s,
                z_tmp0.s);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    } else if (isa == avx) {
        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};
#ifdef DNNL_X64_IMPLEMENTATION
        mov(tmp_gpr, reinterpret_cast<size_t>(&mask[8 - jpp.c_tail]));
        vmovups(vmm_c_tail_mask, ptr[tmp_gpr]);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* mov(tmp_gpr, reinterpret_cast<size_t>(&mask[8 - jpp.c_tail])); */
        CG::mov_imm(xa::XReg(IDX(tmp_gpr)),
                reinterpret_cast<size_t>(&mask[8 - jpp.c_tail]));
        //vmovups(vmm_c_tail_mask, ptr[tmp_gpr]);
        CG::ld1w(xa::ZRegS(IDX(vmm_c_tail_mask)), p_lsb / xa::T_z,
                xa::ptr(xa::XReg(IDX(tmp_gpr))));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    }
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::put_one_in_vmm() {
#ifdef DNNL_X64_IMPLEMENTATION
    mov(tmp_gpr, 1);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //mov(tmp_gpr, 1);
    CG::mov_imm(xa::XReg(IDX(tmp_gpr)), 1);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    uni_broadcast_reg_val(tmp_gpr.getIdx(), vmm_one.getIdx());
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::uni_broadcast_reg_val(
        const int reg_idx, const int vmm_idx) {
#ifdef DNNL_X64_IMPLEMENTATION
    movq(Xmm(vmm_idx), reg64_t(reg_idx));
    uni_vpbroadcastd(Vmm(vmm_idx), Xmm(vmm_idx));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //movq(Xmm(vmm_idx), reg64_t(reg_idx));
    CG::ptrue(p_tmp0.d, xa::VL2);
    CG::mov(xa::ZRegD(vmm_idx), p_tmp0 / xa::T_m, 0);
    CG::ptrue(p_tmp0.d, xa::VL1);
    CG::mov(xa::ZRegD(vmm_idx), p_tmp0 / xa::T_m, xa::XReg(reg_idx));
    //uni_vpbroadcastd(Vmm(vmm_idx), Xmm(vmm_idx));
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        //assert(!"unreachable");
        CG::dup(xa::ZRegS(vmm_idx), xa::ZRegS(vmm_idx)[0]);
    } else if (vlen == 32) {
        CG::dup(xa::ZRegS(vmm_idx), xa::ZRegS(vmm_idx)[0]);
        CG::mov(xa::ZRegS(vmm_idx), P_MSB_256 / xa::T_m, 0);
    } else if (vlen == 16) {
        CG::dup(xa::VReg4S(vmm_idx), xa::VReg4S(vmm_idx)[0]);
        CG::mov(xa::ZRegS(vmm_idx), P_MSB_384 / xa::T_m, 0);
    } else {
        assert(!"unreachable");
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::push_vmm_val(const int idx) {
    Vmm val_to_store(idx);
#ifdef DNNL_X64_IMPLEMENTATION
    sub(rsp, val_to_store.getBit());
    uni_vmovups(ptr[rsp], val_to_store);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //sub(rsp, val_to_store.getBit());
    CG::sub_imm(xa::XReg(idx), xa::XReg(idx), val_to_store.getBit(), x_tmp_0);
    //uni_vmovups(ptr[rsp], val_to_store);
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        CG::str(xa::ZReg(IDX(val_to_store)), xa::ptr(xa::XReg(IDX(rsp))));
    } else if (vlen == 32) {
        CG::st1w(xa::ZRegS(IDX(val_to_store)), p_lsb,
                xa::ptr(xa::XReg(IDX(rsp))));
    } else if (vlen == 16) {
        CG::str(xa::QReg(IDX(val_to_store)), xa::ptr(xa::XReg(IDX(rsp))));
    } else {
        assert(!"unreachable");
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
}

template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::pop_vmm_val(const int idx) {
    Vmm val_to_load(idx);
#ifdef DNNL_X64_IMPLEMENTATION
    uni_vmovups(val_to_load, ptr[rsp]);
    add(rsp, val_to_load.getBit());
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //uni_vmovups(val_to_load, ptr[rsp]);
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) { //vmovups(Ymm, mem)
        CG::ldr(xa::ZReg(IDX(val_to_load)), xa::ptr(xa::XReg(IDX(rsp))));
    } else if (vlen == 32) { //vmovups(Ymm, mem)
        CG::ld1w(xa::ZRegS(IDX(val_to_load)), p_lsb / xa::T_z,
                xa::ptr(xa::XReg(IDX(rsp))));
    } else if (vlen == 16) { //movups(Xmm, mem)
        CG::ldr(xa::QReg(z_tmp0.getIdx()), xa::ptr(xa::XReg(IDX(rsp))));
        CG::mov(xa::ZRegD(IDX(val_to_load)), p_lsb / xa::T_m, z_tmp0.d);
    } else {
        assert(!"unreachable");
    }
    //add(rsp, val_to_load.getBit());
    CG::add_imm(xa::XReg(IDX(rsp)), xa::XReg(IDX(rsp)), val_to_load.getBit(),
            x_tmp_0);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
}

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::load(const int idx,
        const reg64_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    if (jpp.is_bf16) {
        /*TODO: maybe use vpmovzxwd + vpslld,
             * in order to free up vmm_idx() register */
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            Vmm vmm_to_load = is_c_tail_proccessing
                    ? Vmm(idx) | k_c_tail_mask | T_z
                    : Vmm(idx);
            vpmovzxwd(vmm_to_load, ptr[reg_ptr + offset]);
            vpslld(vmm_to_load, vmm_to_load, 16);
        } else {
            vmovups(Ymm(idx), ptr[reg_ptr + offset]);
            vpermw(Vmm(idx) | k_mask_cvt | T_z, vmm_idx(), Vmm(idx));
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == sse41) {
                for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                    pinsrd(Xmm(idx), ptr[reg_ptr + offset + i * jpp.dt_size],
                            i);
                }
            } else if (isa == avx) {
                vmaskmovps(Vmm(idx), vmm_c_tail_mask, ptr[reg_ptr + offset]);
            } else {
                vmovups(Zmm(idx) | k_c_tail_mask | T_z, ptr[reg_ptr + offset]);
            }
        } else {
            uni_vmovups(Vmm(idx), ptr[reg_ptr + offset]);
        }
    }
}
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::load(const int idx, const xreg_t &reg_ptr,
        const int offset, const bool is_c_tail_proccessing) {
    const int vlen = cpu_isa_traits<isa>::vlen;
    if (jpp.is_bf16) {
        /*TODO: maybe use vpmovzxwd + vpslld,
             * in order to free up vmm_idx() register */
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (vlen == 64) {
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                if (is_c_tail_proccessing) {
                    assert(!"unreachable");
                } else {
                    //vpmovzxwd(Vmm(idx), ptr[reg_ptr + offset]);
                    CG::ldr(z_tmp0, xa::ptr(x_tmp_addr));
                    CG::zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                    //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                    CG::uxth(xa::ZReg(idx).s, p_512 / xa::T_m, z_tmp0.s);
                    //vpslld(vmm_to_load, vmm_to_load, 16);
                    CG::lsl(xa::ZReg(idx).s, xa::ZReg(idx).s, 16);
                }
            } else if (vlen == 32) {
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                if (is_c_tail_proccessing) {
                    assert(!"unreachable");
                } else {
                    //vpmovzxwd(vmm_to_load, ptr[reg_ptr + offset]);
                    CG::ldr(z_tmp0, xa::ptr(x_tmp_addr));
                    CG::zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                    //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                    CG::uxth(xa::ZReg(idx).s, p_512 / xa::T_m, z_tmp0.s);
                    CG::mov(xa::ZReg(idx).s, P_MSB_256 / xa::T_m, 0);
                    //vpslld(vmm_to_load, vmm_to_load, 16);
                    CG::lsl(xa::ZReg(idx).s, xa::ZReg(idx).s, 16);
                    //CG::mov(xa::ZReg(idx).s, P_MSB_256/xa::T_m, 0);
                }
            } else if (vlen == 16) {
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                if (is_c_tail_proccessing) {
                    //vpmovzxwd(Vmm(idx) | k_c_tail_mask | T_z, ptr[reg_ptr + offset]);
                    //vpslld(Vmm(idx) | k_c_tail_mask | T_z, Vmm(idx), 16);
                    assert(!"unreachable");
                } else {
                    //vpmovzxwd(vmm_to_load, ptr[reg_ptr + offset]);
                    CG::ldr(z_tmp0, xa::ptr(x_tmp_addr));
                    CG::zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                    //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                    CG::uxth(xa::ZReg(idx).s, p_512 / xa::T_m, z_tmp0.s);
                    CG::mov(xa::ZReg(idx).s, P_MSB_384 / xa::T_m, 0);
                    //vpslld(vmm_to_load, vmm_to_load, 16);
                    CG::lsl(xa::ZReg(idx).s, xa::ZReg(idx).s, 16);
                    //CG::mov(vmm_to_load.s, P_MSB_384/xa::T_m, 0);
                }
            } else {
                assert(!"unreachable");
            }
        } else {
            //get mem address
            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
            //vmovups(Ymm(idx), ptr[reg_ptr + offset]);
            CG::ld1w(xa::ZRegS(idx), p_256 / xa::T_z, xa::ptr(x_tmp_addr));
            //vpermw(Vmm(idx) | k_mask_cvt | T_z, vmm_idx(), Vmm(idx));
            if (vlen == 64) {
                //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                CG::mov(z_tmp0.h, 31);
                CG::and_(z_tmp0.b, p_512, xa::ZRegB(reg_idx()));
                for (int i = 0; i < 16; i++) {
                    CG::cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    CG::dup(z_tmp2.h, xa::ZRegH(idx)[i]);
                    CG::mov(z_tmp3.h, p_tmp1 / xa::T_m, z_tmp2.h);
                }
                CG::sub(z_tmp0.h, 16);
                for (int i = 0; i < 16; i++) {
                    CG::cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    CG::dup(z_tmp2.h, xa::ZRegH(idx)[16 + i]);
                    CG::mov(z_tmp3.h, p_tmp1 / xa::T_m, z_tmp2.h);
                }
                CG::mov(xa::ZRegH(idx), 0);
                CG::mov(xa::ZRegH(idx), xa::PReg(IDX(k_mask_cvt)) / xa::T_m,
                        z_tmp3.h);
            } else if (vlen == 32) {
                //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                CG::mov(z_tmp0.h, 15);
                CG::and_(z_tmp0.b, p_512, xa::ZRegB(reg_idx()));
                for (int i = 0; i < 16; i++) {
                    CG::cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    CG::dup(z_tmp2.h, xa::ZRegH(idx)[i]);
                    CG::mov(z_tmp3.h, p_tmp1 / xa::T_m, z_tmp2.h);
                }
                CG::mov(xa::ZRegH(idx), 0);
                CG::mov(xa::ZRegH(idx), xa::PReg(IDX(k_mask_cvt)) / xa::T_m,
                        z_tmp3.h);
                CG::mov(xa::ZRegH(idx), P_MSB_256 / xa::T_m, 0);
            } else if (vlen == 16) {
                //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                CG::mov(z_tmp0.h, 15);
                CG::and_(z_tmp0.b, p_512, xa::ZRegB(reg_idx()));
                for (int i = 0; i < 16; i++) {
                    CG::cmpeq(p_tmp1.h, p_512, z_tmp0.h, i);
                    CG::dup(z_tmp2.h, xa::ZRegH(idx)[i]);
                    CG::mov(z_tmp3.h, p_tmp1 / xa::T_m, z_tmp2.h);
                }
                CG::mov(xa::ZRegH(idx), 0);
                CG::mov(xa::ZRegH(idx), xa::PReg(IDX(k_mask_cvt)) / xa::T_m,
                        z_tmp3.h);
                CG::mov(xa::ZRegH(idx), P_MSB_384 / xa::T_m, 0);
            } else {
                assert(!"unreachable");
            }
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == sse41) {
                for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                    // pinsrd(Xmm(idx), ptr[reg_ptr + offset + i * jpp.dt_size],
                    //          i);
                    //get mem address
                    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)),
                            (offset + i * jpp.dt_size), x_tmp_0);
                    CG::ld1r(xa::VReg4S(z_tmp0.getIdx()), xa::ptr(x_tmp_addr));
                    CG::ptrue(p_tmp1.s, static_cast<xa::Pattern>(i + 1));
                    if (i) {
                        CG::ptrue(p_tmp2.s, static_cast<xa::Pattern>(i));
                    } else {
                        CG::pfalse(p_tmp2.b);
                    }
                    CG::bic(p_tmp1.b, P_ALL_ONE / xa::T_z, p_tmp1.b, p_tmp2.b);
                    CG::sel(xa::ZRegS(idx), p_tmp1 / xa::T_m,
                            xa::ZRegS(z_tmp0.getIdx()), xa::ZRegS(idx));
                }
            } else if (isa == avx) {
                //vmaskmovps(Vmm(idx), vmm_c_tail_mask, ptr[reg_ptr + offset]);
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                CG::cmplt(p_tmp0.s, p_lsb / xa::T_z,
                        xa::ZReg(IDX(vmm_c_tail_mask)).s, 0);
                CG::ld1w(xa::ZRegS(idx), p_lsb / xa::T_z, xa::ptr(x_tmp_addr));
            } else {
                //vmovups(Zmm(idx) | k_c_tail_mask | T_z, ptr[reg_ptr + offset]);
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                CG::ld1w(xa::ZRegS(idx), xa::PReg(IDX(k_c_tail_mask)) / xa::T_z,
                        xa::ptr(x_tmp_addr));
            }
        } else {
            //uni_vmovups(Vmm(idx), ptr[reg_ptr + offset]);
            //get mem address
            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
            CG::ld1w(xa::ZRegS(idx), p_lsb / xa::T_z, xa::ptr(x_tmp_addr));
        }
    }
}
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

#ifdef DNNL_X64_IMPLEMENTATION
template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::store(const int idx,
        const reg64_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    if (jpp.is_bf16) {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            vmovdqu16(ptr[reg_ptr + offset] | k_c_tail_mask, Ymm(idx));
        } else {
            vmovups(yword[reg_ptr + offset], Ymm(idx));
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == sse41) {
                for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                    pextrd(ptr[reg_ptr + offset + i * jpp.dt_size], Xmm(idx),
                            i);
                }
            } else if (isa == avx) {
                vmaskmovps(ptr[reg_ptr + offset], vmm_c_tail_mask, Vmm(idx));
            } else {
                vmovups(ptr[reg_ptr + offset] | k_c_tail_mask, Zmm(idx));
            }
        } else {
            uni_vmovups(vmmword[reg_ptr + offset], Vmm(idx));
        }
    }
}
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
template <cpu_isa_t isa>
inline void jit_uni_pool_kernel<isa>::store(const int idx,
        const xreg_t &reg_ptr, const int offset,
        const bool is_c_tail_proccessing) {
    const int vlen = cpu_isa_traits<isa>::vlen;
    if (jpp.is_bf16) {
        //get mem address
        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            //vmovdqu16(ptr[reg_ptr + offset] | k_c_tail_mask, Ymm(idx));
            if (vlen == 64) {
                CG::st1h(xa::ZRegH(idx), xa::PReg(IDX(k_c_tail_mask)),
                        xa::ptr(x_tmp_addr));
            } else if (vlen == 32) {
                CG::bic(p_tmp0.b, P_ALL_ONE / xa::T_z,
                        xa::PRegB(IDX(k_c_tail_mask)), P_MSB_256.b);
                CG::st1h(xa::ZRegH(idx), p_tmp0, xa::ptr(x_tmp_addr));
            } else if (vlen == 16) {
                CG::bic(p_tmp0.b, P_ALL_ONE / xa::T_z,
                        xa::PRegB(IDX(k_c_tail_mask)), P_MSB_384.b);
                CG::st1h(xa::ZRegH(idx), p_tmp0, xa::ptr(x_tmp_addr));
            } else {
                assert(!"unreachable");
            }
        } else {
            //vmovups(yword[reg_ptr + offset], Ymm(idx));
            //get mem address
            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
            CG::st1w(xa::ZRegS(idx), p_lsb, xa::ptr(x_tmp_addr));
        }
    } else {
        if (is_c_tail_proccessing && !jpp.is_c_padded) {
            if (isa == sse41) {
                for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++) {
                    //pextrd(ptr[reg_ptr + offset + i * jpp.dt_size], Xmm(idx),
                    //	 i);
                    //get mem address
                    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)),
                            (offset + i * jpp.dt_size), x_tmp_0);
                    uint32_t sel = i & 3;
                    CG::mov(w_tmp_0, xa::VReg(idx).s[sel]);
                    CG::str(w_tmp_0, xa::ptr(x_tmp_addr));
                }
            } else if (isa == avx) {
                //vmaskmovps(ptr[reg_ptr + offset], vmm_c_tail_mask, Vmm(idx));
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                if (vlen == 64) {
                    assert(!"unreachable");
                } else if (vlen == 32) {
                    CG::cmplt(p_tmp0.s, p_lsb / xa::T_z,
                            xa::ZReg(IDX(vmm_c_tail_mask)).s, 0);
                    CG::st1w(xa::ZReg(idx).s, p_tmp0 / xa::T_m,
                            xa::ptr(x_tmp_addr));
                } else if (vlen == 16) {
                    assert(!"unreachable");
                } else {
                    assert(!"unreachable");
                }
            } else {
                //vmovups(ptr[reg_ptr + offset] | k_c_tail_mask, Zmm(idx));
                //get mem address
                CG::add_imm(
                        x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
                CG::st1w(xa::ZRegS(idx), xa::PReg(IDX(k_c_tail_mask)),
                        xa::ptr(x_tmp_addr));
            }
        } else {
            //uni_vmovups(vmmword[reg_ptr + offset], Vmm(idx));
            //get mem address
            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_ptr)), offset, x_tmp_0);
            CG::st1w(xa::ZRegS(idx), p_lsb, xa::ptr(x_tmp_addr));
        }
    }
}
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
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
#ifdef DNNL_X64_IMPLEMENTATION
            mov(tmp_gpr, float2int((float)non_zero_kw));
            movq(xmm_tmp, tmp_gpr);
            uni_vbroadcastss(vmm_tmp, xmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            /* mov(tmp_gpr, float2int((float)non_zero_kw)); */
            CG::mov_imm(xa::XReg(IDX(tmp_gpr)), float2int((float)non_zero_kw));
            //movq(xmm_tmp, tmp_gpr);
            CG::ptrue(p_tmp0.d, xa::VL2);
            CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
            CG::ptrue(p_tmp0.d, xa::VL1);
            CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m,
                    xa::XReg(IDX(tmp_gpr)));
            //uni_vbroadcastss(vmm_tmp, xmm_tmp);
            const int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
                CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_256 / xa::T_m, 0);
            } else if (vlen == 16) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
                CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_384 / xa::T_m, 0);
            } else {
                assert(!"unreachable");
            }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
            if (with_c_tail_proccessing && isa == avx) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                uni_broadcast_reg_val(
                        reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
            }
#ifdef DNNL_X64_IMPLEMENTATION
            uni_vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            /* uni_vmulps(vmm_tmp, vmm_tmp, vmm_ker_area_h);
	       const int vlen = cpu_isa_traits<isa>::vlen; */
            if (vlen == 64) {
                CG::fmul(xa::ZReg(IDX(vmm_tmp)).s, xa::ZReg(IDX(vmm_tmp)).s,
                        xa::ZReg(IDX(vmm_ker_area_h)).s);
            } else if (vlen == 32) {
                CG::fmul(xa::ZReg(IDX(vmm_tmp)).s, xa::ZReg(IDX(vmm_tmp)).s,
                        xa::ZReg(IDX(vmm_ker_area_h)).s);
                CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_256 / xa::T_m, 0);
            } else if (vlen == 16) {
                CG::fmul(xa::VReg(IDX(vmm_tmp)).s4, xa::VReg(IDX(vmm_tmp)).s4,
                        xa::VReg(IDX(vmm_ker_area_h)).s4);
            } else {
                assert(!"unreachable");
            }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
            if (with_c_tail_proccessing && isa == avx) {
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
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto reg_ind = [&](int shift, int bc, int j) {
        return shift * ur_bc * ur_w + bc * ur_w + j;
    };
    auto is_tail_processing = [&](int bc) {
        if (isa == sse41 && !jpp.is_c_padded) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for (int jj = 0; jj < ur_w; jj++) {
        if (jpp.is_backward)
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
        for (int bci = 0; bci < ur_bc; bci++) {
            auto accr_i = reg_ind(0, bci, jj);
            auto accvr = vreg(accr_i);
            if (jpp.is_backward) {
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
#ifdef DNNL_X64_IMPLEMENTATION
                load(accvr.getIdx(), reg_output, output_offset,
                        is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                load(accvr.getIdx(), xreg_output, output_offset,
                        is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
#ifdef DNNL_X64_IMPLEMENTATION
                uni_vdivps(accvr, accvr, vmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                /* uni_vdivps(accvr, accvr, vmm_tmp); */
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    //CG::mov(p_tmp0.b, P_ALL_ONE, P_ALL_ONE.b);
                    CG::fdiv(xa::ZRegS(IDX(accvr)), p_512,
                            xa::ZRegS(IDX(vmm_tmp)));
                } else if (vlen == 32) {
                    //CG::mov(p_tmp0.b, P_ALL_ONE, P_ALL_ONE.b);
                    CG::fdiv(xa::ZRegS(IDX(accvr)), p_512,
                            xa::ZRegS(IDX(vmm_tmp)));
                    CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m, 0);
                } else if (vlen == 16) {
                    CG::fdiv(xa::VReg(IDX(accvr)).s4, xa::VReg(IDX(accvr)).s4,
                            xa::VReg(IDX(vmm_tmp)).s4);
                } else {
                    assert(!"unreachable");
                }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
            } else {
#ifdef DNNL_X64_IMPLEMENTATION
                uni_vpxor(accvr, accvr, accvr);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                /* uni_vpxor(accvr, accvr, accvr); */
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    CG::eor(xa::ZReg(IDX(accvr)).d, xa::ZReg(IDX(accvr)).d,
                            xa::ZReg(IDX(accvr)).d);
                } else if (vlen == 32) {
                    CG::eor(xa::ZRegD(IDX(accvr)), xa::ZRegD(IDX(accvr)),
                            xa::ZRegD(IDX(accvr)));
                    CG::mov(xa::ZRegS(IDX(accvr)), P_MSB_256 / xa::T_m, 0);
                } else if (vlen == 16) {
                    CG::eor(xa::VReg16B(IDX(accvr)), xa::VReg16B(IDX(accvr)),
                            xa::VReg16B(IDX(accvr)));
                } else {
                    assert(!"unreachable");
                }

#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
            }
        }
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
#ifdef DNNL_X64_IMPLEMENTATION
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* push(reg_input); */
        CG::str(xa::XReg(IDX(reg_input)), xa::pre_ptr(X_TRANSLATOR_STACK, -8));
        //push(reg_output);
        CG::str(xa::XReg(IDX(reg_output)), xa::pre_ptr(X_TRANSLATOR_STACK, -8));
        //mov(aux_reg_input_d, reg_input);
        CG::mov(xa::XReg(IDX(aux_reg_input_d)), xa::XReg(IDX(reg_input)));
        //mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        //get mem address
        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(kd_padding),
                x_tmp_0);
        CG::ldr(xa::XReg(IDX(ki)), xa::ptr(x_tmp_addr));
        L(kd_label);
        //mov(aux_reg_input, aux_reg_input_d);
        CG::mov(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(aux_reg_input_d)));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    } else {
#ifdef DNNL_X64_IMPLEMENTATION
        mov(aux_reg_input, reg_input);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* mov(aux_reg_input, reg_input); */
        CG::mov(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(reg_input)));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    }

#ifdef DNNL_X64_IMPLEMENTATION
    xor_(kj, kj);
#else /* #ifdef DNNL_X64_IMPLEMENTATIO */
    //xor_(kj, kj);
    CG::eor(xa::XReg(IDX(kj)), xa::XReg(IDX(kj)), xa::XReg(IDX(kj)));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);

            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                auto accvr = vreg(reg_ind(0, bci, jj));
                auto inpr_i = reg_ind(1, bci, jj);
                auto inpvr = vreg(inpr_i);
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = dt_size * aux_input_offset;
                if (jpp.is_backward) {
                    auto inpyr = yreg(inpr_i);
#ifdef DNNL_X64_IMPLEMENTATION
                    load(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
                    uni_vaddps(inpvr, inpvr, accvr);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    load(reg_idx(inpr_i), aux_xreg_input, input_offset,
                            is_tail_processing(bci));
                    //uni_vaddps(inpvr, inpvr, accvr);
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        CG::fadd(xa::ZReg(IDX(inpvr)).s, xa::ZReg(IDX(inpvr)).s,
                                xa::ZReg(IDX(accvr)).s);
                    } else if (vlen == 32) {
                        CG::fadd(xa::ZReg(IDX(inpvr)).s, xa::ZReg(IDX(inpvr)).s,
                                xa::ZReg(IDX(accvr)).s);
                        CG::mov(xa::ZReg(IDX(inpvr)).s, P_MSB_256 / xa::T_m, 0);
                    } else if (vlen == 16) {
                        CG::fadd(xa::VReg(IDX(inpvr)).s4,
                                xa::VReg(IDX(inpvr)).s4,
                                xa::VReg(IDX(accvr)).s4);
                    } else {
                        assert(!"unreachable");
                    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa))
                            bf16_emu_->vcvtneps2bf16(inpyr, zreg(inpr_i));
                        else
                            vcvtneps2bf16(inpyr, inpvr);
                    }
#ifdef DNNL_X64_IMPLEMENTATION
                    store(reg_idx(inpr_i), aux_reg_input, input_offset,
                            is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    store(reg_idx(inpr_i), aux_xreg_input, input_offset,
                            is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                } else {
                    if (jpp.is_bf16 || is_tail_processing(bci)
                            || (isa == sse41
                                    && c_off % (jpp.c_block / 2) != 0)) {
#ifdef DNNL_X64_IMPLEMENTATION
                        load(vmm_tmp_1.getIdx(), aux_reg_input, input_offset,
                                is_tail_processing(bci));
                        uni_vaddps(accvr, accvr, vmm_tmp_1);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                        load(vmm_tmp_1.getIdx(), aux_xreg_input, input_offset,
                                is_tail_processing(bci));
                        //uni_vaddps(accvr, accvr, vmm_tmp_1);
                        int vlen = cpu_isa_traits<isa>::vlen;
                        if (vlen == 64) {
                            CG::fadd(xa::ZReg(IDX(accvr)).s,
                                    xa::ZReg(IDX(accvr)).s,
                                    xa::ZReg(IDX(vmm_tmp_1)).s);
                        } else if (vlen == 32) {
                            CG::fadd(xa::ZReg(IDX(accvr)).s,
                                    xa::ZReg(IDX(accvr)).s,
                                    xa::ZReg(IDX(vmm_tmp_1)).s);
                            CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m,
                                    0);
                        } else if (vlen == 16) {
                            CG::fadd(xa::VReg(IDX(accvr)).s4,
                                    xa::VReg(IDX(accvr)).s4,
                                    xa::VReg(IDX(vmm_tmp_1)).s4);
                        } else {
                            assert(!"unreachable");
                        }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                    } else {
#ifdef DNNL_X64_IMPLEMENTATION
                        uni_vaddps(accvr, accvr,
                                ptr[aux_reg_input + input_offset]);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                        /* uni_vaddps(accvr, accvr,
			   ptr[aux_reg_input + input_offset]); */
                        CG::ptrue(p_tmp0.b, xa::VL1);
                        CG::ptrue(p_tmp0.b, xa::VL2);
                        int vlen = cpu_isa_traits<isa>::vlen;
                        //get mem address
                        CG::add_imm(x_tmp_addr, xa::XReg(IDX(aux_reg_input)),
                                input_offset, x_tmp_0);
                        if (vlen == 64) {
                            CG::ldr(z_tmp0, xa::ptr(x_tmp_addr));
                            CG::fadd(xa::ZReg(IDX(accvr)).s,
                                    xa::ZReg(IDX(accvr)).s, z_tmp0.s);
                        } else if (vlen == 32) {
                            CG::ldr(z_tmp0, xa::ptr(x_tmp_addr));
                            CG::fadd(xa::ZReg(IDX(accvr)).s,
                                    xa::ZReg(IDX(accvr)).s, z_tmp0.s);
                            CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m,
                                    0);
                        } else if (vlen == 16) {
                            CG::ld1(xa::VReg(z_tmp0.getIdx()).s4,
                                    xa::ptr(x_tmp_addr));
                            CG::fadd(xa::VReg(IDX(accvr)).s4,
                                    xa::VReg(IDX(accvr)).s4,
                                    xa::VReg(z_tmp0.getIdx()).s4);
                        } else {
                            assert(!"unreachable");
                        }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                    }
                }
            }
        }
#ifdef DNNL_X64_IMPLEMENTATION
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* add(aux_reg_input, jpp.dt_size * iw * c_off); */
        CG::add_imm(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(aux_reg_input)),
                (jpp.dt_size * iw * c_off), x_tmp_0);
        //inc(kj);
        CG::adds(xa::XReg(IDX(kj)), xa::XReg(IDX(kj)), 1);
        //cmp(kj, reg_kh);
        CG::cmp(xa::XReg(IDX(kj)), xa::XReg(IDX(reg_kh)));
        //jl(kh_label, T_NEAR);
        CG::b(xa::LT, kh_label);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    }

    if (jpp.simple_alg && jpp.ndims == 5) {
#ifdef DNNL_X64_IMPLEMENTATION
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off); */
        CG::add_imm(xa::XReg(IDX(aux_reg_input_d)),
                xa::XReg(IDX(aux_reg_input_d)),
                (jpp.dt_size * jpp.ih * iw * c_off), x_tmp_0);
        //dec(ki);
        CG::subs(xa::XReg(IDX(ki)), xa::XReg(IDX(ki)), 1);
        //cmp(ki, 0);
        CG::mov_imm(x_tmp_0, 0);
        CG::cmp(xa::XReg(IDX(ki)), x_tmp_0);
        //jg(kd_label, T_NEAR);
        CG::b(xa::GT, kd_label);
        //pop(reg_output);
        CG::ldr(xa::XReg(IDX(reg_output)), xa::post_ptr(X_TRANSLATOR_STACK, 8));
        //pop(reg_input);
        CG::ldr(xa::XReg(IDX(reg_input)), xa::post_ptr(X_TRANSLATOR_STACK, 8));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    }

    if (!jpp.is_backward) {
        for (int jj = 0; jj < ur_w; jj++) {
            maybe_recalculate_divisor(
                    jj, ur_w, pad_l, pad_r, with_c_tail_proccessing);
            for (int bci = 0; bci < ur_bc; bci++) {
                auto accr_i = reg_ind(0, bci, jj);
                auto accvr = vreg(accr_i);
                auto output_offset = dt_size * (jj * c_off + bci * c_block);
#ifdef DNNL_X64_IMPLEMENTATION
                uni_vdivps(accvr, accvr, vmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                /* uni_vdivps(accvr, accvr, vmm_tmp);*/
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    //CG::mov(p_tmp0.b, P_ALL_ONE, P_ALL_ONE.b);
                    CG::fdiv(xa::ZRegS(IDX(accvr)), p_512,
                            xa::ZRegS(IDX(vmm_tmp)));
                } else if (vlen == 32) {
                    //CG::mov(p_tmp0.b, P_ALL_ONE, P_ALL_ONE.b);
                    CG::fdiv(xa::ZRegS(IDX(accvr)), p_512,
                            xa::ZRegS(IDX(vmm_tmp)));
                    CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m, 0);
                } else if (vlen == 16) {
                    CG::fdiv(xa::VReg(IDX(accvr)).s4, xa::VReg(IDX(accvr)).s4,
                            xa::VReg(IDX(vmm_tmp)).s4);
                } else {
                    assert(!"unreachable");
                }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

                if (jpp.is_bf16) {
                    auto acczr = zreg(accr_i);
                    auto accyr = yreg(accr_i);
                    if (!isa_has_bf16(jpp.isa))
                        bf16_emu_->vcvtneps2bf16(accyr, acczr);
                    else
                        vcvtneps2bf16(accyr, accvr);
                }
#ifdef DNNL_X64_IMPLEMENTATION
                store(reg_idx(accr_i), reg_output, output_offset,
                        is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                store(reg_idx(accr_i), xreg_output, output_offset,
                        is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
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
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto reg_ind = [&](int shift, int bc, int j) {
        return shift * ur_bc * ur_w + bc * ur_w + j;
    };
    auto is_tail_processing = [&](int bc) {
        if (isa == sse41 && !jpp.is_c_padded) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

#ifdef DNNL_X64_IMPLEMENTATION
    mov(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));
    movq(xmm_tmp, tmp_gpr);
    uni_vbroadcastss(vmm_tmp, xmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //mov(tmp_gpr, float2int(nstl::numeric_limits<float>::lowest()));
    CG::mov_imm(xa::XReg(IDX(tmp_gpr)),
            float2int(nstl::numeric_limits<float>::lowest()));
    //movq(xmm_tmp, tmp_gpr);
    CG::ptrue(p_tmp0.d, xa::VL2);
    CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
    CG::ptrue(p_tmp0.d, xa::VL1);
    CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, xa::XReg(IDX(tmp_gpr)));
    //uni_vbroadcastss(vmm_tmp, xmm_tmp);
    const int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
    } else if (vlen == 32) {
        CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
        CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_256 / xa::T_m, 0);
    } else if (vlen == 16) {
        CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
        CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_384 / xa::T_m, 0);
    } else {
        assert(!"unreachable");
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        auto accvr = vreg(reg_ind(0, bci, jj));
#ifdef DNNL_X64_IMPLEMENTATION
        uni_vmovups(accvr, vmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* uni_vmovups(accvr, vmm_tmp);*/
        int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) { //vmovups(Zmm, Zmm)
            CG::mov(xa::ZRegD(IDX(accvr)), xa::ZRegD(IDX(vmm_tmp)));
        } else if (vlen == 32) { //vmovups(Ymm, Ymm)
            CG::mov(xa::ZRegD(IDX(accvr)), xa::ZRegD(IDX(vmm_tmp)));
            CG::mov(xa::ZRegS(IDX(accvr)), P_MSB_256 / xa::T_m, 0);
        } else if (vlen == 16) { //movups(Xmm, Xmm)
            CG::mov(xa::VReg16B(IDX(accvr)), xa::VReg16B(IDX(vmm_tmp)));
        } else {
            assert(!"unreachable");
        }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        if (jpp.is_training) {
            auto indvr = vreg(reg_ind(2, bci, jj));
#ifdef DNNL_X64_IMPLEMENTATION
            uni_vpxor(indvr, indvr, indvr);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            /* uni_vpxor(indvr, indvr, indvr);*/
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                CG::eor(xa::ZReg(IDX(indvr)).d, xa::ZReg(IDX(indvr)).d,
                        xa::ZReg(IDX(indvr)).d);
            } else if (vlen == 32) {
                CG::eor(xa::ZRegD(IDX(indvr)), xa::ZRegD(IDX(indvr)),
                        xa::ZRegD(IDX(indvr)));
                CG::mov(xa::ZRegS(IDX(indvr)), P_MSB_256 / xa::T_m, 0);
            } else if (vlen == 16) {
                CG::eor(xa::VReg16B(IDX(indvr)), xa::VReg16B(IDX(indvr)),
                        xa::VReg16B(IDX(indvr)));
            } else {
                assert(!"unreachable");
            }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        }
    }
#ifdef DNNL_X64_IMPLEMENTATION
    if (jpp.is_training) {
        movq(xmm_tmp, reg_k_shift);
        uni_vpbroadcastd(vmm_k_offset, xmm_tmp);
    }
    if (jpp.ndims == 5) {
        push(reg_input);
        push(reg_output);
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }
    xor_(kj, kj);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    if (jpp.is_training) {
        //movq(xmm_tmp, reg_k_shift);
        CG::ptrue(p_tmp0.d, xa::VL2);
        CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
        CG::ptrue(p_tmp0.d, xa::VL1);
        CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m,
                xa::XReg(IDX(reg_k_shift)));
        //uni_vpbroadcastd(vmm_k_offset, xmm_tmp);
        int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) {
            CG::dup(xa::ZRegS(IDX(vmm_k_offset)), xa::ZRegS(IDX(xmm_tmp))[0]);
        } else if (vlen == 32) {
            CG::dup(xa::ZRegS(IDX(vmm_k_offset)), xa::ZRegS(IDX(xmm_tmp))[0]);
            CG::mov(xa::ZRegS(IDX(vmm_k_offset)), P_MSB_256 / xa::T_m, 0);
        } else if (vlen == 16) {
            CG::dup(xa::VReg4S(IDX(vmm_k_offset)), xa::VReg4S(IDX(xmm_tmp))[0]);
            CG::mov(xa::ZRegS(IDX(vmm_k_offset)), P_MSB_384 / xa::T_m, 0);
        } else {
            assert(!"unreachable");
        }
    }
    if (jpp.ndims == 5) {
        //push(reg_input);
        CG::str(xa::XReg(IDX(reg_input)), xa::pre_ptr(X_TRANSLATOR_STACK, -8));
        //push(reg_output);
        CG::str(xa::XReg(IDX(reg_output)), xa::pre_ptr(X_TRANSLATOR_STACK, -8));
        //mov(aux_reg_input_d, reg_input);
        CG::mov(xa::XReg(IDX(aux_reg_input_d)), xa::XReg(IDX(reg_input)));
        //mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        //get mem address
        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(kd_padding),
                x_tmp_0);
        CG::ldr(xa::XReg(IDX(ki)), xa::ptr(x_tmp_addr));
        L(kd_label);
        //mov(aux_reg_input, aux_reg_input_d);
        CG::mov(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(aux_reg_input_d)));
    } else {
        //mov(aux_reg_input, reg_input);
        CG::mov(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(reg_input)));
    }
    //xor_(kj, kj);
    CG::eor(xa::XReg(IDX(kj)), xa::XReg(IDX(kj)), xa::XReg(IDX(kj)));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                auto accvr = vreg(reg_ind(0, bci, jj));
                auto inpr_i = reg_ind(1, bci, jj);
                auto inpvr = vreg(inpr_i);
                auto indvr = vreg(reg_ind(2, bci, jj));
                auto cvtvr = vreg(reg_ind(3, bci, jj));
                int aux_input_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_input_offset >= iw * c_off) continue;
                int input_offset = jpp.dt_size * aux_input_offset;
#ifdef DNNL_X64_IMPLEMENTATION
                load(reg_idx(inpr_i), aux_reg_input, input_offset,
                        is_tail_processing(bci));
                if (isa == sse41) {
                    movups(vmm_mask, accvr);
                    cmpps(vmm_mask, inpvr, _cmp_lt_os);
                    blendvps(accvr, inpvr);
                    if (jpp.is_training) blendvps(indvr, vmm_k_offset);
                } else if (isa == avx) {
                    vcmpps(cvtvr, accvr, inpvr, _cmp_lt_os);
                    vblendvps(accvr, accvr, inpvr, cvtvr);
                    if (jpp.is_training)
                        vblendvps(indvr, indvr, vmm_k_offset, cvtvr);
                } else {
                    vcmpps(k_store_mask, accvr, inpvr, _cmp_lt_os);
                    vblendmps(accvr | k_store_mask, accvr, inpvr);
                    if (jpp.is_training)
                        vblendmps(indvr | k_store_mask, indvr, vmm_k_offset);
                }
            }
            if (jpp.is_training) {
                if (with_c_tail_proccessing && isa == avx) {
                    push_vmm_val(vmm_c_tail_mask.getIdx());
                    put_one_in_vmm();
                }

                if (isa == avx && !mayiuse(avx2))
                    avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
                else
                    uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);

                if (with_c_tail_proccessing && isa == avx)
                    pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
        }
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }

    if (jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        if (jpp.is_training) {
            mov(tmp_gpr, ptr[reg_param + GET_OFF(kd_padding_shift)]);
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
            if (isa == avx && !mayiuse(avx2)) {
                Xmm t(vmm_mask.getIdx());
                avx_vpadd1(vmm_k_offset, xmm_tmp, t);
            } else {
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
            }
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        pop(reg_output);
        pop(reg_input);
    }

    if (with_c_tail_proccessing && jpp.is_c_padded && isa == sse41)
        mov(tmp_gpr, 0); // needed zero to fill padded tail
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                load(reg_idx(inpr_i), aux_xreg_input, input_offset,
                        is_tail_processing(bci));
                if (isa == sse41) {
                    //movups(vmm_mask, accvr);
                    CG::mov(xa::ZRegD(IDX(vmm_mask)), p_128 / xa::T_m,
                            xa::ZRegD(IDX(accvr)));
                    //cmpps(vmm_mask, inpvr, _cmp_lt_os);
                    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, P_MSB_384.b);
                    uint cmpDstIdx = p_tmp0.getIdx();
                    uint cmpMaskIdx = p_tmp0.getIdx();
                    uint cmpSrcIdx = IDX(vmm_mask);
                    uint cmpSrc2Idx = IDX(inpvr);
                    switch (int(_cmp_lt_os)) {
                        case 0:
                            CG::fcmeq(xa::PRegS(cmpDstIdx),
                                    xa::PReg(cmpMaskIdx) / xa::T_z,
                                    xa::ZRegS(cmpSrcIdx),
                                    xa::ZRegS(cmpSrc2Idx));
                            break; //EQ_OQ
                        case 1:
                            CG::fcmlt(xa::PRegS(cmpDstIdx),
                                    xa::PReg(cmpMaskIdx) / xa::T_z,
                                    xa::ZRegS(cmpSrcIdx),
                                    xa::ZRegS(cmpSrc2Idx));
                            break; //LT_OS
                        case 2:
                            CG::fcmle(xa::PRegS(cmpDstIdx),
                                    xa::PReg(cmpMaskIdx) / xa::T_z,
                                    xa::ZRegS(cmpSrcIdx),
                                    xa::ZRegS(cmpSrc2Idx));
                            break; //LE_OS
                        case 4:
                            CG::fcmne(xa::PRegS(cmpDstIdx),
                                    xa::PReg(cmpMaskIdx) / xa::T_z,
                                    xa::ZRegS(cmpSrcIdx),
                                    xa::ZRegS(cmpSrc2Idx));
                            break; //NEQ_UQ
                        case 5:
                            CG::fcmge(xa::PRegS(cmpDstIdx),
                                    xa::PReg(cmpMaskIdx) / xa::T_z,
                                    xa::ZRegS(cmpSrcIdx),
                                    xa::ZRegS(cmpSrc2Idx));
                            break; //NLT_US
                        case 6:
                            CG::fcmgt(xa::PRegS(cmpDstIdx),
                                    xa::PReg(cmpMaskIdx) / xa::T_z,
                                    xa::ZRegS(cmpSrcIdx),
                                    xa::ZRegS(cmpSrc2Idx));
                            break; //NLE_US
                        case 3: //UNORD_Q
                        case 7: //ORD_Q
                        default: assert(!"unreachable"); break;
                    }
                    CG::cpy(z_tmp0.s, p_tmp0 / xa::T_z, 255);
                    CG::not_(p_tmp0.b, P_ALL_ONE / xa::T_z, P_MSB_384.b);
                    CG::mov(xa::ZRegS(IDX(vmm_mask)), p_tmp0 / xa::T_m,
                            z_tmp0.s);
                    //blendvps(accvr, inpvr);
                    CG::cmplt(p_tmp1.s, p_512 / xa::T_z, xa::ZReg(0).s, 0);
                    CG::and_(p_tmp1.b, P_ALL_ONE, p_tmp1.b, p_128.b);
                    CG::mov(xa::ZReg(IDX(accvr)).s, p_tmp1 / xa::T_m,
                            xa::ZReg(IDX(inpvr)).s);
                    if (jpp.is_training) {
                        //blendvps(indvr, vmm_k_offset);
                        CG::cmplt(p_tmp1.s, p_512 / xa::T_z, xa::ZReg(0).s, 0);
                        CG::and_(p_tmp1.b, P_ALL_ONE, p_tmp1.b, p_128.b);
                        CG::mov(xa::ZReg(IDX(indvr)).s, p_tmp1 / xa::T_m,
                                xa::ZReg(IDX(vmm_k_offset)).s);
                    }
                } else if (isa == avx) {
                    //vcmpps(cvtvr, accvr, inpvr, _cmp_lt_os);
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        uint cmpDstIdx = IDX(cvtvr);
                        uint cmpMaskIdx = p_512.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
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
                        //vblendvps(accvr, accvr, inpvr, cvtvr);
                        CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                        CG::cmpgt(p10.s, p_256 / xa::T_z, z_tmp0.s, 0);
                        CG::mov(xa::ZReg(IDX(accvr)).s, p10 / xa::T_m,
                                xa::ZReg(IDX(inpvr)).s);
                        CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                        CG::mov(xa::ZReg(IDX(accvr)).s, p_tmp0 / xa::T_m,
                                xa::ZReg(IDX(accvr)).s);
                        CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m, 0);
                        if (jpp.is_training) {
                            //vblendvps(indvr, indvr, vmm_k_offset, cvtvr);
                            if (vlen == 64) {
                                assert(!"unreachable");
                            } else if (vlen == 32) {
                                CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                                CG::cmpgt(p10.s, p_256 / xa::T_z, z_tmp0.s, 0);
                                CG::mov(xa::ZReg(IDX(indvr)).s, p10 / xa::T_m,
                                        xa::ZReg(IDX(vmm_k_offset)).s);
                                CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        p_tmp0 / xa::T_m,
                                        xa::ZReg(IDX(indvr)).s);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        P_MSB_256 / xa::T_m, 0);
                            } else if (vlen == 16) {
                                CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                                CG::cmpgt(p10.s, p_128 / xa::T_z, z_tmp0.s, 0);
                                CG::mov(xa::ZReg(IDX(indvr)).s, p10 / xa::T_m,
                                        xa::ZReg(IDX(vmm_k_offset)).s);
                                CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        p_tmp0 / xa::T_m,
                                        xa::ZReg(IDX(indvr)).s);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        P_MSB_384 / xa::T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    } else if (vlen == 32) {
                        CG::mov(p_tmp0.b, P_ALL_ONE / xa::T_z, P_MSB_256.b);
                        uint cmpDstIdx = IDX(cvtvr);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
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
                        //vblendvps(accvr, accvr, inpvr, cvtvr);
                        CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                        CG::cmpgt(p10.s, p_256 / xa::T_z, z_tmp0.s, 0);
                        CG::mov(xa::ZReg(IDX(accvr)).s, p10 / xa::T_m,
                                xa::ZReg(IDX(inpvr)).s);
                        CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                        CG::mov(xa::ZReg(IDX(accvr)).s, p_tmp0 / xa::T_m,
                                xa::ZReg(IDX(accvr)).s);
                        CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m, 0);
                        if (jpp.is_training) {
                            //vblendvps(indvr, indvr, vmm_k_offset, cvtvr);
                            if (vlen == 64) {
                                assert(!"unreachable");
                            } else if (vlen == 32) {
                                CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                                CG::cmpgt(p10.s, p_256 / xa::T_z, z_tmp0.s, 0);
                                CG::mov(xa::ZReg(IDX(indvr)).s, p10 / xa::T_m,
                                        xa::ZReg(IDX(vmm_k_offset)).s);
                                CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        p_tmp0 / xa::T_m,
                                        xa::ZReg(IDX(indvr)).s);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        P_MSB_256 / xa::T_m, 0);
                            } else if (vlen == 16) {
                                CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                                CG::cmpgt(p10.s, p_128 / xa::T_z, z_tmp0.s, 0);
                                CG::mov(xa::ZReg(IDX(indvr)).s, p10 / xa::T_m,
                                        xa::ZReg(IDX(vmm_k_offset)).s);
                                CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        p_tmp0 / xa::T_m,
                                        xa::ZReg(IDX(indvr)).s);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        P_MSB_384 / xa::T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    } else if (vlen == 16) {
                        CG::mov(p_tmp0.b, P_ALL_ONE / xa::T_z, P_MSB_256.b);
                        uint cmpDstIdx = IDX(cvtvr);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
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
                        //vblendvps(accvr, accvr, inpvr, cvtvr);
                        CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                        CG::cmpgt(p10.s, p_128 / xa::T_z, z_tmp0.s, 0);
                        CG::mov(xa::ZReg(IDX(accvr)).s, p10 / xa::T_m,
                                xa::ZReg(IDX(inpvr)).s);
                        CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                        CG::mov(xa::ZReg(IDX(accvr)).s, p_tmp0 / xa::T_m,
                                xa::ZReg(IDX(accvr)).s);
                        CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_384 / xa::T_m, 0);
                        if (jpp.is_training) {
                            //vblendvps(indvr, indvr, vmm_k_offset, cvtvr);
                            if (vlen == 64) {
                                assert(!"unreachable");
                            } else if (vlen == 32) {
                                CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                                CG::cmpgt(p10.s, p_256 / xa::T_z, z_tmp0.s, 0);
                                CG::mov(xa::ZReg(IDX(indvr)).s, p10 / xa::T_m,
                                        xa::ZReg(IDX(vmm_k_offset)).s);
                                CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        p_tmp0 / xa::T_m,
                                        xa::ZReg(IDX(indvr)).s);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        P_MSB_256 / xa::T_m, 0);
                            } else if (vlen == 16) {
                                CG::lsr(z_tmp0.s, xa::ZReg(IDX(cvtvr)).s, 31);
                                CG::cmpgt(p10.s, p_128 / xa::T_z, z_tmp0.s, 0);
                                CG::mov(xa::ZReg(IDX(indvr)).s, p10 / xa::T_m,
                                        xa::ZReg(IDX(vmm_k_offset)).s);
                                CG::not_(p_tmp0.b, P_ALL_ONE, p10.b);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        p_tmp0 / xa::T_m,
                                        xa::ZReg(IDX(indvr)).s);
                                CG::mov(xa::ZReg(IDX(indvr)).s,
                                        P_MSB_384 / xa::T_m, 0);
                            } else {
                                assert(!"unreachable");
                            }
                        }
                    } else {
                        assert(!"unreachable");
                    }
                } else {
                    //vcmpps(k_store_mask, accvr, inpvr, _cmp_lt_os);
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        uint cmpDstIdx = IDX(k_store_mask);
                        uint cmpMaskIdx = p_512.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
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
                        CG::mov(p_tmp0.b, P_ALL_ONE / xa::T_z, P_MSB_256.b);
                        uint cmpDstIdx = IDX(k_store_mask);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
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
                        CG::mov(p_tmp0.b, P_ALL_ONE / xa::T_z, P_MSB_384.b);
                        uint cmpDstIdx = IDX(k_store_mask);
                        uint cmpMaskIdx = p_tmp0.getIdx();
                        uint cmpSrcIdx = IDX(accvr);
                        uint cmpSrc2Idx = IDX(inpvr);
                        switch (int(_cmp_lt_os)) {
                            case 0:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OQ
                            case 1:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OS
                            case 2:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OS
                            case 4:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_UQ
                            case 5:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_US
                            case 6:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_US
                            case 8:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_UQ
                            case 9:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_US
                            case 10:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_US
                            case 12:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OQ
                            case 13:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OS
                            case 14:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GT_OS
                            case 16:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_OS
                            case 17:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LT_OQ
                            case 18:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //LE_OQ
                            case 20:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_US
                            case 21:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLT_UQ
                            case 22:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NLE_UQ
                            case 24:
                                CG::fcmeq(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //EQ_US
                            case 25:
                                CG::fcmlt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGE_UQ
                            case 26:
                                CG::fcmle(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NGT_UQ
                            case 28:
                                CG::fcmne(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //NEQ_OS
                            case 29:
                                CG::fcmge(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
                                break; //GE_OQ
                            case 30:
                                CG::fcmgt(xa::PRegS(cmpDstIdx),
                                        xa::PReg(cmpMaskIdx) / xa::T_z,
                                        xa::ZRegS(cmpSrcIdx),
                                        xa::ZRegS(cmpSrc2Idx));
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
                    //vblendmps(accvr | k_store_mask, accvr, inpvr);
                    if (vlen == 64) {
                        CG::sel(xa::ZRegS(IDX(accvr)),
                                xa::PReg(IDX(k_store_mask)) / xa::T_m,
                                xa::ZRegS(IDX(inpvr)), xa::ZRegS(IDX(accvr)));
                    } else if (vlen == 32) {
                        CG::sel(xa::ZRegS(IDX(accvr)),
                                xa::PReg(IDX(k_store_mask)) / xa::T_m,
                                xa::ZRegS(IDX(inpvr)), xa::ZRegS(IDX(accvr)));
                        CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_256 / xa::T_m, 0);
                    } else if (vlen == 16) {
                        CG::sel(xa::ZRegS(IDX(accvr)),
                                xa::PReg(IDX(k_store_mask)) / xa::T_m,
                                xa::ZRegS(IDX(inpvr)), xa::ZRegS(IDX(accvr)));
                        CG::mov(xa::ZReg(IDX(accvr)).s, P_MSB_384 / xa::T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
                    if (jpp.is_training) {
                        //vblendmps(indvr | k_store_mask, indvr, vmm_k_offset);
                        if (vlen == 64) {
                            CG::sel(xa::ZRegS(IDX(indvr)),
                                    xa::PReg(IDX(k_store_mask)) / xa::T_m,
                                    xa::ZRegS(IDX(vmm_k_offset)),
                                    xa::ZRegS(IDX(indvr)));
                        } else if (vlen == 32) {
                            CG::sel(xa::ZRegS(IDX(indvr)),
                                    xa::PReg(IDX(k_store_mask)) / xa::T_m,
                                    xa::ZRegS(IDX(vmm_k_offset)),
                                    xa::ZRegS(IDX(indvr)));
                            CG::mov(xa::ZReg(IDX(indvr)).s, P_MSB_256 / xa::T_m,
                                    0);
                        } else if (vlen == 16) {
                            CG::sel(xa::ZRegS(IDX(indvr)),
                                    xa::PReg(IDX(k_store_mask)) / xa::T_m,
                                    xa::ZRegS(IDX(vmm_k_offset)),
                                    xa::ZRegS(IDX(indvr)));
                            CG::mov(xa::ZReg(IDX(indvr)).s, P_MSB_384 / xa::T_m,
                                    0);
                        } else {
                            assert(!"unreachable");
                        }
                    }
                }
            }
            if (jpp.is_training) {
                if (with_c_tail_proccessing && isa == avx) {
                    push_vmm_val(vmm_c_tail_mask.getIdx());
                    put_one_in_vmm();
                }

                if (isa == avx && !mayiuse(avx2)) {
                    avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
                } else {
                    //uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                                xa::ZReg(IDX(vmm_k_offset)).s,
                                xa::ZReg(IDX(vmm_one)).s);
                    } else if (vlen == 32) {
                        CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                                xa::ZReg(IDX(vmm_k_offset)).s,
                                xa::ZReg(IDX(vmm_one)).s);
                    } else if (vlen == 16) {
                        CG::add(xa::VReg(IDX(vmm_k_offset)).s4,
                                xa::VReg(IDX(vmm_k_offset)).s4,
                                xa::VReg(IDX(vmm_one)).s4);
                        CG::mov(xa::ZReg(IDX(vmm_k_offset)).s,
                                P_MSB_256 / xa::T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
                }

                if (with_c_tail_proccessing && isa == avx)
                    pop_vmm_val(vmm_c_tail_mask.getIdx());
            }
        }
        //add(aux_reg_input, jpp.dt_size * iw * c_off);
        CG::add_imm(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(aux_reg_input)),
                (jpp.dt_size * iw * c_off), x_tmp_0);
        //inc(kj);
        CG::adds(xa::XReg(IDX(kj)), xa::XReg(IDX(kj)), 1);
        //cmp(kj, reg_kh);
        CG::cmp(xa::XReg(IDX(kj)), xa::XReg(IDX(reg_kh)));
        //jl(kh_label, T_NEAR);
        CG::b(xa::LT, kh_label);
    }

    if (jpp.ndims == 5) {
        //add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        CG::add_imm(xa::XReg(IDX(aux_reg_input_d)),
                xa::XReg(IDX(aux_reg_input_d)),
                (jpp.dt_size * jpp.ih * iw * c_off), x_tmp_0);
        if (jpp.is_training) {
            //mov(tmp_gpr, ptr[reg_param + GET_OFF(kd_padding_shift)]);
            //get mem address
            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)),
                    GET_OFF(kd_padding_shift), x_tmp_0);
            CG::ldr(xa::XReg(IDX(tmp_gpr)), xa::ptr(x_tmp_addr));
            //movq(xmm_tmp, tmp_gpr);
            CG::ptrue(p_tmp0.d, xa::VL2);
            CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
            CG::ptrue(p_tmp0.d, xa::VL1);
            CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m,
                    xa::XReg(IDX(tmp_gpr)));
            //uni_vpbroadcastd(vmm_tmp, xmm_tmp);
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
                CG::mov(xa::ZRegS(IDX(vmm_tmp)), P_MSB_256 / xa::T_m, 0);
            } else if (vlen == 16) {
                CG::dup(xa::VReg4S(IDX(vmm_tmp)), xa::VReg4S(IDX(xmm_tmp))[0]);
                CG::mov(xa::ZRegS(IDX(vmm_tmp)), P_MSB_384 / xa::T_m, 0);
            } else {
                assert(!"unreachable");
            }
            if (isa == avx && !mayiuse(avx2)) {
                Xmm t(vmm_mask.getIdx());
                avx_vpadd1(vmm_k_offset, xmm_tmp, t);
            } else {
                //uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_tmp)).s);
                } else if (vlen == 32) {
                    CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_tmp)).s);
                } else if (vlen == 16) {
                    CG::add(xa::VReg(IDX(vmm_k_offset)).s4,
                            xa::VReg(IDX(vmm_k_offset)).s4,
                            xa::VReg(IDX(vmm_tmp)).s4);
                    CG::mov(xa::ZReg(IDX(vmm_k_offset)).s, P_MSB_384 / xa::T_m,
                            0);
                } else {
                    assert(!"unreachable");
                }
            }
        }

        //dec(ki);
        CG::subs(xa::XReg(IDX(ki)), xa::XReg(IDX(ki)), 1);
        //cmp(ki, 0);
        CG::mov_imm(x_tmp_0, 0);
        CG::cmp(xa::XReg(IDX(ki)), x_tmp_0);
        //jg(kd_label, T_NEAR);
        CG::b(xa::GT, kd_label);
        //pop(reg_output);
        CG::ldr(xa::XReg(IDX(reg_output)), xa::post_ptr(X_TRANSLATOR_STACK, 8));
        //pop(reg_input);
        CG::ldr(xa::XReg(IDX(reg_input)), xa::post_ptr(X_TRANSLATOR_STACK, 8));
    }

    if (with_c_tail_proccessing && jpp.is_c_padded && isa == sse41) {
        //mov(tmp_gpr, 0); // needed zero to fill padded tail
        CG::mov_imm(xa::XReg(IDX(tmp_gpr)), 0);
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        auto accr_i = reg_ind(0, bci, jj);
        auto accvr = vreg(accr_i);
        auto output_offset = jpp.dt_size * (jj * c_off + bci * c_block);
        if (jpp.is_bf16) {
            auto acczr = zreg(accr_i);
            auto accyr = yreg(accr_i);
            if (!isa_has_bf16(jpp.isa))
                bf16_emu_->vcvtneps2bf16(accyr, acczr);
            else
                vcvtneps2bf16(accyr, accvr);
        }
#ifdef DNNL_X64_IMPLEMENTATION
        store(reg_idx(accr_i), reg_output, output_offset,
                is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        store(reg_idx(accr_i), xreg_output, output_offset,
                is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

        if (jpp.is_training) {
            const size_t step_index = (jj * c_off + bci * c_block)
                    * types::data_type_size(jpp.ind_dt);

            auto indr_i = reg_ind(2, bci, jj);
            auto vr = vreg(indr_i);
            if (jpp.ind_dt == data_type::u8) {
                auto xr = xreg(indr_i);
                if (isa == sse41) {
                    for (int i = 0; i < (jpp.c_block / 2); ++i) {
                        if (is_tail_processing(bci)
                                && i + (sse_high_half ? (jpp.c_block / 2) : 0)
                                        >= jpp.c_tail) {
                            if (jpp.is_c_padded)
                                mov(ptr[reg_index + step_index + i],
                                        tmp_gpr.cvt8()); // fill padded tail with zeros
                            else
                                break; // tail end
                        } else {
                            // bytes which should be stored are located in
                            // least significant bits(8 to be precise) of 32 bits parts
                            // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                            pextrb(ptr[reg_index + step_index + i], xr, 4 * i);
                        }
                    }
                } else if (isa == avx) {
                    auto yr = yreg(indr_i);
                    if (is_tail_processing(bci) && !jpp.is_c_padded) {
                        const int max_nr_of_vals
                                = jpp.c_tail > (jpp.c_block / 2)
                                ? (jpp.c_block / 2)
                                : jpp.c_tail;
                        for (int i = 0; i < max_nr_of_vals; ++i) {
                            // bytes which should be stored are located in
                            // least significant bits(8 to be precise) of 32 bits parts
                            // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                            vpextrb(ptr[reg_index + step_index + i], xr, 4 * i);
                        }

                        if (jpp.c_tail > (jpp.c_block / 2)) {
                            Xmm higher_128bits(vmm_mask.getIdx());
                            vextractf128(higher_128bits, yr, 1);
                            for (int i = 0; i < jpp.c_tail - (jpp.c_block / 2);
                                    ++i) {
                                // bytes which should be stored are located in
                                // least significant bits(8 to be precise) of 32 bits parts
                                // of xmm thus we need to store 0, 4, 8 and 12 byte of xmm
                                vpextrb(ptr[reg_index + step_index
                                                + (jpp.c_block / 2) + i],
                                        higher_128bits, 4 * i);
                            }
                        }
                    } else {
                        if (is_tail_processing(bci)) {
                            assert(jpp.is_c_padded);
                            vandps(yr, yr, vmm_c_tail_mask);
                        }
                        if (jj == 0) {
                            vmovd(xmm_tmp, reg_shuf_mask);
                            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
                        }
                        if (mayiuse(avx2)) {
                            vpshufb(yr, yr, vmm_tmp);
                            vmovd(ptr[reg_index + step_index], xr);
                            vperm2i128(yr, yr, yr, 0x1u);
                            vmovd(ptr[reg_index + step_index
                                          + (jpp.c_block / 2)],
                                    xr);
                        } else {
                            Xmm t(vmm_mask.getIdx());
                            vextractf128(t, yr, 0);
                            vpshufb(t, t, xmm_tmp);
                            vmovd(ptr[reg_index + step_index], t);
                            vextractf128(t, yr, 1);
                            vpshufb(t, t,
                                    xmm_tmp); // ymm_tmp[:128]==ymm_tmp[127:0]
                            vmovd(ptr[reg_index + step_index
                                          + (jpp.c_block / 2)],
                                    t);
                        }
                    }
                } else {
                    if (is_tail_processing(bci)) {
                        if (jpp.is_c_padded) {
#ifdef DNNL_INDIRECT_JIT_AARCH64
#else
                            knotw(k_c_tail_mask, k_c_tail_mask);
                            vpxord(vr | k_c_tail_mask, vr, vr);
                            knotw(k_c_tail_mask, k_c_tail_mask);
#endif
#ifdef DNNL_X64_IMPLEMENTATION
                            vpmovusdb(ptr[reg_index + step_index], vr);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                            /* get mem address */
                            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_index)),
                                    step_index, x_tmp_0);
                            int vlen = cpu_isa_traits<isa>::vlen;
                            if (vlen == 64) {
                                //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                                CG::mov(z_tmp0.d, xa::ZRegD(IDX(vr)));
                                CG::umin(z_tmp0.s, 255);
                                CG::st1b(z_tmp0.s, p_512, xa::ptr(x_tmp_addr));
                            } else if (vlen == 32) {
                                assert(!"unreachable");
                            } else if (vlen == 16) {
                                assert(!"unreachable");
                            } else {
                                assert(!"unreachable");
                            }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                        } else
#ifdef DNNL_X64_IMPLEMENTATION
                            vpmovusdb(ptr[reg_index + step_index],
                                    vr | k_c_tail_mask);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                        {
                            //get mem address
                            CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_index)),
                                    step_index, x_tmp_0);
                            int vlen = cpu_isa_traits<isa>::vlen;
                            if (vlen == 64) {
                                CG::mov(z_tmp0.d, xa::ZRegD(IDX(vr)));
                                CG::umin(z_tmp0.s, 255);
                                CG::st1b(z_tmp0.s, xa::PReg(IDX(k_c_tail_mask)),
                                        xa::ptr(x_tmp_addr));
                            } else if (vlen == 32) {
                                assert(!"unreachable");
                            } else if (vlen == 16) {
                                assert(!"unreachable");
                            } else {
                                assert(!"unreachable");
                            }
                        }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                    } else {
#ifdef DNNL_X64_IMPLEMENTATION
                        vpmovusdb(ptr[reg_index + step_index], vr);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                        /* get mem address */
                        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_index)),
                                step_index, x_tmp_0);
                        int vlen = cpu_isa_traits<isa>::vlen;
                        if (vlen == 64) {
                            //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                            CG::mov(z_tmp0.d, xa::ZRegD(IDX(vr)));
                            CG::umin(z_tmp0.s, 255);
                            CG::st1b(z_tmp0.s, p_512, xa::ptr(x_tmp_addr));
                        } else if (vlen == 32) {
                            assert(!"unreachable");
                        } else if (vlen == 16) {
                            assert(!"unreachable");
                        } else {
                            assert(!"unreachable");
                        }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                    }
                }
            } else {
#ifdef DNNL_X64_IMPLEMENTATION
                store(vr.getIdx(), reg_index, step_index,
                        is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                store(vr.getIdx(), xreg_index, step_index,
                        is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
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
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;
    Label kd_label, kh_label;

    auto reg_ind = [&](int shift, int bc, int j) {
        return shift * ur_bc * ur_w + bc * ur_w + j;
    };
    auto is_tail_processing = [&](int bc) {
        if (isa == sse41) {
            return with_c_tail_proccessing && bc == (ur_bc - 1)
                    && ((jpp.c_tail > (jpp.c_block / 2) && sse_high_half)
                            || (jpp.c_tail < (jpp.c_block / 2)
                                    && !sse_high_half)
                            || (jpp.c_tail == (jpp.c_block / 2) && sse_high_half
                                    && jpp.is_c_padded));
        } else
            return with_c_tail_proccessing && bc == (ur_bc - 1);
    };

    for_(int jj = 0; jj < ur_w; jj++)
    for (int bci = 0; bci < ur_bc; bci++) {
        auto outr_i = reg_ind(0, bci, jj);
        auto out_offset = jpp.dt_size * (jj * c_off + bci * c_block);
#ifdef DNNL_X64_IMPLEMENTATION
        load(reg_idx(outr_i), reg_output, out_offset, is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        load(reg_idx(outr_i), xreg_output, out_offset, is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        const size_t step_index = (jj * c_off + bci * c_block)
                * types::data_type_size(jpp.ind_dt);

        auto indr_i = reg_ind(1, bci, jj);
        auto indvr = vreg(indr_i);
        if (jpp.ind_dt == data_type::u8) {
            auto indxr = xreg(indr_i);
            if (isa == sse41) {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2); i++)
                        pinsrb(indxr, ptr[reg_index + step_index + i], i);
                } else {
                    movd(indxr, ptr[reg_index + step_index]);
                }
                pmovzxbd(indvr, indxr);
            } else if (isa == avx) {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
                    for (int i = 0; i < jpp.c_tail; i++)
                        vpinsrb(indxr, indxr, ptr[reg_index + step_index + i],
                                i);
                } else {
                    vmovq(indxr, ptr[reg_index + step_index]);
                }
                if (!mayiuse(avx2)) {
                    avx_pmovzxbd(indvr, indxr, xmm_tmp);
                } else {
                    vpmovzxbd(indvr, indxr);
                }
            } else {
                if (is_tail_processing(bci) && !jpp.is_c_padded) {
#ifndef DNNL_INDIRECT_JIT_AARCH64
                    vpmovzxbd(indvr | k_c_tail_mask | T_z,
                            ptr[reg_index + step_index]);
#else //#ifndef DNNL_INDIRECT_JIT_AARCH64
                    xa::ZReg z_indvr(IDX(indvr));
                    CG::pfalse(p_tmp1.b);
                    /* 32-bit context -> 16-bit conext */
                    CG::uzp1(p_tmp0.b, xa::PRegB(IDX(k_c_tail_mask)), p_tmp1.b);
                    /* 16-bit context -> 8-bit conext */
                    CG::uzp1(p_tmp0.b, p_tmp0.b, p_tmp1.b);
                    CG::add_imm(X_DEFAULT_ADDR, xa::XReg(IDX(reg_index)),
                            step_index, X_TMP_0);
                    CG::ld1b(z_indvr.b, p_tmp0 / xa::T_z,
                            xa::ptr(X_DEFAULT_ADDR));
                    CG::zip1(z_indvr.b, z_indvr.b, z_tmp0.b);
                    CG::zip1(z_indvr.h, z_indvr.h, z_tmp0.h);
                    CG::uxtb(xa::ZRegS(IDX(indvr)),
                            xa::PReg(IDX(k_c_tail_mask)) / xa::T_m, z_indvr.s);
#endif //#ifndef DNNL_INDIRECT_JIT_AARCH64
                } else {
#ifdef DNNL_X64_IMPLEMENTATION
                    vpmovzxbd(indvr, ptr[reg_index + step_index]);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    /* get mem address */
                    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_index)),
                            step_index, x_tmp_0);
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        CG::ldr(xa::QReg(IDX(z_tmp0)), xa::ptr(x_tmp_addr));
                        CG::zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                        CG::zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                        CG::uxtb(xa::ZReg(IDX(indvr)).s, p_512 / xa::T_m,
                                z_tmp0.s);
                    } else if (vlen == 32) {
                        CG::ldr(xa::QReg(IDX(z_tmp0)), xa::ptr(x_tmp_addr));
                        CG::zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                        CG::zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                        CG::uxtb(xa::ZReg(IDX(indvr)).s, p_512 / xa::T_m,
                                z_tmp0.s);
                        CG::mov(xa::ZReg(IDX(indvr)).s, P_MSB_256 / xa::T_m, 0);
                    } else if (vlen == 16) {
                        CG::ldr(xa::QReg(IDX(z_tmp0)), xa::ptr(x_tmp_addr));
                        CG::zip1(z_tmp0.b, z_tmp0.b, z_tmp0.b);
                        CG::zip1(z_tmp0.h, z_tmp0.h, z_tmp0.h);
                        //CG::mov(p_tmp0.b, P_ALL_ONE.b);
                        CG::uxtb(xa::ZReg(IDX(indvr)).s, p_512 / xa::T_m,
                                z_tmp0.s);
                        CG::mov(xa::ZReg(IDX(indvr)).s, P_MSB_384 / xa::T_m, 0);
                    } else {
                        assert(!"unreachable");
                    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                }
            }
        } else {
#ifdef DNNL_X64_IMPLEMENTATION
            load(indvr.getIdx(), reg_index, step_index,
                    is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            load(indvr.getIdx(), xreg_index, step_index,
                    is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        }
    }
#ifdef DNNL_X64_IMPLEMENTATION
    movq(xmm_tmp, reg_k_shift);
    uni_vpbroadcastd(vmm_k_offset, xmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    CG::ptrue(p_tmp0.d, xa::VL2);
    CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
    CG::ptrue(p_tmp0.d, xa::VL1);
    CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m,
            xa::XReg(IDX(reg_k_shift)));
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        CG::dup(xa::ZRegS(IDX(vmm_k_offset)), xa::ZRegS(IDX(xmm_tmp))[0]);
    } else if (vlen == 32) {
        CG::dup(xa::ZRegS(IDX(vmm_k_offset)), xa::ZRegS(IDX(xmm_tmp))[0]);
        CG::mov(xa::ZRegS(IDX(vmm_k_offset)), P_MSB_256 / xa::T_m, 0);
    } else if (vlen == 16) {
        CG::dup(xa::VReg4S(IDX(vmm_k_offset)), xa::VReg4S(IDX(xmm_tmp))[0]);
        CG::mov(xa::ZRegS(IDX(vmm_k_offset)), P_MSB_384 / xa::T_m, 0);
    } else {
        assert(!"unreachable");
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

    if (jpp.simple_alg && jpp.ndims == 5) {
#ifdef DNNL_X64_IMPLEMENTATION
        push(reg_input);
        push(reg_output);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        CG::str(xa::XReg(IDX(reg_input)), xa::pre_ptr(X_TRANSLATOR_STACK, -8));
        CG::str(xa::XReg(IDX(reg_output)), xa::pre_ptr(X_TRANSLATOR_STACK, -8));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
#ifdef DNNL_X64_IMPLEMENTATION
            push(dst_ptr);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            CG::str(xa::XReg(IDX(dst_ptr)),
                    xa::pre_ptr(X_TRANSLATOR_STACK, -8));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        }
#ifdef DNNL_X64_IMPLEMENTATION
        mov(aux_reg_input_d, reg_input);
        mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        mov(reg_kd_pad_shift, ptr[reg_param + GET_OFF(kd_padding_shift)]);
        L(kd_label);
        mov(aux_reg_input, aux_reg_input_d);
    } else {
        mov(aux_reg_input, reg_input);
    }

    xor_(kj, kj);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* mov(aux_reg_input_d, reg_input); */
        CG::mov(xa::XReg(IDX(aux_reg_input_d)), xa::XReg(IDX(reg_input)));
        //mov(ki, ptr[reg_param + GET_OFF(kd_padding)]);
        //get mem address
        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(kd_padding),
                x_tmp_0);
        CG::ldr(xa::XReg(IDX(ki)), xa::ptr(x_tmp_addr));
        //mov(reg_kd_pad_shift, ptr[reg_param + GET_OFF(kd_padding_shift)]);
        //get mem address
        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)),
                GET_OFF(kd_padding_shift), x_tmp_0);
        CG::ldr(xa::XReg(IDX(reg_kd_pad_shift)), xa::ptr(x_tmp_addr));
        L(kd_label);
        //mov(aux_reg_input, aux_reg_input_d);
        CG::mov(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(aux_reg_input_d)));
    } else {
        //mov(aux_reg_input, reg_input);
        CG::mov(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(reg_input)));
    }

    //xor_(kj, kj);
    CG::eor(xa::XReg(IDX(kj)), xa::XReg(IDX(kj)), xa::XReg(IDX(kj)));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, utils::div_up(pad_l - ki, stride_w));
            int jj_end = ur_w
                    - utils::div_up(
                            nstl::max(0, ki + pad_r - (kw - 1)), stride_w);
            for_(int jj = jj_start; jj < jj_end; jj++)
            for (int bci = 0; bci < ur_bc; bci++) {
                auto outvr = vreg(reg_ind(0, bci, jj));
                auto indvr = vreg(reg_ind(1, bci, jj));
                auto inpr_i = reg_ind(2, bci, jj);
                auto inpvr = vreg(inpr_i);
                auto cvtvr = vreg(reg_ind(3, bci, jj));
                int aux_inp_offset
                        = (ki + jj * stride_w - pad_l) * c_off + bci * c_block;
                if (aux_inp_offset >= iw * c_off) continue;
                int inp_offset = jpp.dt_size * aux_inp_offset;
#ifdef DNNL_X64_IMPLEMENTATION
                load(reg_idx(inpr_i), aux_reg_input, inp_offset,
                        is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                load(reg_idx(inpr_i), aux_xreg_input, inp_offset,
                        is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                if (isa == sse41) {
                    mov(dst_ptr, aux_reg_input);
                    add(dst_ptr, inp_offset);

                    movups(cvtvr, indvr);
                    pcmpeqd(cvtvr, vmm_k_offset);
                    addps(inpvr, outvr);
                    if (is_tail_processing(bci)) {
                        Label end_cond_move[4];
                        for (int i = 0; i < jpp.c_tail % (jpp.c_block / 2);
                                i++) {
                            pextrd(tmp_gpr.cvt32(), cvtvr, i);
                            cmp(tmp_gpr, 0);
                            je(end_cond_move[i], T_NEAR);
                            pextrd(ptr[dst_ptr + i * jpp.dt_size], inpvr, i);
                            L(end_cond_move[i]);
                        }
                    } else
                        maskmovdqu(inpvr, cvtvr);
                } else if (isa == avx) {
                    if (mayiuse(avx2)) {
                        vpcmpeqd(cvtvr, indvr, vmm_k_offset);
                    } else {
                        avx_pcmpeqd(cvtvr, indvr, vmm_k_offset, xmm_tmp);
                    }
                    vaddps(inpvr, inpvr, outvr);
                    if (is_tail_processing(bci)) {
                        vandps(cvtvr, cvtvr, vmm_c_tail_mask);
                    }
                    vmaskmovps(
                            vmmword[aux_reg_input + inp_offset], cvtvr, inpvr);
                } else {
                    auto indzr = zreg(inpr_i);
                    auto indyr = yreg(inpr_i);
#ifdef DNNL_X64_IMPLEMENTATION
                    vpcmpeqd(k_store_mask, indvr, vmm_k_offset);
                    vblendmps(vmm_tmp | k_store_mask | T_z, outvr, outvr);
                    vaddps(inpvr, inpvr, vmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    /* vpcmpeqd(k_store_mask, indvr, vmm_k_offset); */
                    CG::cmpeq(xa::PRegS(IDX(k_store_mask)), p_lsb / xa::T_z,
                            xa::ZRegS(IDX(indvr)),
                            xa::ZRegS(IDX(vmm_k_offset)));
                    //vblendmps(vmm_tmp | k_store_mask | T_z, outvr, outvr);
                    //vaddps(inpvr, inpvr, vmm_tmp);
                    CG::not_(p_tmp0.b, P_ALL_ONE.b,
                            xa::PRegB(IDX(k_store_mask)));
                    int vlen = cpu_isa_traits<isa>::vlen;
                    if (vlen == 64) {
                        CG::mov(xa::ZRegD(IDX(vmm_tmp)), xa::ZRegD(IDX(outvr)));
                        CG::mov(xa::ZReg(IDX(vmm_tmp)).s, p_tmp0 / xa::T_m, 0);
                        CG::fadd(xa::ZReg(IDX(inpvr)).s, xa::ZReg(IDX(inpvr)).s,
                                xa::ZReg(IDX(vmm_tmp)).s);
                    } else if (vlen == 32) {
                        CG::mov(xa::ZRegD(IDX(vmm_tmp)), xa::ZRegD(IDX(outvr)));
                        CG::mov(xa::ZReg(IDX(vmm_tmp)).s, p_tmp0 / xa::T_m, 0);
                        CG::fadd(xa::ZReg(IDX(inpvr)).s, xa::ZReg(IDX(inpvr)).s,
                                xa::ZReg(IDX(vmm_tmp)).s);
                        CG::mov(xa::ZReg(IDX(inpvr)).s, P_MSB_256 / xa::T_m, 0);
                    } else if (vlen == 16) {
                        CG::mov(xa::VReg16B(IDX(vmm_tmp)),
                                xa::VReg16B(IDX(outvr)));
                        CG::mov(xa::ZReg(IDX(vmm_tmp)).s, p_tmp0 / xa::T_m, 0);
                        CG::fadd(xa::VReg(IDX(inpvr)).s4,
                                xa::VReg(IDX(inpvr)).s4,
                                xa::VReg(IDX(vmm_tmp)).s4);
                    } else {
                        assert(!"unreachable");
                    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

                    if (jpp.is_bf16) {
                        if (!isa_has_bf16(jpp.isa))
                            bf16_emu_->vcvtneps2bf16(indyr, indzr);
                        else
                            vcvtneps2bf16(indyr, inpvr);
                    }
#ifdef DNNL_X64_IMPLEMENTATION
                    store(inpvr.getIdx(), aux_reg_input, inp_offset,
                            is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    store(inpvr.getIdx(), aux_xreg_input, inp_offset,
                            is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                }
            }

            if (with_c_tail_proccessing && isa == avx) {
                push_vmm_val(vmm_c_tail_mask.getIdx());
                put_one_in_vmm();
            }

            if (isa == avx && !mayiuse(avx2)) {
                avx_vpadd1(vmm_k_offset, vmm_one, xmm_tmp);
            } else {
#ifdef DNNL_X64_IMPLEMENTATION
                uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                /* uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_one); */
                int vlen = cpu_isa_traits<isa>::vlen;
                if (vlen == 64) {
                    CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_one)).s);
                } else if (vlen == 32) {
                    CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_k_offset)).s,
                            xa::ZReg(IDX(vmm_one)).s);
                } else if (vlen == 16) {
                    CG::add(xa::VReg(IDX(vmm_k_offset)).s4,
                            xa::VReg(IDX(vmm_k_offset)).s4,
                            xa::VReg(IDX(vmm_one)).s4);
                    CG::mov(xa::ZReg(IDX(vmm_k_offset)).s, P_MSB_384 / xa::T_m,
                            0);
                } else {
                    assert(!"unreachable");
                }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
            }

            if (with_c_tail_proccessing && isa == avx)
                pop_vmm_val(vmm_c_tail_mask.getIdx());
        }

#ifdef DNNL_X64_IMPLEMENTATION
        add(aux_reg_input, jpp.dt_size * iw * c_off);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_label, T_NEAR);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);

        mov(tmp_gpr, reg_kd_pad_shift);
        movq(xmm_tmp, tmp_gpr);
        uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        if (isa == avx && !mayiuse(avx2)) {
            Xmm t(vmm_mask.getIdx());
            avx_vpadd1(vmm_k_offset, vmm_tmp, t);
        } else {
            uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
        }

        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            pop(dst_ptr);
        }
        pop(reg_output);
        pop(reg_input);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* add(aux_reg_input, jpp.dt_size * iw * c_off); */
        CG::add_imm(xa::XReg(IDX(aux_reg_input)), xa::XReg(IDX(aux_reg_input)),
                (jpp.dt_size * iw * c_off), x_tmp_0);
        //inc(kj);
        CG::adds(xa::XReg(IDX(kj)), xa::XReg(IDX(kj)), 1);
        //cmp(kj, reg_kh);
        CG::cmp(xa::XReg(IDX(kj)), xa::XReg(IDX(reg_kh)));
        //jl(kh_label, T_NEAR);
        CG::b(xa::LT, kh_label);
    }
    if (jpp.simple_alg && jpp.ndims == 5) {
        //add(aux_reg_input_d, jpp.dt_size * jpp.ih * iw * c_off);
        CG::add_imm(xa::XReg(IDX(aux_reg_input_d)),
                xa::XReg(IDX(aux_reg_input_d)),
                (jpp.dt_size * jpp.ih * iw * c_off), x_tmp_0);
        //mov(tmp_gpr, reg_kd_pad_shift);
        CG::mov(xa::XReg(IDX(tmp_gpr)), xa::XReg(IDX(reg_kd_pad_shift)));
        //movq(xmm_tmp, tmp_gpr);
        CG::ptrue(p_tmp0.d, xa::VL2);
        CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
        CG::ptrue(p_tmp0.d, xa::VL1);
        CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m,
                xa::XReg(IDX(tmp_gpr)));
        //uni_vpbroadcastd(vmm_tmp, xmm_tmp);
        const int vlen = cpu_isa_traits<isa>::vlen;
        if (vlen == 64) {
            CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
        } else if (vlen == 32) {
            CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
            CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_256 / xa::T_m, 0);
        } else if (vlen == 16) {
            CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
            CG::mov(xa::ZReg(IDX(vmm_tmp)).s, P_MSB_384 / xa::T_m, 0);
        } else {
            assert(!"unreachable");
        }
        if (isa == avx && !mayiuse(avx2)) {
            Xmm t(vmm_mask.getIdx());
            avx_vpadd1(vmm_k_offset, vmm_tmp, t);
        } else {
            //uni_vpaddd(vmm_k_offset, vmm_k_offset, vmm_tmp);
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                        xa::ZReg(IDX(vmm_k_offset)).s,
                        xa::ZReg(IDX(vmm_tmp)).s);
            } else if (vlen == 32) {
                CG::add(xa::ZReg(IDX(vmm_k_offset)).s,
                        xa::ZReg(IDX(vmm_k_offset)).s,
                        xa::ZReg(IDX(vmm_tmp)).s);
            } else if (vlen == 16) {
                CG::add(xa::VReg(IDX(vmm_k_offset)).s4,
                        xa::VReg(IDX(vmm_k_offset)).s4,
                        xa::VReg(IDX(vmm_tmp)).s4);
                CG::mov(xa::ZReg(IDX(vmm_k_offset)).s, P_MSB_384 / xa::T_m, 0);
            } else {
                assert(!"unreachable");
            }
        }

        //dec(ki);
        CG::subs(xa::XReg(IDX(ki)), xa::XReg(IDX(ki)), 1);
        //cmp(ki, 0);
        CG::mov_imm(x_tmp_0, 0);
        CG::cmp(xa::XReg(IDX(ki)), x_tmp_0);
        //jg(kd_label, T_NEAR);
        CG::b(xa::GT, kd_label);
        if (isa == sse41) {
            // Save rdi since it is used in maskmovdqu
            assert(dst_ptr == rdi);
            pop(dst_ptr);
        }
        //pop(reg_output);
        CG::ldr(xa::XReg(IDX(reg_output)), xa::post_ptr(X_TRANSLATOR_STACK, 8));
        //pop(reg_input);
        CG::ldr(xa::XReg(IDX(reg_input)), xa::post_ptr(X_TRANSLATOR_STACK, 8));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    }
}

template <cpu_isa_t isa>
void jit_uni_pool_kernel<isa>::zero_diff_src(
        int ur_bc, bool with_c_tail_proccessing) {
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : jpp.c_block;

    Label l_skip, l_ih_loop, l_id_loop;

    auto is_tail_processing = [&](int bc) {
        return with_c_tail_proccessing && bc == (ur_bc - 1);
    };
#ifdef DNNL_X64_IMPLEMENTATION
    mov(reg_zero_id, ptr[reg_param + GET_OFF(zero_id)]);
    cmp(reg_zero_id, 0);
    jz(l_skip, T_NEAR);

    mov(reg_zero_ih, ptr[reg_param + GET_OFF(zero_ih)]);
    cmp(reg_zero_ih, 0);
    jz(l_skip, T_NEAR);

    mov(reg_zero_ptr, ptr[reg_param + GET_OFF(zero_ptr)]);

    Vmm vzero = vmm_tmp;
    uni_vpxor(vzero, vzero, vzero);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //mov(reg_zero_id, ptr[reg_param + GET_OFF(zero_id)]);
    //get mem address
    CG::add_imm(
            x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(zero_id), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_zero_id)), xa::ptr(x_tmp_addr));
    //cmp(reg_zero_id, 0);
    CG::mov_imm(x_tmp_0, 0);
    CG::cmp(xa::XReg(IDX(reg_zero_id)), x_tmp_0);
    //jz(l_skip, T_NEAR);
    CG::b(xa::EQ, l_skip);
    //mov(reg_zero_ih, ptr[reg_param + GET_OFF(zero_ih)]);
    //get mem address
    CG::add_imm(
            x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(zero_ih), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_zero_ih)), xa::ptr(x_tmp_addr));
    //cmp(reg_zero_ih, 0);
    CG::mov_imm(x_tmp_0, 0);
    CG::cmp(xa::XReg(IDX(reg_zero_ih)), x_tmp_0);
    //jz(l_skip, T_NEAR);
    CG::b(xa::EQ, l_skip);

    //mov(reg_zero_ptr, ptr[reg_param + GET_OFF(zero_ptr)]);
    //get mem address
    CG::add_imm(
            x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(zero_ptr), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_zero_ptr)), xa::ptr(x_tmp_addr));

    Vmm vzero = vmm_tmp;
    //uni_vpxor(vzero, vzero, vzero);
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        CG::eor(xa::ZReg(IDX(vzero)).d, xa::ZReg(IDX(vzero)).d,
                xa::ZReg(IDX(vzero)).d);
    } else if (vlen == 32) {
        CG::eor(xa::ZRegD(IDX(vzero)), xa::ZRegD(IDX(vzero)),
                xa::ZRegD(IDX(vzero)));
        CG::mov(xa::ZRegS(IDX(vzero)), P_MSB_256 / xa::T_m, 0);
    } else if (vlen == 16) {
        CG::eor(xa::VReg16B(IDX(vzero)), xa::VReg16B(IDX(vzero)),
                xa::VReg16B(IDX(vzero)));
    } else {
        assert(!"unreachable");
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

    const int width_size = jpp.iw * c_off * jpp.dt_size;

    auto aux_reg_zero_ptr = tmp_gpr;

    L(l_id_loop);
    {
#ifdef DNNL_X64_IMPLEMENTATION
        mov(aux_reg_zero_ptr, reg_zero_ptr);
        mov(aux_reg_zero_ih, reg_zero_ih);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* mov(aux_reg_zero_ptr, reg_zero_ptr); */
        CG::mov(xa::XReg(IDX(aux_reg_zero_ptr)), xa::XReg(IDX(reg_zero_ptr)));
        //mov(aux_reg_zero_ih, reg_zero_ih);
        CG::mov(xa::XReg(IDX(aux_reg_zero_ih)), xa::XReg(IDX(reg_zero_ih)));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        L(l_ih_loop);
        {
            const auto vlen = cpu_isa_traits<isa>::vlen;
            const int step = c_off * jpp.dt_size;

            // TODO: maybe a big code generated here
            for_(int i = 0; i < width_size; i += step)
            for (int bci = 0; bci < ur_bc; bci++) {
                const int offs = i + bci * jpp.c_block * jpp.dt_size;
                if (isa == sse41) {
                    bool is_needed_c_tail_processing = false;
                    if (is_tail_processing(bci)
                            && jpp.c_tail < (jpp.c_block / 2))
                        is_needed_c_tail_processing = true;
#ifdef DNNL_X64_IMPLEMENTATION
                    store(vzero.getIdx(), reg_zero_ptr, offs,
                            is_needed_c_tail_processing);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    store(vzero.getIdx(), xreg_zero_ptr, offs,
                            is_needed_c_tail_processing);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

                    if (!is_tail_processing(bci)
                            || (is_tail_processing(bci)
                                    && (jpp.is_c_padded
                                            || jpp.c_tail
                                                    > (jpp.c_block / 2)))) {
#ifdef DNNL_X64_IMPLEMENTATION
                        store(vzero.getIdx(), reg_zero_ptr, offs + vlen,
                                is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                        store(vzero.getIdx(), xreg_zero_ptr, offs + vlen,
                                is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                    }

                } else {
#ifdef DNNL_X64_IMPLEMENTATION
                    store(vzero.getIdx(), reg_zero_ptr, offs,
                            is_tail_processing(bci));
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                    store(vzero.getIdx(), xreg_zero_ptr, offs,
                            is_tail_processing(bci));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
                }
            }
#ifdef DNNL_X64_IMPLEMENTATION
            add(reg_zero_ptr, width_size);
            dec(aux_reg_zero_ih);
            jnz(l_ih_loop, T_NEAR);
        }
        mov(reg_zero_ptr, aux_reg_zero_ptr);
        add(reg_zero_ptr, width_size * jpp.ih);
        dec(reg_zero_id);
        jnz(l_id_loop, T_NEAR);
    }
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            /* add(reg_zero_ptr, width_size); */
            CG::add_imm(xa::XReg(IDX(reg_zero_ptr)),
                    xa::XReg(IDX(reg_zero_ptr)), width_size, x_tmp_0);
            //dec(aux_reg_zero_ih);
            CG::subs(xa::XReg(IDX(aux_reg_zero_ih)),
                    xa::XReg(IDX(aux_reg_zero_ih)), 1);
            //jnz(l_ih_loop, T_NEAR);
            CG::b(xa::NE, l_ih_loop);
        }
        //mov(reg_zero_ptr, aux_reg_zero_ptr);
        CG::mov(xa::XReg(IDX(reg_zero_ptr)), xa::XReg(IDX(aux_reg_zero_ptr)));
        //add(reg_zero_ptr, width_size * jpp.ih);
        CG::add_imm(xa::XReg(IDX(reg_zero_ptr)), xa::XReg(IDX(reg_zero_ptr)),
                (width_size * jpp.ih), x_tmp_0);
        //dec(reg_zero_id);
        CG::subs(xa::XReg(IDX(reg_zero_id)), xa::XReg(IDX(reg_zero_id)), 1);
        //jnz(l_id_loop, T_NEAR);
        CG::b(xa::NE, l_id_loop);
    }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

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
    const int c_off = (jpp.tag_kind == jptg_nspc) ? jpp.c : c_block;

    int vlen = cpu_isa_traits<isa>::vlen;

#if defined(_WIN32)
    // Always mimic the Unix ABI (see the note about maskmovdqu in the header
    // file).
    xor_(rdi, rcx);
    xor_(rcx, rdi);
    xor_(rdi, rcx);
#endif

#ifndef DNNL_X64_IMPLEMENTATION
    CG::ptrue(p_512.b);
    CG::ptrue(p_256.b, xa::VL32);
    CG::ptrue(p_128.b, xa::VL16);
    if (cpu_isa_traits<isa>::vlen == 32) {
        p_lsb = p_256;
    } else if (cpu_isa_traits<isa>::vlen == 16) {
        p_lsb = p_128;
    }
#endif

    if (!isa_has_bf16(jpp.isa) && jpp.is_bf16) bf16_emu_->init_vcvtneps2bf16();

#ifdef DNNL_X64_IMPLEMENTATION
    mov(reg_input, ptr[reg_param + GET_OFF(src)]);
    mov(reg_output, ptr[reg_param + GET_OFF(dst)]);
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward))
        mov(reg_index, ptr[reg_param + GET_OFF(indices)]);
    mov(reg_kh, ptr[reg_param + GET_OFF(kh_padding)]);
    mov(reg_k_shift, ptr[reg_param + GET_OFF(kh_padding_shift)]);
    mov(reg_ker_area_h, ptr[reg_param + GET_OFF(ker_area_h)]);
    mov(reg_nbc, ptr[reg_param + GET_OFF(ur_bc)]);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
    //mov(reg_input, ptr[reg_param + GET_OFF(src)]);
    //get mem address
    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(src), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_input)), xa::ptr(x_tmp_addr));
    //mov(reg_output, ptr[reg_param + GET_OFF(dst)]);
    //get mem address
    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(dst), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_output)), xa::ptr(x_tmp_addr));
    if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
        //mov(reg_index, ptr[reg_param + GET_OFF(indices)]);
        //get mem address
        CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(indices),
                x_tmp_0);
        CG::ldr(xa::XReg(IDX(reg_index)), xa::ptr(x_tmp_addr));
    }
    //mov(reg_kh, ptr[reg_param + GET_OFF(kh_padding)]);
    //get mem address
    CG::add_imm(
            x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(kh_padding), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_kh)), xa::ptr(x_tmp_addr));
    //mov(reg_k_shift, ptr[reg_param + GET_OFF(kh_padding_shift)]);
    //get mem address
    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(kh_padding_shift),
            x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_k_shift)), xa::ptr(x_tmp_addr));
    //mov(reg_ker_area_h, ptr[reg_param + GET_OFF(ker_area_h)]);
    //get mem address
    CG::add_imm(
            x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(ker_area_h), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_ker_area_h)), xa::ptr(x_tmp_addr));
    //mov(reg_nbc, ptr[reg_param + GET_OFF(ur_bc)]);
    //get mem address
    CG::add_imm(x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(ur_bc), x_tmp_0);
    CG::ldr(xa::XReg(IDX(reg_nbc)), xa::ptr(x_tmp_addr));
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */

    if (jpp.is_bf16) {
        mov(tmp_gpr.cvt32(), 0xAAAAAAAA);
        kmovd(k_mask_cvt, tmp_gpr.cvt32());

        mov(tmp_gpr, idx_table);
        vmovups(vmm_idx(), ptr[tmp_gpr]);
    }

    int r_pad
            = nstl::max(0, calculate_end_padding(l_pad, ow, iw, stride_w, kw));

    auto process_oi = [&](int ur_w, int ur_bc, int lpad, int rpad,
                              bool with_c_tail_proccessing,
                              bool inc_reg = true) {
        step(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);

        if (isa == sse41) {
            if (with_c_tail_proccessing && !jpp.is_c_padded
                    && jpp.c_tail <= (jpp.c_block / 2)) {
                // In nspc format in case of c tail processing if c tail is
                // equal or lower than 4 we don't have to process
                // last high half block, because it doesn't exist
                ur_bc -= 1;
            }
            sse_high_half = true;
            step_high_half(ur_w, ur_bc, lpad, rpad, with_c_tail_proccessing);
            sse_high_half = false;
        }

        if (!inc_reg) return;

        auto dt_size = jpp.dt_size;
        auto shift = (isa == sse41) ? vlen : 0;
#ifdef DNNL_X64_IMPLEMENTATION
        add(reg_input, dt_size * (ur_w * stride_w - lpad) * c_off - shift);
        add(reg_output, dt_size * ur_w * c_off - shift);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* add(reg_input, dt_size * (ur_w * stride_w - lpad) * c_off - shift); */
        CG::add_imm(xa::XReg(IDX(reg_input)), xa::XReg(IDX(reg_input)),
                (dt_size * (ur_w * stride_w - lpad) * c_off - shift), x_tmp_0);
        //add(reg_output, dt_size * ur_w * c_off - shift);
        CG::add_imm(xa::XReg(IDX(reg_output)), xa::XReg(IDX(reg_output)),
                (dt_size * ur_w * c_off - shift), x_tmp_0);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            auto ishift = (isa == sse41) ? jpp.c_block / 2 : 0;
            auto ind_dt_size = types::data_type_size(jpp.ind_dt);
#ifdef DNNL_X64_IMPLEMENTATION
            add(reg_index, (ur_w * c_off - ishift) * ind_dt_size);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            /* add(reg_index, (ur_w * c_off - ishift) * ind_dt_size); */
            CG::add_imm(xa::XReg(IDX(reg_index)), xa::XReg(IDX(reg_index)),
                    ((ur_w * c_off - ishift) * ind_dt_size), x_tmp_0);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        }
    };

    auto perform_ker = [&](int ur_bc, bool with_c_tail_processing) {
        prev_kw = 0; // re-initialize this value for avg steps

        if (jpp.is_backward && jpp.simple_alg)
            zero_diff_src(ur_bc, with_c_tail_processing);

        if (jpp.alg == pooling_avg_exclude_padding
                && (!with_c_tail_processing || isa != avx)) {
            // vmm_ker_area_h and vmm_c_tail_mask are stored in one register
            // so when vmm_c_tail_mask is used we need to load vmm_ker_area_h
            // exactly where this information is needed with the
            // vmm_c_tail_mask information being saved first
            uni_broadcast_reg_val(
                    reg_ker_area_h.getIdx(), vmm_ker_area_h.getIdx());
        }

        if (jpp.alg == pooling_avg_include_padding) {
#ifdef DNNL_X64_IMPLEMENTATION
            mov(tmp_gpr, float2int((float)(kw * kh * jpp.kd)));
            movq(xmm_tmp, tmp_gpr);
            uni_vpbroadcastd(vmm_tmp, xmm_tmp);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
            /* mov(tmp_gpr, float2int((float)(kw * kh * jpp.kd))); */
            CG::mov_imm(xa::XReg(IDX(tmp_gpr)),
                    float2int((float)(kw * kh * jpp.kd)));
            //movq(xmm_tmp, tmp_gpr);
            CG::ptrue(p_tmp0.d, xa::VL2);
            CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m, 0);
            CG::ptrue(p_tmp0.d, xa::VL1);
            CG::mov(xa::ZRegD(IDX(xmm_tmp)), p_tmp0 / xa::T_m,
                    xa::XReg(IDX(tmp_gpr)));
            //uni_vpbroadcastd(vmm_tmp, xmm_tmp);
            int vlen = cpu_isa_traits<isa>::vlen;
            if (vlen == 64) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
            } else if (vlen == 32) {
                CG::dup(xa::ZRegS(IDX(vmm_tmp)), xa::ZRegS(IDX(xmm_tmp))[0]);
                CG::mov(xa::ZRegS(IDX(vmm_tmp)), P_MSB_256 / xa::T_m, 0);
            } else if (vlen == 16) {
                CG::dup(xa::VReg4S(IDX(vmm_tmp)), xa::VReg4S(IDX(xmm_tmp))[0]);
                CG::mov(xa::ZRegS(IDX(vmm_tmp)), P_MSB_384 / xa::T_m, 0);
            } else {
                assert(!"unreachable");
            }
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
        }

        if (jpp.alg == pooling_max && (jpp.is_training || jpp.is_backward)) {
            if (!with_c_tail_processing || isa != avx) {
                // The same situation as above(vmm_ker_area_h).
                put_one_in_vmm();
            }

            if (isa == avx) { mov(reg_shuf_mask, 0x0c080400); }
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

        xor_(oi_iter, oi_iter);
        if (n_oi > 0) {
            Label ow_loop;
            L(ow_loop);
            {
                process_oi(ur_w, ur_bc, 0, 0, with_c_tail_processing);

#ifdef DNNL_X64_IMPLEMENTATION
                inc(oi_iter);
                cmp(oi_iter, n_oi);
                jl(ow_loop, T_NEAR);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
                /* inc(oi_iter); */
                CG::adds(xa::XReg(IDX(oi_iter)), xa::XReg(IDX(oi_iter)), 1);
                //cmp(oi_iter, n_oi);
                CG::mov_imm(x_tmp_0, n_oi);
                CG::cmp(xa::XReg(IDX(oi_iter)), x_tmp_0);
                //jl(ow_loop, T_NEAR);
                CG::b(xa::LT, ow_loop);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
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
#ifdef DNNL_X64_IMPLEMENTATION
        cmp(reg_nbc, jpp.ur_bc);
        jne(ur_bc_tail_label, T_NEAR);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* cmp(reg_nbc, jpp.ur_bc); */
        CG::mov_imm(x_tmp_0, jpp.ur_bc);
        CG::cmp(xa::XReg(IDX(reg_nbc)), x_tmp_0);
        //jne(ur_bc_tail_label, T_NEAR);
        b(xa::NE, ur_bc_tail_label);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    } else if (jpp.c_tail != 0) {
        // ur_bc contains number of channel blocks to processing
        // b_c contains number of channel blocks already processed
        // If reg_nbc + tmp_gpr == jpp.nb_c then this is
        // information that probably channel tail processing will be needed.
#ifdef DNNL_X64_IMPLEMENTATION
        mov(tmp_gpr, ptr[reg_param + GET_OFF(b_c)]);
        add(tmp_gpr, reg_nbc);
        cmp(tmp_gpr, jpp.nb_c);
        je(c_tail_processing_label, T_NEAR);
#else /* #ifdef DNNL_X64_IMPLEMENTATION */
        /* mov(tmp_gpr, ptr[reg_param + GET_OFF(b_c)]); */
        /* get mem address */
        CG::add_imm(
                x_tmp_addr, xa::XReg(IDX(reg_param)), GET_OFF(b_c), x_tmp_0);
        CG::ldr(xa::XReg(IDX(tmp_gpr)), xa::ptr(x_tmp_addr));
        //add(tmp_gpr, reg_nbc);
        CG::add(xa::XReg(IDX(tmp_gpr)), xa::XReg(IDX(tmp_gpr)),
                xa::XReg(IDX(reg_nbc)));
        //cmp(tmp_gpr, jpp.nb_c);
        CG::mov_imm(x_tmp_0, jpp.nb_c);
        CG::cmp(xa::XReg(IDX(tmp_gpr)), x_tmp_0);
        //je(c_tail_processing_label, T_NEAR);
        b(Xbyak_aarch64::EQ, c_tail_processing_label);
#endif /* #ifdef DNNL_X64_IMPLEMENTATION */
    }

    perform_ker(jpp.ur_bc, false);

    if (jpp.ur_bc_tail > 0) {
        jmp(finish_label, T_NEAR);

        // If ur_bc_tail exists then we know that this is
        // last set of blocks to process and we need
        // care of c tail processing if number of channels
        // is not divided by number of channels in block
        L(ur_bc_tail_label);
        if (jpp.c_tail != 0) prepare_tail_mask();
        perform_ker(jpp.ur_bc_tail, jpp.c_tail != 0);

        L(finish_label);
    } else if (jpp.c_tail != 0) {
        jmp(finish_label, T_NEAR);

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
#ifdef DNNL_X64_IMPLEMENTATION
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            CodeArray::dw(_idx[i]);
#else //#ifdef DNNL_X64_IMPLEMENTATION
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            CodeArray::dw(_idx[i]);
        binCommit();
#endif //#ifdef DNNL_X64_IMPLEMENTATION
    }
}

template struct jit_uni_pool_kernel<sse41>;
template struct jit_uni_pool_kernel<avx>; // implements both <avx> and <avx2>
template struct jit_uni_pool_kernel<avx512_common>;
template struct jit_uni_pool_kernel<avx512_core>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
