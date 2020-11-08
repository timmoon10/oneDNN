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

//#ifndef CPU_X64_JIT_UNI_POOL_KERNEL_HPP
//#define CPU_X64_JIT_UNI_POOL_KERNEL_HPP
#ifndef CPU_AARCH64_JIT_UNI_POOL_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_POOL_KERNEL_HPP

#include <cfloat>
#include <functional>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
//#include "cpu/x64/jit_generator.hpp"
#include "cpu/aarch64/jit_generator.hpp"

//#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
//#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
//#include "cpu/x64/jit_primitive_conf.hpp"
//#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/jit_sve_512_core_bf16cvt.hpp"

//#include "cpu/x64/xbyak/xbyak.h"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

using namespace Xbyak_aarch64;
//using Xbyak = dnnl::impl::cpu::x64::Xbyak;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct jit_uni_pool_kernel : public jit_generator {

    jit_uni_pool_kernel(
            const jit_pool_conf_t &ajpp, const memory_desc_t *dst_md);
    jit_pool_conf_t jpp;
    ~jit_uni_pool_kernel();

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_pool_kernel)

    static status_t init_conf(jit_pool_conf_t &jbp,
            memory_tracking::registrar_t &scratchpad, const pooling_pd_t *ppd,
            int nthreads);

private:
    /*
    using Xmm = Xbyak::Xmm;
    using Ymm = Xbyak::Ymm;
    using Zmm = Xbyak::Zmm;
    using Opmask = Xbyak::Opmask;
    using Reg32 = Xbyak::Reg32;
    using Reg64 = Xbyak::Reg64;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    int vmm_idx_upper_bound() const noexcept {
        return utils::one_of(isa, avx512_common, avx512_core) ? 31 : 15;
    }
  */
    int vmm_idx_upper_bound() const noexcept { return 31; }

    int reg_idx(int idx) const noexcept { return vmm_idx_upper_bound() - idx; }

    /*
    Xmm xreg(int idx) const noexcept { return Xmm(reg_idx(idx)); }
    Ymm yreg(int idx) const noexcept { return Ymm(reg_idx(idx)); }
    Zmm zreg(int idx) const noexcept { return Zmm(reg_idx(idx)); }
    Vmm vreg(int idx) const noexcept { return Vmm(reg_idx(idx)); }
  */
    XReg xreg(int idx) const noexcept { return XReg(reg_idx(idx)); }
    ZReg yreg(int idx) const noexcept { return ZReg(reg_idx(idx)); }
    ZReg zreg(int idx) const noexcept { return ZReg(reg_idx(idx)); }
    VReg vreg(int idx) const noexcept { return VReg(reg_idx(idx)); }
    /*
    const Xbyak::AddressFrame &vmmword = (isa == sse41)
            ? xword
            : (isa == avx || isa == avx2) ? yword : zword;

    const Xbyak::AddressFrame &vmmword = (isa == asimd)
            ? xword
            : (isa == asimd) ? yword : zword;  
  */
    /*
    Xmm vmm_mask = Xmm(0);
    Ymm ymm_tmp_1 = Ymm(0);
    Vmm vmm_tmp_1 = Vmm(0);
  */
    XReg vmm_mask = XReg(0);
    ZReg ymm_tmp_1 = ZReg(0);
    VReg vmm_tmp_1 = VReg(0);

    // Used only for avx and if c tail is present
    /*
    Vmm vmm_c_tail_mask = Vmm(2);

    Xmm xmm_ker_area_h = Xmm(2);
    Xmm xmm_one = Xmm(2);
    Xmm xmm_tmp = Xmm(3);

    Vmm vmm_ker_area_h = Vmm(2);
    Vmm vmm_one = Vmm(2);
    Vmm vmm_tmp = Vmm(3);
    Ymm ymm_tmp = Ymm(3);

    Vmm vmm_k_offset = Vmm(1);
  */
    VReg vmm_c_tail_mask = VReg(2);

    XReg xmm_ker_area_h = XReg(2);
    XReg xmm_one = XReg(2);
    XReg xmm_tmp = XReg(3);

    VReg vmm_ker_area_h = VReg(2);
    VReg vmm_one = VReg(2);
    VReg vmm_tmp = VReg(3);
    ZReg ymm_tmp = ZReg(3);

    VReg vmm_k_offset = VReg(1);

    // Used only for avx512 when bf16 is present
    /*
    inline Vmm vmm_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? Vmm(4) : Vmm(1);
        } else
            return Vmm(4);
    }
  */
    /*
    inline VReg vmm_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? VReg(4) : VReg(1);
        } else
            return VReg(4);
    }
  */

    inline uint32_t reg_idx() {
        if (!jpp.is_backward) {
            return (jpp.is_training) ? 4 : 1;
        } else
            return 4;
    }

    /*
    Zmm bf16_emu_reserv_1 = Zmm(5);
    Zmm bf16_emu_reserv_2 = Zmm(6);
    Zmm bf16_emu_reserv_3 = Zmm(7);
    Reg64 bf16_emu_reserv_4 = r11;
    Zmm bf16_emu_reserv_5 = Zmm(8);
  */
    ZReg bf16_emu_reserv_1 = ZReg(5);
    ZReg bf16_emu_reserv_2 = ZReg(6);
    ZReg bf16_emu_reserv_3 = ZReg(7);
    XReg bf16_emu_reserv_4 = XReg(11);
    ZReg bf16_emu_reserv_5 = ZReg(8);

    const std::vector<uint32_t> tmp_vec_idx
            //  = {20, 21, 22, 23, 24, 25, 26, 27};
            = {4, 5, 6, 7};
    ZReg z_tmp0 = ZReg(4);
    ZReg z_tmp1 = ZReg(5);
    ZReg z_tmp2 = ZReg(6);
    ZReg z_tmp3 = ZReg(7);

    /*
    Opmask k_c_tail_mask = Opmask(4);
    Opmask k_mask_cvt = Opmask(5);
    Opmask k_store_mask = Opmask(6);
  */
    PReg k_c_tail_mask = PReg(4);
    PReg k_mask_cvt = PReg(5);
    PReg k_store_mask = PReg(6);

    /* Caution: Chose predicate registers not used by x64's implementation. */
    PReg p_256 = PReg(1);
    PReg p_512 = PReg(2);
    PReg p_tmp0 = PReg(3);
    PReg p_128 = PReg(7);
    PReg p_lsb = PReg(2);
    PReg p_tmp1 = PReg(11);
    PReg p_tmp2 = PReg(12);
    PReg P_MSB_256 = PReg(13);
    PReg P_MSB_384 = PReg(14);
    PReg P_ALL_ONE = PReg(15);

    // Here be some (tame) dragons. This kernel does not follow the regular
    // OS-agnostic ABI pattern because when isa is sse41 it uses maskmovdqu
    // instruction which has its destination hardcoded in rdi. Therefore:
    // - all registers are hardcoded
    // - on Windows rdi and rcx are swapped to mimic the Unix x86_64 ABI
    //
    // While this is only required by the backward pass, the quirk above
    // is applied to the forward pass as well to keep things simpler.
    /*
    using reg64_t = const Reg64;
    reg64_t reg_param = rdi; // Always mimic the Unix ABI
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t reg_index = r10;
    reg64_t reg_output = r12;
    reg64_t reg_kd_pad_shift = r13;
    reg64_t dst_ptr = rdi; // Must be rdi due to maskmovdqu

    reg64_t kj = r14;
    reg64_t oi_iter = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_k_shift = rbx;
    reg64_t tmp_gpr = rcx; // Must be rcx because rdi is used above
    reg64_t reg_ker_area_h = rdx;
    reg64_t reg_nbc = rsi;

    reg64_t reg_zero_ptr = r9;
    reg64_t reg_zero_id = r13;
    reg64_t reg_zero_ih = r14;
    reg64_t aux_reg_zero_ih = r15;
    reg64_t ki = r12;
    reg64_t aux_reg_input_d = r8;
  */
    using xreg_t = const XReg;
    xreg_t reg_param = XReg(7); // Always mimic the Unix ABI
    xreg_t reg_input = XReg(8);
    xreg_t aux_reg_input = XReg(9);
    xreg_t reg_index = XReg(10);
    xreg_t reg_output = XReg(12);
    xreg_t reg_kd_pad_shift = XReg(13);
    xreg_t dst_ptr = XReg(7); // Must be rdi due to maskmovdqu

    xreg_t kj = XReg(14);
    xreg_t oi_iter = XReg(15);
    xreg_t reg_kh = XReg(0);
    xreg_t reg_k_shift = XReg(3);
    xreg_t tmp_gpr = XReg(1); // Must be rcx because rdi is used above
    xreg_t reg_ker_area_h = XReg(2);
    xreg_t reg_nbc = XReg(6);

    xreg_t reg_zero_ptr = XReg(9);
    xreg_t reg_zero_id = XReg(13);
    xreg_t reg_zero_ih = XReg(14);
    xreg_t aux_reg_zero_ih = XReg(15);
    xreg_t ki = XReg(12);
    xreg_t aux_reg_input_d = XReg(8);

    using wreg_t = const WReg;
    wreg_t w_tmp_0 = WReg(23);
    wreg_t W_TMP_0 = WReg(23);

    xreg_t aux_xreg_input = XReg(9);
    xreg_t xreg_output = XReg(12);
    xreg_t xreg_index = XReg(10);
    xreg_t xreg_zero_ptr = XReg(9);
    xreg_t x_tmp_addr = XReg(28);
    xreg_t x_tmp_0 = XReg(23);
    xreg_t X_TMP_0 = XReg(23);
    xreg_t X_TRANSLATOR_STACK = XReg(22);

    //Reg32 reg_shuf_mask = esi;
    WReg reg_shuf_mask = WReg(7);

    bool sse_high_half = false;
    bool disable_postops_when_sse_high_half_processed_ = false;

    int prev_kw;

    void prepare_tail_mask();
    void put_one_in_vmm();
    void uni_broadcast_reg_val(const int reg_idx, const int vmm_idx);
    void push_vmm_val(const int idx);
    void pop_vmm_val(const int idx);
    void load(const int idx, const xreg_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);
    void store(const int idx, const xreg_t &reg_ptr, const int offset,
            const bool is_c_tail_proccessing);

    void maybe_recalculate_divisor(int jj, int ur_w, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void avg_step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_fwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);
    void max_step_bwd(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing);

    void zero_diff_src(int ur_bc, bool with_c_tail_proccessing);

    void step(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_proccessing) {
        if (jpp.alg == alg_kind::pooling_max) {
            if (jpp.is_backward)
                max_step_bwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
            else
                max_step_fwd(
                        ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
        } else
            avg_step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_proccessing);
    }
    /*
    void step_high_half(int ur_w, int ur_bc, int pad_l, int pad_r,
            bool with_c_tail_processing) {
      //add(reg_input, sizeof(float) * 4);
	add_imm(reg_input, reg_input, sizeof(float) * 4, x_tmp_0);	
        //add(reg_output, sizeof(float) * 4);
	add_imm(reg_output, reg_input, sizeof(float) * 4, x_tmp_0);		
        if (jpp.alg == alg_kind::pooling_max
                && (jpp.is_training || jpp.is_backward))
	  //add(reg_index, types::data_type_size(jpp.ind_dt) * 4);
	    add_imm(reg_index, reg_index, types::data_type_size(jpp.ind_dt) * 4, x_tmp_0);	

        step(ur_w, ur_bc, pad_l, pad_r, with_c_tail_processing);
    }
  */
    void generate() override;

    //void avx_vpadd1(const Ymm &y0, const Xmm &x1, const Xmm &xtmp) {
    void avx_vpadd1(const ZReg &y0, const XReg &x1, const XReg &xtmp) {
        /*
        assert(y0.getIdx() != x1.getIdx());
        mov(VReg(IDX(xtmp)).b16, VReg(IDX(y0)).b16);

        add(VReg(IDX(xtmp)).s4, VReg(IDX(xtmp)).s4,
                VReg(IDX(x1)).s4);
        mov(ZReg(IDX(xtmp)).s, P_MSB_384 / T_m, 0);

        ptrue(p_tmp0.d, VL2);
        sel(ZRegD(IDX(y0)), p_tmp0, ZRegD(IDX(xtmp)),
                ZRegD(IDX(y0)));
        mov(ZReg(IDX(y0)).s, P_MSB_256 / T_m, 0);

        mov(z_tmp0.d, ZRegD(IDX(y0)));
        ext(z_tmp0.b, ZRegB(IDX(y0)), 16);
        mov(VReg(IDX(xtmp)).b16, VReg(IDX(z_tmp0)).b16);

        add(VReg(IDX(xtmp)).s4, VReg(IDX(xtmp)).s4,
                VReg(IDX(x1)).s4);
        mov(ZReg(IDX(xtmp)).s, P_MSB_384 / T_m, 0);

        ptrue(p_tmp0.d, VL2);
        mov(z_tmp0.d, ZRegD(IDX(y0)));
        splice(z_tmp0.d, p_tmp0, ZRegD(IDX(xtmp)));
        mov(ZReg(IDX(y0)).d, z_tmp0.d);
        mov(ZReg(IDX(y0)).s, P_MSB_256 / T_m, 0);
      */
    }

    //void avx_vpadd1(const Xmm &x0, const Xmm &x1, const Xmm &) {
    void avx_vpadd1(const XReg &x0, const XReg &x1, const XReg &) {
        assert(false /*function should not be used*/);
        //paddd(x0, x1);
    }

    //void avx_pmovzxbd(const Ymm &y0, const Xmm &x1, const Xmm &xtmp) {
    void avx_pmovzxbd(const ZReg &y0, const XReg &x1, const XReg &xtmp) {
        //Xmm x0(y0.getIdx());
        /*
        XReg x0(y0.getIdx());
        pshufd(xmm_tmp, x1, 1);
        pmovzxbd(x0, x1);
        pmovzxbd(xmm_tmp, xmm_tmp);
        vinsertf128(y0, y0, xmm_tmp, 1);
      */
    }

    //void avx_pmovzxbd(const Xmm &x0, const Xmm &x1, const Xmm &) {
    void avx_pmovzxbd(const XReg &x0, const XReg &x1, const XReg &) {
        assert(false /*function should not be used*/);
        //pmovzxbd(x0, x1);
    }
    /*
    void avx_pcmpeqd(
            const Ymm &y0, const Ymm &y1, const Ymm &y2, const Xmm &xtmp) {
  */
    void avx_pcmpeqd(
            const ZReg &y0, const ZReg &y1, const ZReg &y2, const XReg &xtmp) {
        /*
        assert(y0.getIdx() != y1.getIdx());
        assert(y0.getIdx() != y2.getIdx());
        XReg x0(y0.getIdx());
        XReg x2(y2.getIdx());
        vextractf128(x0, y1, 1);
        vextractf128(xtmp, y2, 1);
        pcmpeqd(xtmp, x0);
        vextractf128(x0, y1, 0);
        pcmpeqd(x0, x2);
        vinsertf128(y0, y0, xtmp, 1);
      */
    }

    //void avx_pcmpeqd(const Xmm &x0, const Xmm &x1, const Xmm &, const Xmm &) {
    void avx_pcmpeqd(
            const XReg &x0, const XReg &x1, const XReg &, const XReg &) {
        assert(false /*function should not be used*/);
        //pcmpeqd(x0, x1);
    }

    void apply_postops(int ur_bc, int ur_w, int c_block,
            const std::function<bool(int)> &is_tail_predicate);

    static bool post_ops_ok(jit_pool_conf_t &jpp, const primitive_attr_t &attr,
            const memory_desc_wrapper &dst_d);

    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    /*
    std::unique_ptr<injector::jit_uni_postops_injector_t<isa>>
            postops_injector_;
  */
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
