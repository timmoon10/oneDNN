/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_SVE_512_CORE_BF16CVT_HPP
#define CPU_AARCH64_JIT_SVE_512_CORE_BF16CVT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "dnnl_debug.h"

#include "common/bfloat16.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace xa = Xbyak_aarch64;

namespace bf16_support {
struct jit_call_t {
    void *inp;
    void *out;
    void *add;
    size_t nelems;
    int mask;
};
} // namespace bf16_support

#define GET_OFF(field) offsetof(bf16_support::jit_call_t, field)

struct bf16_emulation_t {
    /*
  using opmask_t = const x64::Xbyak::Opmask;
  using Zmm_t = const x64::Xbyak::Zmm;
  using Ymm_t = const x64::Xbyak::Ymm;
  using Xmm_t = const x64::Xbyak::Xmm;
  using reg64_t = const x64::Xbyak::Reg64;
  */
    using opmask_t = const xa::PReg;
    using Zmm_t = const xa::ZReg;
    using Ymm_t = const xa::ZReg;
    using Xmm_t = const xa::VReg;
    using reg64_t = const xa::XReg;

    bf16_emulation_t(jit_generator *host, Zmm_t one, Zmm_t even, Zmm_t selector,
            reg64_t scratch, Zmm_t tr0, Zmm_t tr1)
        : host_(host)
        , one_(one)
        , even_(even)
        , selector_(selector)
        , scratch_(scratch)
        , tr0_(tr0)
        , tr1_(tr1) {}

    bf16_emulation_t(jit_generator *host, Zmm_t one, Zmm_t even, Zmm_t selector,
            reg64_t scratch, Zmm_t tr0)
        : bf16_emulation_t(host, one, even, selector, scratch, tr0, tr0) {}

    void vdpbf16ps(Zmm_t &acc, Zmm_t wei, Zmm_t inp) {
        /*
        host_->vpsrad(tr0_, wei, 16);
        host_->vpslld(tr0_, tr0_, 16);

        host_->vpsrad(tr1_, inp, 16);
        host_->vpslld(tr1_, tr1_, 16);

        host_->vfmadd231ps(acc, tr1_, tr0_);

        host_->vpslld(tr0_, wei, 16);
        host_->vpslld(tr1_, inp, 16);

        host_->vfmadd231ps(acc, tr1_, tr0_);
      */
    }

    void vcvtneps2bf16(Ymm_t &out, Zmm_t in) {
        //vcvtneps2bf16(out, in, tr0_, one_, even_, selector_);
        vcvtneps2bf16(out, xa::VReg(IDX(in)), xa::VReg(IDX(tr0_)), one_,
                xa::VReg(IDX(even_)), selector_);
    }

    void vcvtneps2bf16(Xmm_t &out, Ymm_t in) {
        /*
        const Ymm_t tr0_y {tr0_.getIdx()};
        const Ymm_t even_y {even_.getIdx()};
        const Ymm_t selector_y {selector_.getIdx()};
        const Ymm_t one_y {one_.getIdx()};

        vcvtneps2bf16(out, in, tr0_y, one_y, even_y, selector_y);
      */
    }

private:
    void vcvtneps2bf16(const Xmm_t &out, const Xmm_t &in, const Xmm_t &tr0,
            const Ymm_t &one, const Xmm_t &even, const Ymm_t &selector) {
        //host_->vpsrld(tr0, in, 16);
        host_->xa_->lsr(xa::ZRegS(IDX(tr0)), xa::ZRegS(IDX(in)), 16);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpandd(tr0, tr0, one);
        host_->xa_->and_(xa::VReg(IDX(tr0)).b16, xa::VReg(IDX(tr0)).b16,
                xa::VReg(IDX(one)).b16);

        //host_->vpaddd(tr0, even, tr0);
        host_->xa_->add(xa::VReg(IDX(tr0)).s4, xa::VReg(IDX(even)).s4,
                xa::VReg(IDX(tr0)).s4);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpaddd(tr0, in, tr0);
        host_->xa_->add(xa::VReg(IDX(tr0)).s4, xa::VReg(IDX(in)).s4,
                xa::VReg(IDX(tr0)).s4);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);
        //host_->vfixupimmps(tr0, in, selector, 0);
        xa::Label l_table, l_exec;
        host_->xa_->b(l_exec);
        // gen table
        host_->xa_->L(l_table);
        host_->xa_->dd(0xFFFFFFFF); // dummy for dest[31:0]
        host_->xa_->dd(0xFFFFFFFF); // dummy for tsrc[31:0]
        host_->xa_->dd(0x7FC00000); // QNAN(tsrc[31:0])
        host_->xa_->dd(0xFFC00000); // QNAN_Indefinite
        host_->xa_->dd(0xFF800000); // -INF
        host_->xa_->dd(0x7F800000); // +INF
        host_->xa_->dd(0x7F800000); // dummy for tsrc.sign? -INF:+INF
        host_->xa_->dd(0x80000000); // -0;
        host_->xa_->dd(0x00000000); // +0;
        host_->xa_->dd(0xBF800000); // -1;
        host_->xa_->dd(0x3F800000); // +1;
        host_->xa_->dd(0x3F000000); // 1/2;
        host_->xa_->dd(0x42B40000); // 90.0
        host_->xa_->dd(0x3FC90FDB); // PI/2
        host_->xa_->dd(0x7F7FFFFF); // MAX_FLOAT
        host_->xa_->dd(0xFF7FFFFF); // -MIN_FLOAT
        host_->xa_->L(l_exec);
        host_->xa_->mov(z_tmp.d, xa::ZRegD(IDX(in)));
        host_->xa_->mov(z_tmp3.d, 0);
        host_->xa_->mov(z_tmp2.s, uint64_t(1) << 22);
        host_->xa_->ptrue(p_tmp.b);
        host_->xa_->adr(x_tmp_addr, l_table);
        host_->xa_->fcmuo(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp.s);
        host_->xa_->and_(z_tmp2.d, z_tmp.d, z_tmp2.d);
        host_->xa_->cmpeq(P_TMP_1.s, p_tmp / xa::T_z, z_tmp2.s, 0);
        host_->xa_->and_(P_TMP_1.b, p_tmp / xa::T_z, P_TMP_0.b, P_TMP_1.b);
        host_->xa_->mov(z_tmp3.s, P_TMP_1 / xa::T_m, 1);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fdup(z_tmp2.s, float(1.0));
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, float(0.0));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 2);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp2.s);
        host_->xa_->dupm(z_tmp2.s, uint64_t(0xff800000));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 3);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp2.s);
        host_->xa_->dupm(z_tmp2.s, uint64_t(0x7f800000));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 4);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp2.s);
        host_->xa_->mov(z_tmp2.d, xa::ZRegD(IDX(selector)));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 5);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->cmplt(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, 0);
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 6);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->mov(z_tmp3.s, p_tmp / xa::T_m, 7);
        host_->xa_->ptrue(p_tmp.b);
        host_->xa_->lsl(z_tmp3.s, z_tmp3.s, 2);
        host_->xa_->lsr(z_tmp2.s, p_tmp, z_tmp3.s);
        host_->xa_->and_(z_tmp2.s, uint64_t(0xf));
        host_->xa_->cmpne(P_TMP_0.s, p_tmp / xa::T_z, z_tmp2.s, 0);
        host_->xa_->ld1w(z_tmp3.s, p_tmp / xa::T_z,
                xa::ptr(x_tmp_addr, z_tmp2.s, xa::UXTW, 2));
        host_->xa_->mov(xa::ZRegS(IDX(tr0)), P_TMP_0 / xa::T_m, z_tmp3.s);
        host_->xa_->dupm(z_tmp3.s, uint64_t(0x807fffff));
        host_->xa_->and_(z_tmp3.d, z_tmp3.d, z_tmp.d);
        host_->xa_->cmpeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp2.s, 1);
        host_->xa_->mov(xa::ZRegS(IDX(tr0)), P_TMP_0 / xa::T_m, z_tmp.s);
        host_->xa_->and_(z_tmp.s, uint64_t(0x80000000));
        host_->xa_->cmpeq(p_tmp2.s, p_tmp / xa::T_z, z_tmp2.s, 2);
        host_->xa_->orr(xa::ZRegS(IDX(tr0)), p_tmp2 / xa::T_m, z_tmp3.s);
        host_->xa_->cmpeq(p_tmp2.s, p_tmp / xa::T_z, z_tmp2.s, 6);
        host_->xa_->orr(xa::ZRegS(IDX(tr0)), p_tmp2 / xa::T_m, z_tmp.s);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpsrad(tr0, tr0, 16);
        host_->xa_->asr(xa::ZRegS(IDX(tr0)), xa::ZRegS(IDX(tr0)), 16);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);
        //host_->vpmovdw(out, tr0);
        host_->xa_->mov(z_tmp.d, xa::ZRegD(IDX(tr0)));
        host_->xa_->mov(z_tmp.b, P_MSB_384 / xa::T_m, 0);
        host_->xa_->dup(xa::ZRegS(IDX(out)), 0);
        host_->xa_->uzp1(xa::ZRegH(IDX(out)), z_tmp.h, xa::ZRegH(IDX(out)));
    }

    void vcvtneps2bf16(const Ymm_t &out, const Xmm_t &in, const Xmm_t &tr0,
            const Zmm_t &one, const Xmm_t &even, const Zmm_t &selector) {
        //host_->vpsrld(tr0, in, 16);
        host_->xa_->lsr(xa::ZRegS(IDX(tr0)), xa::ZRegS(IDX(in)), 16);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpandd(tr0, tr0, one);
        host_->xa_->mov(z_tmp.d, one.d);
        host_->xa_->and_(xa::ZReg(IDX(tr0)).d, xa::ZReg(IDX(tr0)).d, z_tmp.d);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpaddd(tr0, even, tr0);
        host_->xa_->add(xa::VReg(IDX(tr0)).s4, xa::VReg(IDX(even)).s4,
                xa::VReg(IDX(tr0)).s4);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpaddd(tr0, in, tr0);
        host_->xa_->add(xa::VReg(IDX(tr0)).s4, xa::VReg(IDX(in)).s4,
                xa::VReg(IDX(tr0)).s4);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);
        //host_->vfixupimmps(tr0, in, selector, 0);
        xa::Label l_exec, l_table;
        host_->xa_->b(l_exec);
        // gen table
        //L_aarch64(l_table);
        host_->xa_->L(l_table);
        host_->xa_->dd(0xFFFFFFFF); // dummy for dest[31:0]
        host_->xa_->dd(0xFFFFFFFF); // dummy for tsrc[31:0]
        host_->xa_->dd(0x7FC00000); // QNAN(tsrc[31:0])
        host_->xa_->dd(0xFFC00000); // QNAN_Indefinite
        host_->xa_->dd(0xFF800000); // -INF
        host_->xa_->dd(0x7F800000); // +INF
        host_->xa_->dd(0x7F800000); // dummy for tsrc.sign? -INF:+INF
        host_->xa_->dd(0x80000000); // -0;
        host_->xa_->dd(0x00000000); // +0;
        host_->xa_->dd(0xBF800000); // -1;
        host_->xa_->dd(0x3F800000); // +1;
        host_->xa_->dd(0x3F000000); // 1/2;
        host_->xa_->dd(0x42B40000); // 90.0
        host_->xa_->dd(0x3FC90FDB); // PI/2
        host_->xa_->dd(0x7F7FFFFF); // MAX_FLOAT
        host_->xa_->dd(0xFF7FFFFF); // -MIN_FLOAT
        //L_aarch64(l_exec);
        host_->xa_->L(l_exec);
        host_->xa_->mov(z_tmp.d, xa::ZRegD(IDX(in)));
        host_->xa_->mov(z_tmp3.d, 0);
        host_->xa_->mov(z_tmp2.s, uint64_t(1) << 22);
        host_->xa_->ptrue(p_tmp.b);
        host_->xa_->adr(x_tmp_addr, l_table);
        host_->xa_->fcmuo(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp.s);
        host_->xa_->and_(z_tmp2.d, z_tmp.d, z_tmp2.d);
        host_->xa_->cmpeq(P_TMP_1.s, p_tmp / xa::T_z, z_tmp2.s, 0);
        host_->xa_->and_(P_TMP_1.b, p_tmp / xa::T_z, P_TMP_0.b, P_TMP_1.b);
        host_->xa_->mov(z_tmp3.s, P_TMP_1 / xa::T_m, 1);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fdup(z_tmp2.s, float(1.0));
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, float(0.0));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 2);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp2.s);
        host_->xa_->dupm(z_tmp2.s, uint64_t(0xff800000));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 3);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp2.s);
        host_->xa_->dupm(z_tmp2.s, uint64_t(0x7f800000));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 4);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->fcmeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, z_tmp2.s);
        host_->xa_->mov(z_tmp2.d, xa::ZRegD(IDX(selector)));
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 5);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->cmplt(P_TMP_0.s, p_tmp / xa::T_z, z_tmp.s, 0);
        host_->xa_->mov(z_tmp3.s, P_TMP_0 / xa::T_m, 6);
        host_->xa_->bic(p_tmp.b, p_tmp / xa::T_z, p_tmp.b, P_TMP_0.b);
        host_->xa_->mov(z_tmp3.s, p_tmp / xa::T_m, 7);
        host_->xa_->ptrue(p_tmp.b);
        host_->xa_->lsl(z_tmp3.s, z_tmp3.s, 2);
        host_->xa_->lsr(z_tmp2.s, p_tmp, z_tmp3.s);
        host_->xa_->and_(z_tmp2.s, uint64_t(0xf));
        host_->xa_->cmpne(P_TMP_0.s, p_tmp / xa::T_z, z_tmp2.s, 0);
        host_->xa_->ld1w(z_tmp3.s, p_tmp / xa::T_z,
                xa::ptr(x_tmp_addr, z_tmp2.s, xa::UXTW, 2));
        host_->xa_->mov(xa::ZRegS(IDX(tr0)), P_TMP_0 / xa::T_m, z_tmp3.s);
        host_->xa_->dupm(z_tmp3.s, uint64_t(0x807fffff));
        host_->xa_->and_(z_tmp3.d, z_tmp3.d, z_tmp.d);
        host_->xa_->cmpeq(P_TMP_0.s, p_tmp / xa::T_z, z_tmp2.s, 1);
        host_->xa_->mov(xa::ZRegS(IDX(tr0)), P_TMP_0 / xa::T_m, z_tmp.s);
        host_->xa_->and_(z_tmp.s, uint64_t(0x80000000));
        host_->xa_->cmpeq(p_tmp2.s, p_tmp / xa::T_z, z_tmp2.s, 2);
        host_->xa_->orr(xa::ZRegS(IDX(tr0)), p_tmp2 / xa::T_m, z_tmp3.s);
        host_->xa_->cmpeq(p_tmp2.s, p_tmp / xa::T_z, z_tmp2.s, 6);
        host_->xa_->orr(xa::ZRegS(IDX(tr0)), p_tmp2 / xa::T_m, z_tmp.s);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);

        //host_->vpsrad(tr0, tr0, 16);
        host_->xa_->asr(xa::ZRegS(IDX(tr0)), xa::ZRegS(IDX(tr0)), 16);
        host_->xa_->mov(xa::ZReg(IDX(tr0)).s, P_MSB_384 / xa::T_m, 0);
        //host_->vpmovdw(out, tr0);
        host_->xa_->mov(z_tmp.d, xa::ZRegD(IDX(tr0)));
        host_->xa_->dup(xa::ZRegS(IDX(out)), 0);
        host_->xa_->uzp1(xa::ZRegH(IDX(out)), z_tmp.h, xa::ZRegH(IDX(out)));
    }

public:
    void init_vcvtneps2bf16() {
        const int selector_int32 =
                /* qnan input to qnan output (presenrving input bits 0..21) */
                encode_fixup_selector(
                        fixup_input_code_snan, fixup_output_code_qnan_input)
                |
                /* snan input to qnan output (presenrving input bits 0..21) */
                encode_fixup_selector(
                        fixup_input_code_qnan, fixup_output_code_qnan_input)
                |
                /* neg inf input copied to output */
                encode_fixup_selector(
                        fixup_input_code_ninf, fixup_output_code_copy_input)
                |
                /* pos inf input copied to output */
                encode_fixup_selector(
                        fixup_input_code_pinf, fixup_output_code_copy_input);

        //host_->xor_(scratch_, scratch_);
        host_->xa_->eor(xa::XReg(IDX(scratch_)), xa::XReg(IDX(scratch_)),
                xa::XReg(IDX(scratch_)));
        //host_->mov(scratch_.cvt32(), 0x1);
        host_->xa_->mov_imm(xa::WReg(IDX(scratch_)), 0x1);
        //host_->vpbroadcastd(one_, scratch_.cvt32());
        host_->xa_->dup(xa::ZRegS(IDX(one_)), xa::WReg(IDX(scratch_)));
        /*
	int vlen = cpu_isa_traits<isa>::vlen;
	if (vlen == 64) {
	  host_->xa_->dup(xa::ZRegS(IDX(one_)), xa::WReg(IDX(scratch_)));
	} else if (vlen == 32) {
	  host_->xa_->dup(xa::ZRegS(IDX(one_)), xa::WReg(IDX(scratch_)));
	  host_->xa_->mov(xa::ZRegS(IDX(one_)), P_MSB_256/xa::T_m, 0);
	} else if (vlen == 16) {
	  host_->xa_->dup(xa::ZRegS(IDX(one_)), xa::WReg(IDX(scratch_)));
	  host_->xa_->mov(xa::ZRegS(IDX(one_)), P_MSB_384/xa::T_m, 0);
	} else {
	  assert(!"unreachable");
	}
	*/
        //host_->xor_(scratch_, scratch_);
        host_->xa_->eor(xa::XReg(IDX(scratch_)), xa::XReg(IDX(scratch_)),
                xa::XReg(IDX(scratch_)));
        //host_->mov(scratch_.cvt32(), 0x7fff);
        host_->xa_->mov_imm(xa::WReg(IDX(scratch_)), 0x7fff);
        //host_->vpbroadcastd(even_, scratch_.cvt32());
        host_->xa_->dup(xa::ZRegS(IDX(even_)), xa::WReg(IDX(scratch_)));
        /*
	if (vlen == 64) {
	  host_->xa_->dup(xa::ZRegS(IDX(even_)), xa::WReg(IDX(scratch_)));
	} else if (vlen == 32) {
	  host_->xa_->dup(xa::ZRegS(IDX(even_)), xa::WReg(IDX(scratch_)));
	  host_->xa_->mov(xa::ZRegS(IDX(even_)), P_MSB_256/xa::T_m, 0);
	} else if (vlen == 16) {
	  host_->xa_->dup(xa::ZRegS(IDX(even_)), xa::WReg(IDX(scratch_)));
	  host_->xa_->mov(xa::ZRegS(IDX(even_)), P_MSB_384/xa::T_m, 0);
	} else {
	  assert(!"unreachable");
	}
	*/
        //host_->xor_(scratch_, scratch_);
        host_->xa_->eor(xa::XReg(IDX(scratch_)), xa::XReg(IDX(scratch_)),
                xa::XReg(IDX(scratch_)));
        //host_->mov(scratch_.cvt32(), selector_int32);
        host_->xa_->mov_imm(xa::WReg(IDX(scratch_)), selector_int32);
        //host_->vpbroadcastd(selector_, scratch_.cvt32());
        host_->xa_->dup(xa::ZRegS(IDX(selector_)), xa::WReg(IDX(scratch_)));
        /*
	if (vlen == 64) {
	  host_->xa_->dup(xa::ZRegS(IDX(selector_)), xa::WReg(IDX(scratch_)));
	} else if (vlen == 32) {
	  host_->xa_->dup(xa::ZRegS(IDX(selector_)), xa::WReg(IDX(scratch_)));
	  host_->xa_->mov(xa::ZRegS(IDX(selector_)), P_MSB_256/xa::T_m, 0);
	} else if (vlen == 16) {
	  host_->xa_->dup(xa::ZRegS(IDX(selector_)), xa::WReg(IDX(scratch_)));
	  host_->xa_->mov(xa::ZRegS(IDX(selector_)), P_MSB_384/xa::T_m, 0);
	} else {
	  assert(!"unreachable");
	}
	*/
    }

    //static cpu_isa_t get_isa() { return avx512_core; }
    static cpu_isa_t get_isa() { return sve_512; }

private:
    jit_generator *const host_;
    Zmm_t one_;
    Zmm_t even_;
    Zmm_t selector_;
    reg64_t scratch_;
    Zmm_t tr0_;
    Zmm_t tr1_;

    xa::PReg p_tmp = xa::PReg(0);
    xa::PReg p_tmp2 = xa::PReg(1);
    xa::PReg P_TMP_0 = xa::PReg(11);
    xa::PReg P_TMP_1 = xa::PReg(12);
    xa::PReg P_MSB_256 = xa::PReg(13);
    xa::PReg P_MSB_384 = xa::PReg(14);
    xa::PReg P_ALL_ONE = xa::PReg(15);

    Zmm_t z_tmp = xa::ZReg(24);
    Zmm_t z_tmp2 = xa::ZReg(25);
    Zmm_t z_tmp3 = xa::ZReg(26);

    reg64_t x_tmp_addr = xa::XReg(28);
    reg64_t x_tmp_0 = xa::XReg(23);

    int encode_fixup_selector(int input, int output) {
        return ((output) << (4 * (input)));
    }

    enum {
        fixup_input_code_qnan = 0,
        fixup_input_code_snan = 1,
        fixup_input_code_ninf = 4,
        fixup_input_code_pinf = 5,
        fixup_output_code_copy_input = 1,
        fixup_output_code_qnan_input = 2,
    };
};

struct jit_avx512_core_cvt_ps_to_bf16_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_cvt_ps_to_bf16)

    jit_avx512_core_cvt_ps_to_bf16_t(size_t nelems = 0)
        : nelems_(nelems)
        , simd_w_(16)
        , tail_mask_((1 << (nelems % simd_w_)) - 1)
        , is_dynamic_size_(nelems_ == 0) {

        bf16_emu_ = new bf16_emulation_t(
                this, one, even, selector, scratch, fp32_tmp);
        /*
        generate();
        jit_ker_ = (void (*)(bf16_support::jit_call_t *))getCode();
	*/
        create_kernel();
    }

    ~jit_avx512_core_cvt_ps_to_bf16_t() { delete bf16_emu_; }

    void generate() {
        preamble();

        //bool use_bf16_emu = !mayiuse(avx512_core_bf16);
        bool use_bf16_emu = false;
        //auto cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
        auto cvt = [&](size_t idx, xa::PReg ktail_mask) {
            /*
            vmovups(fp32_inp | ktail_mask | T_z,
                    ptr[reg_inp + sizeof(float) * (idx)]);
		     */
            add_imm(x_tmp_addr, xa::XReg(IDX(reg_inp)), sizeof(float) * (idx),
                    x_tmp_0);
            ld1w(xa::ZRegS(IDX(fp32_inp)), xa::PReg(IDX(ktail_mask)) / xa::T_z,
                    xa::ptr(x_tmp_addr));
            /*
	    int vlen = cpu_isa_traits<isa>::vlen;
	    if (vlen == 64) {
	      add_imm(x_tmp_addr, xa::XReg(IDX(reg_inp)), sizeof(float) * (idx), x_tmp_0);
	      ld1w(xa::ZRegS(IDX(fp32_inp)), xa::PReg(IDX(ktail_mask))/xa::T_z, xa::ptr(x_tmp_addr));
	    } else if (vlen == 32) {
	      bic(p_tmp.b, P_ALL_ONE/xa::T_z, xa::PRegB(IDX(ktail_mask)), P_MSB_256.b);
	      ld1w(xa::ZRegS(IDX(fp32_inp)), p_tmp/xa::T_z, xa::ptr(x_tmp_addr));
	    } else if (vlen == 16) {
	      bic(p_tmp.b, P_ALL_ONE/xa::T_z, xa::PRegB(IDX(ktail_mask)), P_MSB_384.b);
	      ld1w(xa::ZRegS(IDX(fp32_inp)), p_tmp/xa::T_z, xa::ptr(x_tmp_addr));
	    } else {
	      assert(!"unreachable");
	    }
		     */
            if (use_bf16_emu) {
                bf16_emu_->vcvtneps2bf16(bf16_out, fp32_inp);
            } else {
                /*
	      vcvtneps2bf16(bf16_out, fp32_inp);
	      */
            }
            //vmovdqu16(yword[reg_out + sizeof(bfloat16_t) * (idx)] | ktail_mask, bf16_out);
            bic(p_tmp.b, P_ALL_ONE / xa::T_z, xa::PRegB(IDX(ktail_mask)),
                    P_MSB_256.b);
            add_imm(x_tmp_addr, xa::XReg(IDX(reg_out)),
                    sizeof(bfloat16_t) * (idx), x_tmp_0);
            st1h(xa::ZRegH(IDX(bf16_out)), p_tmp, xa::ptr(x_tmp_addr));
        };
        //mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        add_imm(x_tmp_addr, xa::XReg(IDX(abi_param1)), GET_OFF(inp), x_tmp_0);
        ldr(x_tmp_0, xa::ptr(x_tmp_addr));
        xa_->mov(xa::XReg(IDX(reg_inp)), x_tmp_0);
        //mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        add_imm(x_tmp_addr, xa::XReg(IDX(abi_param1)), GET_OFF(out), x_tmp_0);
        ldr(x_tmp_0, xa::ptr(x_tmp_addr));
        xa_->mov(xa::XReg(IDX(reg_out)), x_tmp_0);
        if (is_dynamic_size_) {
            //mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);
            add_imm(x_tmp_addr, xa::XReg(IDX(abi_param1)), GET_OFF(nelems),
                    x_tmp_0);
            ldr(x_tmp_0, xa::ptr(x_tmp_addr));
            xa_->mov(xa::XReg(IDX(reg_nelems)), x_tmp_0);
        }

        if (use_bf16_emu) { bf16_emu_->init_vcvtneps2bf16(); }
        /*
        xa_->mov(reg32_tail, 0xffff);
        kmovw(ktail_mask, reg32_tail);
	*/

        if (is_dynamic_size_) { // determine nelems after JIT is called
            constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
            //Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
            xa::Label l_simd_loop[n_unroll + 2], l_simd_notail;
            for (int i = n_unroll; i >= 0; i--) {
                const int unroll = 1 << i; // 4, 2, 1
                L(l_simd_loop[i + 1]);
                {
                    //cmp(reg_nelems, simd_w_ * unroll);
                    cmp_imm(xa::XReg(IDX(reg_nelems)), simd_w_ * unroll,
                            x_tmp_0);
                    //jl(l_simd_loop[i], T_NEAR);
                    b(xa::LT, l_simd_loop[i]);
                    for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                        cvt(j, ktail_mask);
                    }
                    //add(reg_inp, simd_w_ * unroll * sizeof(float));
                    add_imm(xa::XReg(IDX(reg_inp)), xa::XReg(IDX(reg_inp)),
                            (simd_w_ * unroll * sizeof(float)), x_tmp_0);
                    //add(reg_out, simd_w_ * unroll * sizeof(bfloat16_t));
                    add_imm(xa::XReg(IDX(reg_out)), xa::XReg(IDX(reg_out)),
                            (simd_w_ * unroll * sizeof(bfloat16_t)), x_tmp_0);
                    //sub(reg_nelems, simd_w_ * unroll);
                    sub_imm(xa::XReg(IDX(reg_nelems)),
                            xa::XReg(IDX(reg_nelems)), (simd_w_ * unroll),
                            x_tmp_0);
                    //jmp(l_simd_loop[i + 1], T_NEAR);
                    b(l_simd_loop[i + 1]);
                }
            }
            L(l_simd_loop[0]);
            //test(reg_nelems, reg_nelems);
            xa_->and_(X_TMP_2, xa::XReg(IDX(reg_nelems)),
                    xa::XReg(IDX(reg_nelems)));
            ands(X_TMP_2, xa::XReg(IDX(reg_nelems)), xa::XReg(IDX(reg_nelems)));
            lsr(X_TMP_2, X_TMP_2, 63);
            lsl(X_TMP_2, X_TMP_2, 31);
            mrs(X_TMP_0, 0x3, 0x3, 0x4, 0x2, 0x0);
            orr(X_TMP_0, X_TMP_0, X_TMP_2);
            msr(0x3, 0x3, 0x4, 0x2, 0x0, X_TMP_0);
            //jz(l_simd_notail);
            b(xa::EQ, l_simd_notail);
            // JIT of `tail_mask_ = (1 << (nelems_ % simd_w_)) - 1;`
            //mov(reg32_mask, 1);
            xa_->mov_imm(xa::WReg(IDX(reg32_mask)), 1);
            //mov(reg64_tail, reg_nelems);
            xa_->mov(xa::XReg(IDX(reg64_tail)), xa::XReg(IDX(reg_nelems)));
            //shl(reg32_mask, reg8_mask_shift);
            xa_->and_(W_TMP_0, xa::WReg(IDX(reg8_mask_shift)), 0x1f);
            lsl(xa::WReg(IDX(reg32_mask)), xa::WReg(IDX(reg32_mask)), W_TMP_0);
            //sub(reg32_mask, 1);
            sub_imm(xa::WReg(IDX(reg32_mask)), xa::WReg(IDX(reg32_mask)), 1,
                    w_tmp_0);
            //kmovd(ktail_mask, reg32_mask);
            cvt(0, ktail_mask);
            L(l_simd_notail);

        } else {

            size_t blocked_size = (nelems_ / simd_w_) * simd_w_;
            const size_t loop_length = 1024;
            const size_t number_of_loops = blocked_size / loop_length;
            const size_t tail_of_loops = blocked_size % loop_length;

            if (number_of_loops > 0) {
                //Xbyak::Label l_number_of_loops;
                Xbyak_aarch64::Label l_number_of_loops;
                //mov(reg_nelems, number_of_loops);
                L(l_number_of_loops);
                /*
                for (size_t i = 0; i < loop_length; i += simd_w_)
                    cvt(i, ktail_mask);
                add(reg_inp, sizeof(float) * loop_length);
                add(reg_out, sizeof(bfloat16_t) * loop_length);

                dec(reg_nelems);
                cmp(reg_nelems, 0);
                jg(l_number_of_loops, T_NEAR);
		*/
            }
            if (tail_of_loops > 0) {
                /*
                for (size_t i = 0; i < tail_of_loops; i += simd_w_)
                    cvt(i, ktail_mask);
                add(reg_inp, sizeof(float) * tail_of_loops);
                add(reg_out, sizeof(bfloat16_t) * tail_of_loops);
	      */
            }
            if (tail_mask_ != 0) {
                /* kurihara implement later
                xa_->mov(reg32_tail, tail_mask_);
                kmovw(ktail_mask, reg32_tail);
                cvt(0, ktail_mask);
	      */
            }
        }
        postamble();
    }

    void jit_ker(bf16_support::jit_call_t *params) const {
        jit_ker_(params);
        msan_unpoison(params->out,
                (nelems_ ? nelems_ : params->nelems) * sizeof(bfloat16_t));
    }

private:
    size_t nelems_;
    int simd_w_;
    int tail_mask_;

    void (*jit_ker_)(bf16_support::jit_call_t *);

    bf16_emulation_t *bf16_emu_;
    bool is_dynamic_size_;
    /*
    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm fp32_inp = Xbyak::Zmm(0);
    Xbyak::Zmm fp32_tmp = Xbyak::Zmm(1);

    Xbyak::Zmm one = Xbyak::Zmm(2);
    Xbyak::Zmm even = Xbyak::Zmm(3);
    Xbyak::Zmm selector = Xbyak::Zmm(4);

    Xbyak::Ymm bf16_out = Xbyak::Ymm(5);

    Xbyak::Reg64 scratch = r15;
    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_nelems = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;
  */
    Xbyak_aarch64::PReg ktail_mask = Xbyak_aarch64::PReg(2);
    Xbyak_aarch64::ZReg fp32_inp = Xbyak_aarch64::ZReg(0);
    Xbyak_aarch64::ZReg fp32_tmp = Xbyak_aarch64::ZReg(1);

    Xbyak_aarch64::ZReg one = Xbyak_aarch64::ZReg(2);
    Xbyak_aarch64::ZReg even = Xbyak_aarch64::ZReg(3);
    Xbyak_aarch64::ZReg selector = Xbyak_aarch64::ZReg(4);

    Xbyak_aarch64::ZReg bf16_out = Xbyak_aarch64::ZReg(5);

    Xbyak_aarch64::XReg scratch = Xbyak_aarch64::XReg(15);
    Xbyak_aarch64::XReg reg_inp = Xbyak_aarch64::XReg(0);
    Xbyak_aarch64::XReg reg_out = Xbyak_aarch64::XReg(1);
    Xbyak_aarch64::XReg reg_nelems = Xbyak_aarch64::XReg(3);

    Xbyak_aarch64::XReg reg64_tail = Xbyak_aarch64::XReg(2);
    Xbyak_aarch64::WReg reg32_tail = Xbyak_aarch64::WReg(2);
    ;
    Xbyak_aarch64::WReg reg8_mask_shift = Xbyak_aarch64::WReg(2);
    Xbyak_aarch64::WReg reg32_mask = Xbyak_aarch64::WReg(8);

    xa::PReg p_tmp = xa::PReg(0);
    xa::PReg p_tmp2 = xa::PReg(1);
    xa::PReg P_TMP_0 = xa::PReg(11);
    xa::PReg P_TMP_1 = xa::PReg(12);
    xa::PReg P_MSB_256 = xa::PReg(13);
    xa::PReg P_MSB_384 = xa::PReg(14);
    xa::PReg P_ALL_ONE = xa::PReg(15);

    xa::ZReg z_tmp = xa::ZReg(24);
    xa::ZReg z_tmp2 = xa::ZReg(25);
    xa::ZReg z_tmp3 = xa::ZReg(26);

    xa::XReg x_tmp_addr = xa::XReg(28);
    xa::XReg x_tmp_0 = xa::XReg(23);

    xa::WReg w_tmp_0 = xa::WReg(23);
};

struct jit_avx512_core_cvt_bf16_to_ps_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_cvt_bf16_to_ps_t)

    jit_avx512_core_cvt_bf16_to_ps_t(
            bool with_add = false, size_t row_stride = 0)
        : with_add_(with_add), row_stride_(row_stride) {
        /*
        generate();
        jit_ker_ = (decltype(jit_ker_))getCode();
      */
        create_kernel();
    }

    void generate();

    void jit_ker(float *out, const bfloat16_t *inp, size_t nelems,
            size_t rows = 1) const {
        jit_ker_(out, inp, nelems, rows);
        msan_unpoison(out, nelems * sizeof(float));
    }

private:
    bool with_add_;
    size_t row_stride_;

    void (*jit_ker_)(
            float *out, const bfloat16_t *inp, size_t nelems, size_t nrows);
};

// performs element-by-element sum of inp and add float arrays and stores
// result to bfloat16 out array with downconversion
struct jit_avx512_core_add_cvt_ps_to_bf16_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_add_cvt_ps_to_bf16)

    jit_avx512_core_add_cvt_ps_to_bf16_t() : simd_w_(16) {
        bf16_emu_ = new bf16_emulation_t(
                this, one, even, selector, scratch, fp32_tmp, fp32_tmp);

        /*
        generate();
        jit_ker_ = (void (*)(bf16_support::jit_call_t *))getCode();
	*/
    }

    ~jit_avx512_core_add_cvt_ps_to_bf16_t() { delete bf16_emu_; }
    /*
    void generate() {
        preamble();

        bool use_bf16_emu = !mayiuse(avx512_core_bf16);
	bool use_bf16_emu = false;
        auto add_cvt = [&](size_t idx, Xbyak::Opmask ktail_mask) {
            vmovups(fp32_inp | ktail_mask | T_z,
                    ptr[reg_inp + sizeof(float) * (idx)]);
            vaddps(fp32_inp | ktail_mask | T_z, fp32_inp,
                    ptr[reg_add + sizeof(float) * (idx)]);
            if (use_bf16_emu)
                bf16_emu_->vcvtneps2bf16(bf16_out, fp32_inp);
            else
                vcvtneps2bf16(bf16_out, fp32_inp);

            vmovdqu16(yword[reg_out + sizeof(bfloat16_t) * (idx)] | ktail_mask,
                    bf16_out);
        };
        xa_->mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        xa_->mov(reg_add, ptr[abi_param1 + GET_OFF(add)]);
        xa_->mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        xa_->mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

        if (use_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

        xa_->mov(reg32_tail, 0xffff);
        kmovw(ktail_mask, reg32_tail);

        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]);
            {
                cmp(reg_nelems, simd_w_ * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < simd_w_ * unroll; j += simd_w_) {
                    add_cvt(j, ktail_mask);
                }
                add(reg_inp, simd_w_ * unroll * sizeof(float));
                add(reg_add, simd_w_ * unroll * sizeof(float));
                add(reg_out, simd_w_ * unroll * sizeof(bfloat16_t));

                sub(reg_nelems, simd_w_ * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);
        test(reg_nelems, reg_nelems);
        jz(l_simd_notail);
        // JIT of `tail_mask_ = (1 << (nelems_ % simd_w_)) - 1;`
        xa_->mov(reg32_mask, 1);
        xa_->mov(reg64_tail, reg_nelems);
        shl(reg32_mask, reg8_mask_shift);
        sub(reg32_mask, 1);
        kmovd(ktail_mask, reg32_mask);
        add_cvt(0, ktail_mask);
        L(l_simd_notail);
        postamble();
    }
  */
    void jit_ker(bf16_support::jit_call_t *params) const {
        /*
        jit_ker_(params);
        msan_unpoison(params->out, params->nelems * sizeof(bfloat16_t));
      */
    }

private:
    int simd_w_;
    void (*jit_ker_)(bf16_support::jit_call_t *);

    bf16_emulation_t *bf16_emu_;
    /*
    Xbyak::Opmask ktail_mask = k2;
    Xbyak::Zmm fp32_inp = Xbyak::Zmm(0);
    Xbyak::Zmm fp32_tmp = Xbyak::Zmm(1);

    Xbyak::Zmm one = Xbyak::Zmm(2);
    Xbyak::Zmm even = Xbyak::Zmm(3);
    Xbyak::Zmm selector = Xbyak::Zmm(4);
    Xbyak::Reg64 scratch = r15;

    Xbyak::Ymm bf16_out = Xbyak::Ymm(5);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_add = r11;
    Xbyak::Reg64 reg_nelems = rdx;

    Xbyak::Reg64 reg64_tail = rcx;
    Xbyak::Reg32 reg32_tail = ecx;
    Xbyak::Reg8 reg8_mask_shift = cl;
    Xbyak::Reg32 reg32_mask = r8d;
  */
    Xbyak_aarch64::PReg ktail_mask = Xbyak_aarch64::PReg(2);
    Xbyak_aarch64::ZReg fp32_inp = Xbyak_aarch64::ZReg(0);
    Xbyak_aarch64::ZReg fp32_tmp = Xbyak_aarch64::ZReg(1);

    Xbyak_aarch64::ZReg one = Xbyak_aarch64::ZReg(2);
    Xbyak_aarch64::ZReg even = Xbyak_aarch64::ZReg(3);
    Xbyak_aarch64::ZReg selector = Xbyak_aarch64::ZReg(4);
    Xbyak_aarch64::XReg scratch = Xbyak_aarch64::XReg(15);

    Xbyak_aarch64::ZReg bf16_out = Xbyak_aarch64::ZReg(5);

    Xbyak_aarch64::XReg reg_inp = Xbyak_aarch64::XReg(0);
    Xbyak_aarch64::XReg reg_out = Xbyak_aarch64::XReg(1);
    Xbyak_aarch64::XReg reg_add = Xbyak_aarch64::XReg(11);
    Xbyak_aarch64::XReg reg_nelems = Xbyak_aarch64::XReg(3);

    Xbyak_aarch64::XReg reg64_tail = Xbyak_aarch64::XReg(2);
    Xbyak_aarch64::WReg reg32_tail = Xbyak_aarch64::WReg(2);
    Xbyak_aarch64::WReg reg8_mask_shift = Xbyak_aarch64::WReg(2);
    ;
    Xbyak_aarch64::WReg reg32_mask = Xbyak_aarch64::WReg(8);
};

// implementation of reorder of part of tensor [s][16c] -> [S][16c][2s]
// it is required for quick implementation of 1x1 bf16 bwd_w jit kernel
// w/o using permw instruction inside
// TODO: consider modification/replacement for outer transformation jit kernel
struct jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_bf16_reorder_s16c_to_S16c2s)

    jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t()
        : simd_w_(16), in_stride_(16) {
        /*
        generate();
        jit_ker_ = (void (*)(bf16_support::jit_call_t *))getCode();
      */
    }

    jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t(int in_stride)
        : simd_w_(16), in_stride_(in_stride) {
        /*
        generate();
        jit_ker_ = (void (*)(bf16_support::jit_call_t *))getCode();
      */
    }

    ~jit_avx512_core_bf16_reorder_s16c_to_S16c2s_t() {}
    /*
    void generate() {
        preamble();

        xa_->mov(reg32_tail, ptr[abi_param1 + GET_OFF(mask)]);
        xa_->mov(reg_inp, ptr[abi_param1 + GET_OFF(inp)]);
        xa_->mov(reg_out, ptr[abi_param1 + GET_OFF(out)]);
        xa_->mov(reg_nelems, ptr[abi_param1 + GET_OFF(nelems)]);

        auto zmm_reg = [=](int idx) {
            assert(idx < 31);
            return Xbyak::Zmm(idx);
        };

        kmovd(ktail_mask_lo, reg32_tail);
        kshiftld(ktail_mask_hi, ktail_mask_lo, 16);

        Xbyak::Label dst_prm_table;
        xa_->mov(reg_prm, dst_prm_table);
        vmovups(zmm_prm, ptr[reg_prm]);

        constexpr int n_unroll = 2; // unroll by powers of 2 from 2^n to 2^0
        int sizeofcacheline = 2 * simd_w_ * sizeof(bfloat16_t);
        int in_stride_bytes = in_stride_ * sizeof(bfloat16_t);
        Xbyak::Label l_simd_loop[n_unroll + 2], l_simd_notail;
        for (int i = n_unroll; i >= 0; i--) {
            const int unroll = 1 << i; // 4, 2, 1
            L(l_simd_loop[i + 1]);
            {
                cmp(reg_nelems, 2 * unroll);
                jl(l_simd_loop[i], T_NEAR);
                for (int j = 0; j < unroll; j++) {
                    auto zmm_inp = zmm_reg(j);
                    if (in_stride_ == 16)
                        vmovups(zmm_inp, zword[reg_inp + j * sizeofcacheline]);
                    else {
                        vmovdqu16(zmm_inp | ktail_mask_lo | T_z,
                                zword[reg_inp + 2 * j * in_stride_bytes]);
                        vmovdqu16(zmm_inp | ktail_mask_hi,
                                zword[reg_inp + (2 * j + 1) * in_stride_bytes
                                        - 32]);
                    }
                    vpermw(zmm_inp, zmm_prm, zmm_inp);
                    vmovups(zword[reg_out + j * sizeofcacheline], zmm_inp);
                }
                add(reg_inp,
                        unroll
                                * (in_stride_ == 16 ? sizeofcacheline
                                                    : 2 * in_stride_bytes));
                add(reg_out, unroll * sizeofcacheline);

                sub(reg_nelems, 2 * unroll);
                jmp(l_simd_loop[i + 1], T_NEAR);
            }
        }
        L(l_simd_loop[0]);

        test(reg_nelems, reg_nelems);
        jz(l_simd_notail);

        auto zmm_inp = zmm_reg(0);
        vpxord(zmm_inp, zmm_inp, zmm_inp);
        vmovdqu16(zmm_inp | ktail_mask_lo | T_z, ptr[reg_inp]);
        vpermw(zmm_inp, zmm_prm, zmm_inp);
        vmovups(zword[reg_out], zmm_inp);

        L(l_simd_notail);

        postamble();

        const uint16_t dst_prm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20,
                5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13,
                29, 14, 30, 15, 31};

        align(64);
        L(dst_prm_table);
        for (size_t i = 0; i < 32; ++i)
            CodeArray::dw(dst_prm_array[i]);
        binCommit();
    }
  */

    void jit_ker(bf16_support::jit_call_t *params) const {
        jit_ker_(params);
        msan_unpoison(params->out, params->nelems * sizeof(bfloat16_t));
    }

private:
    int simd_w_;
    int in_stride_;
    void (*jit_ker_)(bf16_support::jit_call_t *);
    /*
    Xbyak::Opmask ktail_mask_lo = k2;
    Xbyak::Opmask ktail_mask_hi = k3;
    Xbyak::Zmm zmm_prm = Xbyak::Zmm(31);

    Xbyak::Reg64 reg_inp = rax;
    Xbyak::Reg64 reg_out = rbx;
    Xbyak::Reg64 reg_prm = r11;
    Xbyak::Reg64 reg_nelems = rdx;

    Xbyak::Reg32 reg32_tail = abi_not_param1.cvt32();
  */
    Xbyak_aarch64::PReg ktail_mask_lo = Xbyak_aarch64::PReg(2);
    Xbyak_aarch64::PReg ktail_mask_hi = Xbyak_aarch64::PReg(3);
    Xbyak_aarch64::ZReg zmm_prm = Xbyak_aarch64::ZReg(31);

    Xbyak_aarch64::XReg reg_inp = Xbyak_aarch64::XReg(0);
    Xbyak_aarch64::XReg reg_out = Xbyak_aarch64::XReg(1);
    Xbyak_aarch64::XReg reg_prm = Xbyak_aarch64::XReg(11);
    Xbyak_aarch64::XReg reg_nelems = Xbyak_aarch64::XReg(3);

    Xbyak_aarch64::WReg reg32_tail
            = Xbyak_aarch64::WReg(abi_not_param1.getIdx());
};

#undef GET_OFF
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
