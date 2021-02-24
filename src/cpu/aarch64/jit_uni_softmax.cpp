/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_uni_softmax.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

#if __INTEL_COMPILER && __INTEL_COMPILER < 1900
// Intel Compilers 17.x and 18.x do not like that diff_src_ptr() is only used
// in a single descendant class and marks it as unused. This breaks builds
// with DNNL_WERROR=on. Disabling the warning for this file seems to be less
// ugly than all the fixes that I came up with.
#pragma warning disable : 177
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
struct jit_softmax_base_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src, *dst, *diff_dst; // src dubs as diff_src
        size_t spat_offt_count;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_t)

    // cpu specific part
    const int vlen = cpu_isa_traits<isa>::vlen;

    const softmax_pd_t *pd_;
    const memory_desc_wrapper data_d_;

    virtual void operator()(const call_params_t *p) = 0;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> log_injector_;

    XReg reg_param = abi_param1;

#if 1
    XReg reg_exp_injector_table = x1;
    XReg reg_log_injector_table = x3;
    XReg reg_src = x8;
    XReg reg_diff_src = reg_src;
    XReg reg_dst = x9;
    XReg reg_diff_dst = x14;
    XReg reg_spat_offt = x10;
    XReg reg_spat_offt_count = x11;
    XReg reg_reverse_spat_offt = x12;
    XReg reg_tmp = x13;
#else
    XReg reg_exp_injector_table = x0;
    XReg reg_log_injector_table = x3;
    XReg reg_src = x11;
    XReg reg_diff_src = reg_src;
    XReg reg_dst = x8;
    XReg reg_diff_dst = x10;
    XReg reg_spat_offt = x9;
    XReg reg_spat_offt_count = x13;
    XReg reg_reverse_spat_offt = x12;
    WReg reg_tmp = w13;
#endif

    // PReg(2) is used by jit_softmax_t
    const PReg p_512 = p3;
    const PReg p_shuff0 = p11;
    const PReg p_shuff1 = p5;
    const PReg injector_mask = p1;
    const PReg injector_tmp = p6;

    ZReg vtmp = ZReg(27); // assigned at placed where used
    ZReg tail_vmask = ZReg(0);
    ZReg vneg_flt_max = ZReg(28);
    ZReg vone = ZReg(29);
    ZReg vsum = ZReg(30);
    ZReg vmax = ZReg(31);
    ZReg vsbr = vsum; // must be not equal to vmax

    bool is_bf16_ = false;
    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();

    size_t data_type_size_ = 0;
    size_t simd_w_ = 0;
    size_t unroll_regs_ = 4;

    size_t axis_simd_full_;
    size_t axis_simd_tail_;
    size_t n_loops_;
    size_t loop_tail_;
    size_t axis_stride_;

    /* Caution: Chose predicate registers not used by x64's implementation. */
    ZReg z_tmp0 = z23;

    void compute_predefined_variables() {
        axis_simd_full_ = pd_->axis_size() / simd_w_;
        axis_simd_tail_ = pd_->axis_size() % simd_w_;
        n_loops_ = axis_simd_full_ / unroll_regs_;
        loop_tail_ = axis_simd_full_ - n_loops_ * unroll_regs_;
        axis_stride_ = compute_axis_stride();
    }

    size_t compute_axis_stride() {
        const auto &bd = data_d_.blocking_desc();

        if (bd.inner_nblks) return data_type_size_ * bd.strides[pd_->axis()];
        return is_bf16_ ? vlen / 2 : vlen;
    }

    void load_common_params() {
        xa_->mov(WReg(IDX(reg_tmp)), float2int(1.0f));
        dup(ZRegS(IDX(vone)), WReg(IDX(reg_tmp)));
        xa_->mov(reg_tmp, float2int(-FLT_MAX));
        dup(ZRegS(IDX(vneg_flt_max)), WReg(IDX(reg_tmp)));

#define PARAM_OFF(x) offsetof(call_params_t, x)
#define PARAM_OFF_DIFF(x, y) \
    (static_cast<int32_t>(PARAM_OFF(x)) - static_cast<int32_t>(PARAM_OFF(y)))
#define LDR_PARAM(r, x, y) \
    assert(-256 <= PARAM_OFF_DIFF(x, y) && PARAM_OFF_DIFF(x, y) <= 255); \
    ldr(XReg(IDX(r)), pre_ptr(X_DEFAULT_ADDR, PARAM_OFF_DIFF(x, y)))
#define LDR_PARAM_TMP(x, y) \
    assert(-256 <= PARAM_OFF_DIFF(x, y) && PARAM_OFF_DIFF(x, y) <= 255); \
    ldr(X_TMP_0, pre_ptr(X_DEFAULT_ADDR, PARAM_OFF_DIFF(x, y)));
#define STR_PARAM_TMP(x, y) \
    assert(-256 <= static_cast<int32_t>(x) - static_cast<int32_t>(y) \
            && static_cast<int32_t>(x) - static_cast<int32_t>(y) <= 256); \
    str(X_TMP_0, pre_ptr(x_tmp_sp, x - y));

        xa_->mov(X_DEFAULT_ADDR, XReg(IDX(reg_param)));
        ldr(XReg(IDX(reg_spat_offt_count)),
                pre_ptr(X_DEFAULT_ADDR, PARAM_OFF(spat_offt_count)));
        LDR_PARAM(reg_dst, dst, spat_offt_count);
        if (pd_->is_fwd()) {
            LDR_PARAM(reg_src, src, dst);
        } else {
            LDR_PARAM(reg_diff_src, src, dst); // src is reused
            LDR_PARAM(reg_diff_dst, diff_dst, src);
        }
#undef LDR_PARAM
#undef LDR_PARAM_TMP
#undef STR_PARAM_TMP
#undef PARAM_OFF
    }

    void uni_fmax(const ZReg &dst, const ZReg &src, const ZReg &src2,
            const PReg &mask) {
        const uint32_t idxDst = dst.getIdx();
        const uint32_t idxSrc = src.getIdx();
        const uint32_t idxSrc2 = src2.getIdx();
        uint32_t pattern = 0;

        pattern += (idxDst == idxSrc) ? (1 << 2) : 0;
        pattern += (idxDst == idxSrc2) ? (1 << 1) : 0;
        pattern += (idxSrc == idxSrc2) ? 1 : 0;

        switch (pattern) {
            case 0x4: /* dst = src && dst != src2 && src != src2
		   This is the most popular case. */
                fmax(dst.s, mask / Xbyak_aarch64::T_m, src2.s);
                break;
            default: assert(!"Unreachable!"); break;
        }
    }

    void uni_fmax(const ZReg &dst, const ZReg &src, const ZReg &src2) {
        const uint32_t idxDst = dst.getIdx();
        const uint32_t idxSrc = src.getIdx();
        const uint32_t idxSrc2 = src2.getIdx();
        uint32_t pattern = 0;

        pattern += (idxDst == idxSrc) ? (1 << 2) : 0;
        pattern += (idxDst == idxSrc2) ? (1 << 1) : 0;
        pattern += (idxSrc == idxSrc2) ? 1 : 0;

        switch (pattern) {
            case 0x4: /* dst = src && dst != src2 && src != src2
		   This is the most popular case. */
                fmax(dst.s, p_512 / Xbyak_aarch64::T_m, src2.s);
                break;
            default: assert(!"Unreachable!"); break;
        }
    }

#if 1
    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
            const int disp = 0) {
        XReg x_addr = base;
        uint32_t offIdx = off.getIdx();

        if (offIdx <= SP_IDX) {
            xa_->add(X_DEFAULT_ADDR, base, off);
            x_addr = X_DEFAULT_ADDR;
        }
        if (disp) {
            add_imm(X_DEFAULT_ADDR, x_addr, disp, X_TMP_0);
            x_addr = X_DEFAULT_ADDR;
        }

        return x_addr;
    }
#endif

    XReg diff_src_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_src, reg_spat_offt, offt);
    }

    XReg src_ptr(size_t offt = 0) {
        return xreg_addr(reg_src, reg_spat_offt, offt);
    }

    XReg dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_dst, reg_spat_offt, offt);
    }

    XReg diff_dst_ptr(size_t offt = 0) {
        return xreg_addr(reg_diff_dst, reg_spat_offt, offt);
    }

    enum class op_t : unsigned { max, sum };

    void perform_op(ZReg v, ZReg vtmp, op_t op) {
        if (op == op_t::max)
            uni_fmax(v, v, vtmp);
        else if (op == op_t::sum)
            xa_->fadd(ZRegS(IDX(v)), ZRegS(IDX(v)), ZRegS(IDX(vtmp)));
    }

    template <typename body_t>
    void axis_loop(body_t body) {
        Label main_loop, tail_loop, tail_axis;

        // reverse_spat_offt to dispatch between labels
        xa_->mov(reg_reverse_spat_offt, reg_spat_offt_count);
        eor(reg_spat_offt, reg_spat_offt,
                reg_spat_offt); // spat_offt to get addr of src/dst
        L(main_loop);
        {
            if (n_loops_) {
                xa_->cmp(reg_reverse_spat_offt, unroll_regs_ * axis_stride_);
                b(LT, tail_loop);

                body(unroll_regs_, false);
                xa_->sub_imm(reg_reverse_spat_offt, reg_reverse_spat_offt,
                        unroll_regs_ * axis_stride_, X_TMP_0);
                xa_->add_imm(reg_spat_offt, reg_spat_offt,
                        unroll_regs_ * axis_stride_, X_TMP_0);
                b(main_loop);
            }
        }

        L(tail_loop);
        {
            if (loop_tail_) {
                body(loop_tail_, false);
                xa_->add_imm(reg_spat_offt, reg_spat_offt,
                        loop_tail_ * axis_stride_, X_TMP_0);
            }
        }

        L(tail_axis);
        {
            if (axis_simd_tail_) { body(1, true); }
        }
    }

    virtual void prepare_tail_mask() = 0;
    virtual void get_horizontal_op(const ZReg &v, const ZReg &vtmp, op_t op)
            = 0;
    virtual void accumulate_vmax() = 0;
    virtual void accumulate_vsum() = 0;
    virtual void compute_dst() = 0;
    virtual void initialization_hook() {}
    virtual void accumulate_vsbr() {}
    virtual void compute_diff_src() {}

    void forward() {
        accumulate_vmax();
        accumulate_vsum();
        compute_dst();
    }

    void backward() {
        accumulate_vsbr();
        compute_diff_src();
    }

    void prepare_mask() {
        if (isa == sve_512) {
            xa_->sub_imm(
                    X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 64 * 3, X_TMP_0);
            xa_->str(p_512, Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 0, MUL_VL));
            xa_->str(p_shuff0,
                    Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 1, MUL_VL));
            xa_->str(p_shuff1,
                    Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 2, MUL_VL));
            ptrue(p_512.b);
            xa_->not_(P_TMP_1.b, P_ALL_ONE, P_ALL_ONE.b);
            trn1(p_shuff0.d, P_ALL_ONE.d, P_TMP_1.d);
            trn1(p_shuff0.d, p_shuff0.d, p_shuff0.d);
            trn1(p_shuff1.s, P_ALL_ONE.s, P_TMP_1.s);
        }
    }

    void restore_mask() {
        if (isa == sve_512) {
            xa_->ldr(p_512, Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 0, MUL_VL));
            xa_->ldr(p_shuff0,
                    Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 1, MUL_VL));
            xa_->ldr(p_shuff1,
                    Xbyak_aarch64::ptr(X_TRANSLATOR_STACK, 2, MUL_VL));
            xa_->add_imm(
                    X_TRANSLATOR_STACK, X_TRANSLATOR_STACK, 64 * 3, X_TMP_0);
        }
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        if (pd_->is_fwd() || is_logsoftmax_)
            exp_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, true,
                    reg_exp_injector_table, injector_mask, p_512,
                    injector_tmp));
        if (pd_->is_fwd() && is_logsoftmax_) {
            log_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                    reg_log_injector_table, injector_mask, p_512,
                    injector_tmp));
        }

        compute_predefined_variables();
        preamble(true);
        initialization_hook();

        prepare_mask();

        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        //adr(x0, 0xe00);
        if (axis_simd_tail_) prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd())
            forward();
        else
            backward();

        restore_mask();
        postamble();
        if (exp_injector_) { exp_injector_->prepare_table(); }
        if (log_injector_) { log_injector_->prepare_table(); }
    }

    jit_softmax_base_t(const softmax_pd_t *pd)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, isa)
        , pd_(pd)
        , data_d_(pd_->dst_md()) {
        is_bf16_ = data_d_.data_type() == data_type::bf16;
        data_type_size_ = is_bf16_ ? sizeof(bfloat16_t) : sizeof(float);
        simd_w_ = vlen / sizeof(float); // bf16 works on ymms
    }
};

template <cpu_isa_t isa>
struct jit_softmax_t;

template <>
struct jit_softmax_t<sve_512> : public jit_softmax_base_t<sve_512> {
    PReg tail_opmask = PReg(2);

    void store(const XReg &addr, const ZReg &vmm, bool tail = false) {
        if (tail)
            st1w(ZRegS(IDX(vmm)), PReg(IDX(tail_opmask)) / Xbyak_aarch64::T_m,
                    Xbyak_aarch64::ptr(XReg(IDX(addr))));
        else
            str(ZReg(IDX(vmm)), Xbyak_aarch64::ptr(XReg(IDX(addr))));
    };

    void load(const ZReg &vmm, const XReg &addr, bool tail = false) {
        if (tail)
            ld1w(ZRegS(IDX(vmm)), PReg(IDX(tail_opmask)) / Xbyak_aarch64::T_z,
                    Xbyak_aarch64::ptr(XReg(IDX(addr))));
        else
            ldr(ZReg(IDX(vmm)), Xbyak_aarch64::ptr(XReg(IDX(addr))));
    };

    void prepare_tail_mask() override {
        const int sw_tail = axis_simd_tail_;
        uint32_t idx = IDX(tail_opmask);
        switch (sw_tail) {
            case 16: ptrue(PRegS(idx), VL16); break;
            case 8: ptrue(PRegS(idx), VL8); break;
            case 7: ptrue(PRegS(idx), VL7); break;
            case 6: ptrue(PRegS(idx), VL6); break;
            case 5: ptrue(PRegS(idx), VL5); break;
            case 4: ptrue(PRegS(idx), VL4); break;
            case 3: ptrue(PRegS(idx), VL3); break;
            case 2: ptrue(PRegS(idx), VL2); break;
            case 1: ptrue(PRegS(idx), VL1); break;
            default:
                index(ZRegS(IDX(vtmp)), 1, 1);
                cmple(PRegS(idx), p_512 / Xbyak_aarch64::T_z, ZRegS(IDX(vtmp)),
                        sw_tail);
                break;
        }
    }

    void get_horizontal_op(const ZReg &v, const ZReg &vtmp, op_t op) override {
        xa_->mov(ZRegS(IDX(vtmp)), P_ALL_ONE, ZRegS(IDX(v)));
        ext(ZRegB(IDX(vtmp)), ZRegB(IDX(v)), 32);
        perform_op(v, vtmp, op);
        xa_->mov(ZRegS(IDX(vtmp)), P_ALL_ONE, ZRegS(IDX(v)));
        xa_->mov(z_tmp0.s, P_ALL_ONE, ZRegS(IDX(v)));
        ext(z_tmp0.b, ZRegB(IDX(v)), 48);
        ext(ZRegB(IDX(vtmp)), ZRegB(IDX(v)), 16);
        xa_->mov(ZRegD(IDX(vtmp)), p_shuff0 / Xbyak_aarch64::T_m, z_tmp0.d);
        perform_op(v, vtmp, op);
        uzp2(z_tmp0.d, ZRegD(IDX(v)), ZRegD(IDX(v)));
        trn1(ZRegD(IDX(vtmp)), z_tmp0.d, ZRegD(IDX(v)));
        perform_op(v, vtmp, op);
        trn1(ZRegS(IDX(vtmp)), ZRegS(IDX(v)), ZRegS(IDX(v)));
        trn2(z_tmp0.s, ZRegS(IDX(v)), ZRegS(IDX(v)));
        xa_->mov(ZRegS(IDX(vtmp)), p_shuff1 / Xbyak_aarch64::T_m, z_tmp0.s);
        perform_op(v, vtmp, op);
    }

    void accumulate_vmax() override {
        // flush to -FLT_MAX before accumulation
        xa_->mov(ZRegB(IDX(vmax)), P_ALL_ONE, ZRegB(IDX(vneg_flt_max)));

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_src = ZReg(i + 1);
                load(vreg_tmp_src, src_ptr(axis_stride_ * i), tail); // SEGV
                if (tail)
                    uni_fmax(vmax, vmax, vreg_tmp_src, tail_opmask);
                else
                    uni_fmax(vmax, vmax, vreg_tmp_src);
            }
        });

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vsum() override {
        eor(vsum.d, vsum.d, vsum.d); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_src = ZReg(i + 1);
                load(vreg_tmp_src, src_ptr(axis_stride_ * i), tail);
                xa_->fsub(vreg_tmp_src.s, vreg_tmp_src.s, vmax.s);
                if (is_logsoftmax_) // store before applying exp
                    store(dst_ptr(axis_stride_ * i), vreg_tmp_src, tail);
                exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                if (tail)
                    xa_->fadd(ZRegS(IDX(vsum)),
                            PReg(IDX(tail_opmask)) / Xbyak_aarch64::T_m,
                            ZRegS(IDX(vreg_tmp_src)));
                else
                    xa_->fadd(ZRegS(IDX(vsum)), ZRegS(IDX(vsum)),
                            ZRegS(IDX(vreg_tmp_src)));
                if (is_softmax_) // store after applying exp
                    store(dst_ptr(axis_stride_ * i), vreg_tmp_src, tail);
            }
        });

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);
        if (is_softmax_) {
            xa_->mov(z_tmp0.d, ZRegD(IDX(vsum)));
            xa_->mov(ZRegD(IDX(vsum)), P_ALL_ONE, ZRegD(IDX(vone)));
            xa_->fdiv(ZRegS(IDX(vsum)), p_512 / Xbyak_aarch64::T_m, z_tmp0.s);
        }
        if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
    }

    void compute_dst() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_src = ZReg(i + 1);
                if (is_softmax_) {
                    load(vreg_tmp_src, dst_ptr(axis_stride_ * i), tail);
                    xa_->fmul(ZRegS(IDX(vreg_tmp_src)),
                            ZRegS(IDX(vreg_tmp_src)), ZRegS(IDX(vsum)));
                }
                if (is_logsoftmax_) {
                    load(vreg_tmp_src, dst_ptr(axis_stride_ * i), tail);
                    xa_->fsub(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s);
                }
                store(dst_ptr(axis_stride_ * i), vreg_tmp_src, tail);
            }
        });
    }

    void accumulate_vsbr() override {
        eor(vsbr.d, vsbr.d, vsbr.d); // flush to zero before accumulation

        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_dst = ZReg(i * 2 + 1);
                ZReg vreg_tmp_diff_dst = ZReg(i * 2 + 2);
                load(vreg_tmp_diff_dst, diff_dst_ptr(axis_stride_ * i), tail);
                if (is_softmax_) {
                    load(vreg_tmp_dst, dst_ptr(axis_stride_ * i), tail);
                    xa_->fmul(ZRegS(IDX(vreg_tmp_diff_dst)),
                            ZRegS(IDX(vreg_tmp_diff_dst)),
                            ZRegS(IDX(vreg_tmp_dst)));
                }
                xa_->fadd(ZRegS(IDX(vsbr)), ZRegS(IDX(vsbr)),
                        ZRegS(IDX(vreg_tmp_diff_dst)));
            }
        });

        get_horizontal_op(vsbr, vtmp = vmax, op_t::sum);
    }

    void compute_diff_src() override {
        axis_loop([&](int unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                ZReg vreg_tmp_dst = ZReg(i * 2 + 1);
                ZReg vreg_tmp_diff_dst = ZReg(i * 2 + 2);
                load(vreg_tmp_dst, dst_ptr(axis_stride_ * i), tail);
                load(vreg_tmp_diff_dst, diff_dst_ptr(axis_stride_ * i), tail);
                if (is_softmax_) {
                    xa_->fsub(vreg_tmp_diff_dst.s, vreg_tmp_diff_dst.s, vsbr.s);
                    xa_->fmul(ZRegS(IDX(vreg_tmp_diff_dst)),
                            ZRegS(IDX(vreg_tmp_dst)),
                            ZRegS(IDX(vreg_tmp_diff_dst)));
                }
                if (is_logsoftmax_) {
                    exp_injector_->compute_vector(vreg_tmp_dst.getIdx());
                    fmls(ZRegS(IDX(vreg_tmp_diff_dst)),
                            p_512 / Xbyak_aarch64::T_m,
                            ZRegS(IDX(vreg_tmp_dst)), ZRegS(IDX(vsbr)));
                }
                store(diff_src_ptr(axis_stride_ * i), vreg_tmp_diff_dst, tail);
            }
        });
    }

    void initialization_hook() override {
        //if (bf16_emu_) bf16_emu_->init_vcvtneps2bf16();
    }

    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_base_t(pd) {
        /*        if (is_bf16_ && !mayiuse(avx512_core_bf16))
            bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_zmm_1,
                    bf16_emu_zmm_2, bf16_emu_zmm_3, bf16_emu_gpr,
                    bf16_emu_zmm_4, bf16_emu_zmm_5)); */
    }

    void operator()(const call_params_t *p) override {
        return jit_generator::operator()(p);
    }
};

} // namespace

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(new softmax_impl::driver_t<isa>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto data_type_size = data_d.data_type() == data_type::bf16
            ? sizeof(bfloat16_t)
            : sizeof(float);
    const auto &bd = data_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto outer_stride = data_d.padded_dims()[axis] * inner_size;
    const auto outer_size = data_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size, [&](dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride) * data_type_size;
        const char *src_ptr = src + offset;
        char *dst_ptr = dst + offset;
        softmax_driver_->exec(src_ptr, dst_ptr, outer_stride);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::jit_uni_softmax_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(new softmax_impl::driver_t<isa>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::~jit_uni_softmax_bwd_t() {
    delete softmax_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const char *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->dst_md());
    const auto data_type_size = data_d.data_type() == data_type::bf16
            ? sizeof(bfloat16_t)
            : sizeof(float);
    const auto &bd = data_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto outer_stride = data_d.padded_dims()[axis] * inner_size;
    const auto outer_size = data_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size, [&](dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride) * data_type_size;
        char *diff_src_ptr = diff_src + offset;
        const char *dst_ptr = dst + offset;
        const char *diff_dst_ptr = diff_dst + offset;
        softmax_driver_->exec(
                diff_src_ptr, dst_ptr, diff_dst_ptr, outer_stride);
    });

    return status::success;
}

namespace softmax_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {

    driver_t(const softmax_pd_t *pd) : pd_(pd), ker_(pd_) {}

    void exec(const void *src, void *dst, const dim_t outer_stride) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.spat_offt_count = outer_stride * ker_.data_type_size_;
        p.src = src;
        p.dst = dst;
        ker_(&p);
    }

    void exec(void *diff_src, const void *dst, const void *diff_dst,
            const dim_t outer_stride) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.spat_offt_count = outer_stride * ker_.data_type_size_;
        p.src = diff_src;
        p.dst = dst;
        p.diff_dst = diff_dst;
        ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const softmax_pd_t *pd_;
    jit_softmax_t<isa> ker_;
};

} // namespace softmax_impl

/* struct instantiation */
template struct jit_uni_softmax_fwd_t<sve_512>;
template struct jit_uni_softmax_bwd_t<sve_512>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
