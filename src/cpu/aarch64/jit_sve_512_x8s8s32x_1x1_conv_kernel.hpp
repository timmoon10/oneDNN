/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_SVE512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP
#define CPU_AARCH64_JIT_SVE512_CORE_X8S8S32X_1X1_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

template <typename Vmm>
struct _jit_sve_512_x8s8s32x_1x1_conv_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_sve_512_x8s8s32x_1x1_conv_fwd_ker_t)
    _jit_sve_512_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jit_generator(nullptr, 1024 * 1024)
        , jcp(ajcp)
        , attr_(attr)
        , eltwise_injector_(nullptr) {
        if (jcp.with_eltwise) {
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);
        }
    }

    ~_jit_sve_512_x8s8s32x_1x1_conv_kernel() { delete eltwise_injector_; }

    bool maybe_eltwise(int position);
    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    using reg64_t = const XReg;
    using zmm_t = const ZReg;
    using mask_t = const PReg;

    /* register mapping */
    const XReg reg_last_load = x8;
    const XReg reg_bcast_data = x8;
    const XReg reg_ptr_scales = x8;
    const XReg reg_ptr_saturation_ubound = x8;
    const XReg reg_output_data = x9;
    const XReg reg_load_data = x10;
    const XReg reg_ptr_sum_scale = x10;
    const XReg reg_reduce_loop_work = x11;
    const XReg reg_bias_data = x12;
    const XReg reg_comp_data = x13;
    const XReg reg_scratch = x13;
    const XReg aux_reg_bcast_data = x14;
    const XReg aux_reg_load_data = x15;
    const XReg imm_addr64 = x15;
    const XReg reg_reduce_pos_flag = x7; //rax;
    const XReg aux1_reg_bcast_data = x3; //rbx;
    const XReg reg_bcast_loop_work = x3; //rbx;
    const XReg bcast_loop_iter = x2; //rdx; // Note: Fix me
    const XReg reg_load_loop_work = x6; //rsi;
    const XReg reg_rsp = x21; //rsp;
    const XReg aux_reg_output_data = x1; //abi_not_param1;
    const XReg reduce_loop_iter = x5; //abi_param1;
    const XReg reg_abi_param1 = x5; // abi_param1
    // zero-point computation
    const XReg reg_zp_compensation = aux_reg_load_data; // x15
    const XReg reg_src_zero_point = aux_reg_bcast_data; // x14
    const XReg reg_dst_zero_point = reg_src_zero_point;

    const PReg ktail_load_mask = p5;
    const PReg ktail_mask = p3;
    const PReg vmask = p4;
    const PReg mask_tmp = p8;
    const PReg mask_all_zero = p9;

    /* Temporay registers */
    const XReg reg_tmp0_imm = x18; // tmp for add_imm
    const XReg reg_tmp1_imm = x19; // tmp for add_imm
    const XReg reg_tmp2_imm = x20; // tmp for add_imm
    const XReg reg_tmp3_imm = x27; // tmp for add_imm
    const XReg reg_tmp0_adr = x23; // tmp for address value
    const XReg reg_tmp1_adr = x24; // tmp for address value
    const XReg reg_tmp2_adr = x25; // tmp for address value
    const XReg reg_tmp3_adr = x26; // tmp for address value

    const ZReg vmm_tmp = ZReg(28);
    const ZReg vmm_saturation = ZReg(28);
    const ZReg vmm_one = ZReg(29);
    const ZReg vmm_zero = ZReg(30);
    const ZReg vmm_prev_dst = ZReg(30);
    const ZReg vmm_shift = ZReg(30);
    const ZReg vmm_bcast = ZReg(31);
    const ZReg vmm_bcast2 = ZReg(30);
    /* zero-point */
    const ZReg vmm_zp = ZReg(30);
    const ZReg vmm_zp2 = ZReg(27);
    const ZReg vmm_zp_tmp = vmm_zp;

    int bcast_loop_work_off = 0;
    int reg_bias_data_off = 8;
    int reg_bcast_data_off = 16;
    int reg_load_data_off = 24;
    int reg_ptr_sum_scale_off = 32;
    int reg_comp_data_off = 40;
    int reg_zp_compensation_off = 48;
    int reg_src_zero_point_off = 56;
    int reg_dst_zero_point_off = 64;
    int stack_space_needed = 72;

    void bcast_loop(int load_loop_blk);
    void reduce_loop(int load_loop_blk, int ur, int substep, bool wraparound);

    void generate() override;
    // void cvt2ps(data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
    //         bool mask_flag);

    int get_offset(int raw_offt) {

        assert(raw_offt <= INT_MAX);
        auto offt = static_cast<int>(raw_offt);

        int scale = 0;
        const int EVEX_max_8b_offt = 0x200;

        if (EVEX_max_8b_offt <= offt && offt < 3 * EVEX_max_8b_offt) {
            offt = offt - 2 * EVEX_max_8b_offt;
            scale = 1;
        } else if (3 * EVEX_max_8b_offt <= offt
                && offt < 5 * EVEX_max_8b_offt) {
            offt = offt - 4 * EVEX_max_8b_offt;
            scale = 2;
        }

        auto re = offt;
        if (scale) re = re + (2 * EVEX_max_8b_offt) * scale;

        return re;
    }

    XReg get_comp_addr_reg(XReg base, int offset = 0) {
        auto offt = get_offset(offset);

        if (offt == 0) return base;

        auto reg_tmp_adr = reg_tmp0_adr;
        auto reg_tmp_imm = reg_tmp0_imm;
        add_imm(reg_tmp_adr, base, offt, reg_tmp_imm);

        return reg_tmp_adr;
    }
};

struct jit_sve_512_x8s8s32x_1x1_conv_kernel {
    jit_sve_512_x8s8s32x_1x1_conv_kernel(
            const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : kernel_(nullptr) {
        int ch_block = ajcp.ic_block;
        switch (ch_block) {
            case 16:
                kernel_ = new _jit_sve_512_x8s8s32x_1x1_conv_kernel<ZReg>(
                        ajcp, attr);
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    status_t create_kernel() { return kernel_->create_kernel(); }

    ~jit_sve_512_x8s8s32x_1x1_conv_kernel() { delete kernel_; }

    static bool post_ops_ok(
            jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t *&src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads,
            bool reduce_src);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr);

    void operator()(const jit_1x1_conv_call_s *p) const { (*kernel_)(p); }
    const uint8_t *jit_ker() const { return kernel_->jit_ker(); }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_sve_512_x8s8s32x_1x1_conv_kernel);
    jit_generator *kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
