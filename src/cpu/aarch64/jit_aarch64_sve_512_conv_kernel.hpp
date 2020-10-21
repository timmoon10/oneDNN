/*******************************************************************************
* Copyright 2019-2020 FUJITSU LIMITED
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

/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_AARCH64_JIT_SVE_CONV_KERNEL_HPP
#define CPU_AARCH64_JIT_SVE_CONV_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
//[info]取り敢えずコメント化。
#include "cpu/aarch64/jit_uni_eltwise_injector.hpp"

//[info]v0.21変更をそのまま追加
#define PRFWMAX 31
#define LDRMAX 255
#define LDRWMAX 252
#define ADDMAX 4095
#define PRFMMAX 32760
#define MOVMAX 65535
/* Get vector offsets, ofs / VL(VL: 512bits = 64Bytes) */
#define VL_OFS(ofs) ((ofs) >> 6)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <typename Vmm>
struct _jit_aarch64_sve_512_conv_fwd_kernel : public jit_generator {

    _jit_aarch64_sve_512_conv_fwd_kernel(
            const jit_conv_conf_t &ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr) {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                    this, jcp.eltwise);

        generate();
        jit_ker_ = (void (*)(jit_conv_call_s *))getCode32();
    }

    ~_jit_aarch64_sve_512_conv_fwd_kernel() { delete eltwise_injector_; }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_aarch64_sve_512_conv_fwd_kernel)

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    using reg64_t = const xa::XReg;
    enum {
        typesize = sizeof(float),
        unroll_4fma = 4,
        ker_reg_base_idx = 28,
    };

    const xa::PReg reg_p_all_ones = p2;

    reg64_t param = abi_param1_aarch64;
    reg64_t reg_inp = x1; // src base addr (2d)
    reg64_t reg_ker = x2; // ker base addr (2d)
    reg64_t aux_reg_ker_d = x2; // ker addr (3d)
    reg64_t reg_out = x3; // dst base addr (2d)
    reg64_t reg_ki = x3; // d-dim loop var? (3d)
    reg64_t reg_owb = x5; // num of ow-block
    reg64_t reg_out_prf = x6; // addr for prefetch

    reg64_t aux_reg_inp = x7; // src addr (main loop)
    reg64_t reg_out_ofs = x7; // dst addr (store_output)
    reg64_t aux_reg_ker = x8; // ker addr (main loop)
    reg64_t reg_channel = x9; // reduce workload
    reg64_t reg_bias = x10; // bias addr (prepare_out)

    reg64_t aux_reg_inp_d = x11; // src addr (3d)
    reg64_t reg_oi = x11;

    reg64_t reg_kh = x12; // ker h size
    reg64_t reg_kj = x13; // ker h workload

#if 0
    reg64_t reg_tail            = aux_reg_ker;
    reg64_t reg_load_work       = reg_tail;
#endif
    /* Temporary registers for ARM insts */
    reg64_t reg_tmp_addr = x14;
    reg64_t reg_prev_bcast_addr = x15;
    reg64_t reg_prev_wei_addr = x16;
    reg64_t reg_tmp_imm = x17;

    reg64_t reg_out_org = x18; // dst base addr (3d)
    reg64_t reg_oi_org = x19; // base oi (3d)
    reg64_t aux_reg_ker_d_org = x20;
    reg64_t reg_ker_org = x22; // ker base addr (3d)
    reg64_t reg_inp_org = x23; // src base addr (3d)

    void prefetch(
            const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_aligned = ((ofs & 0xFF) == 0) ? true : false;
        if (cacheline_aligned == true) {
            xa::Prfop op = xa::PLDL1KEEP;
            switch (level) {
                case 1:
                    op = (for_load == true) ? xa::PLDL1KEEP : xa::PSTL1KEEP;
                    break;
                case 2:
                    op = (for_load == true) ? xa::PLDL2KEEP : xa::PSTL2KEEP;
                    break;
                case 3:
                    op = (for_load == true) ? xa::PLDL3KEEP : xa::PSTL3KEEP;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            if ((ofs <= PRFMMAX) && (ofs >= 0)) {
                CGA64::prfm(op, xa::ptr(in, static_cast<int32_t>(ofs)));
            } else {
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfm(op, xa::ptr(reg_tmp_addr));
            }
        } else {
            xa::PrfopSve op_sve = xa::PLDL1KEEP_SVE;
            switch (level) {
                case 1:
                    op_sve = (for_load == true) ? xa::PLDL1KEEP_SVE
                                                : xa::PSTL1KEEP_SVE;
                    break;
                case 2:
                    op_sve = (for_load == true) ? xa::PLDL2KEEP_SVE
                                                : xa::PSTL2KEEP_SVE;
                    break;
                case 3:
                    op_sve = (for_load == true) ? xa::PLDL3KEEP_SVE
                                                : xa::PSTL3KEEP_SVE;
                    break;
                default: assert(!"invalid level"); break;
            }

            if ((VL_OFS(ofs) <= PRFWMAX)
                    && (VL_OFS(ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones,
                        xa::ptr(in, static_cast<int32_t>(VL_OFS(ofs))));
            } else {
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(reg_tmp_addr));
            }
        }
    }

    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma_core(int ur_w, int pad_l, int pad_r);
    inline void compute_loop(int ur_w, int pad_l, int pad_r);

    void generate();

    inline size_t get_output_offset(int oi, int n_oc_block) {
        const bool is_nxc_layout = is_dst_layout_nxc();
        size_t ow_str = is_nxc_layout ? jcp.ngroups * jcp.oc : jcp.oc_block;
        size_t ocb_str = is_nxc_layout
                ? jcp.oc_block
                : (size_t)jcp.od * jcp.oh * jcp.ow * jcp.oc_block;

        return jcp.typesize_out * (n_oc_block * ocb_str + oi * ow_str);
    }

    inline size_t get_input_offset(int ki, int ic, int oi, int pad_l) {
        const bool is_nxc_layout = is_src_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic
                                      : (!jcp.is_1stconv ? jcp.ic_block : 1);
        size_t ic_str = !jcp.is_1stconv || is_nxc_layout
                ? 1
                : (size_t)jcp.iw * jcp.ih * jcp.id;
        size_t iw_idx = ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l;

        return jcp.typesize_in * (iw_idx * iw_str + ic * ic_str);
    }

    inline int get_kernel_offset(
            int ki, int ic, int n_oc_block, int ker_number) {
        return jcp.typesize_in * jcp.oc_block
                * (n_oc_block * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw
                                * jcp.kd
                        + (ic + ker_number) + ki * jcp.ic_block);
    }

    inline int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }

    inline int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w
                - nstl::max(0,
                        utils::div_up(
                                pad_r - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1),
                                jcp.stride_w));
    }
    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_dst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

struct jit_aarch64_sve_512_conv_fwd_kernel {

    jit_aarch64_sve_512_conv_fwd_kernel(
            const jit_conv_conf_t ajcp, const primitive_attr_t &attr)
        : jit_ker(nullptr), sve_512_kernel_(nullptr) {
        switch (ajcp.oc_block) {
            case 16:
                sve_512_kernel_
                        = new _jit_aarch64_sve_512_conv_fwd_kernel<xa::ZReg>(
                                ajcp, attr);
                jit_ker = sve_512_kernel_->jit_ker_;
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    ~jit_aarch64_sve_512_conv_fwd_kernel() { delete sve_512_kernel_; }

    enum { typesize = sizeof(float) };

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_pd,
            memory_desc_t &weights_pd, memory_desc_t &dst_pd,
            memory_desc_t &bias_pd, const primitive_attr_t &attr, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void (*jit_ker)(jit_conv_call_s *);
    _jit_aarch64_sve_512_conv_fwd_kernel<xa::ZReg> *sve_512_kernel_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_aarch64_sve_512_conv_fwd_kernel);
};

template <typename Vmm>
struct _jit_aarch64_sve_512_conv_bwd_data_kernel_f32 : public jit_generator {

    _jit_aarch64_sve_512_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp)
        : jcp(ajcp) {
        generate();
        jit_ker_ = (void (*)(jit_conv_call_s *))getCode32();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_aarch64_sve_512_conv_bwd_data_kernel_f32)
    jit_conv_conf_t jcp;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    using reg64_t = const xa::XReg;
    enum {
        typesize = sizeof(float),
    };
    int ker_reg_base_idx = (jcp.nb_ic_blocking == 1) ? 16 : 24;

    //[info]v0.21のcodeを追加。v1.6追加codeは未反映。
    //[info]取り敢えずv0.21のcodeを追加したが、全面書き換えが必要か？
    reg64_t param = abi_param1_aarch64;
    reg64_t reg_dst = x1;
    reg64_t reg_ker = x2;
    reg64_t reg_src = x3;

    reg64_t reg_dst_prf = x23;
    reg64_t reg_ker_prf = x5;
    reg64_t reg_src_prf = x6;
    reg64_t reg_iwb = x24;

    reg64_t aux_reg_dst = x7;
    reg64_t aux_reg_ker = x8;

    reg64_t aux_reg_dst_prf = x9;
    reg64_t aux_reg_ker_prf = x10;

    reg64_t aux_reg_dst_d_prf = x6;
    reg64_t aux_reg_dst_d = x11;
    reg64_t aux_reg_ker_d_prf = x12;
    reg64_t aux_reg_ker_d = x2;
    reg64_t reg_ki = x3;

    reg64_t reg_kj = x13;
    reg64_t reg_oi = x11;
    reg64_t reg_kh = x12;

    reg64_t reg_channel = x9;

    reg64_t reg_tmp = x14;
    reg64_t reg_long_offt = x7;

    /* Temporary registers for ARM insts */
    reg64_t reg_prev_bcast_addr = x15;
    reg64_t reg_prev_bcast_addr2 = x17;
    reg64_t reg_prev_bcast_addr3 = x21;
    reg64_t reg_tmp_imm = x16;
    reg64_t reg_tmp_addr = x18;

    reg64_t reg_src_prf_org = x19;
    reg64_t reg_src_org = x20;
    reg64_t reg_oi_org = x25;
    //[info]レジスタ割り当ては適当
    reg64_t reg_dst_org = x22;
    reg64_t reg_ker_org = x26;
    reg64_t reg_input_org = x22;
    reg64_t reg_kernel_org = x26;

    const xa::PReg reg_p_all_ones = p2;

    long long int prefetch(const std::string prfop, int level, reg64_t in,
            long long int ofs, long long int prev_ofs) {
        bool for_load;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_alinged = ((ofs & 0xFF) == 0) ? true : false;
        if (cacheline_alinged == true) {
            xa::Prfop op = xa::PLDL1KEEP;
            switch (level) {
                case 1:
                    op = (for_load == true) ? xa::PLDL1KEEP : xa::PSTL1KEEP;
                    break;
                case 2:
                    op = (for_load == true) ? xa::PLDL2KEEP : xa::PSTL2KEEP;
                    break;
                case 3:
                    op = (for_load == true) ? xa::PLDL3KEEP : xa::PSTL3KEEP;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            long long int tmp_ofs = ofs - prev_ofs;
            if ((ofs <= PRFMMAX) && (ofs >= 0)) {
                CGA64::prfm(op, xa::ptr(in, static_cast<int32_t>(ofs)));
            } else if ((tmp_ofs <= PRFMMAX) && (tmp_ofs >= 0)) {
                CGA64::prfm(op,
                        xa::ptr(reg_tmp_addr, static_cast<int32_t>(tmp_ofs)));
            } else {
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfm(op, xa::ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        } else {
            xa::PrfopSve op_sve = xa::PLDL1KEEP_SVE;
            switch (level) {
                case 1:
                    op_sve = (for_load == true) ? xa::PLDL1KEEP_SVE
                                                : xa::PSTL1KEEP_SVE;
                    break;
                case 2:
                    op_sve = (for_load == true) ? xa::PLDL2KEEP_SVE
                                                : xa::PSTL2KEEP_SVE;
                    break;
                case 3:
                    op_sve = (for_load == true) ? xa::PLDL3KEEP_SVE
                                                : xa::PSTL3KEEP_SVE;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            long long int tmp_ofs = ofs - prev_ofs;
            if ((VL_OFS(ofs) <= PRFWMAX)
                    && (VL_OFS(ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones,
                        xa::ptr(in, static_cast<int32_t>(VL_OFS(ofs))));
            } else if ((VL_OFS(tmp_ofs) <= PRFWMAX)
                    && (VL_OFS(tmp_ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones,
                        xa::ptr(reg_tmp_addr,
                                static_cast<int32_t>(VL_OFS(tmp_ofs))));
            } else {
                CGA64::add_imm(reg_tmp_addr, in, ofs, reg_tmp_imm);
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(reg_tmp_addr));
                prev_ofs = ofs;
            }
        }
        return prev_ofs;
    }

    xa::ZReg reg_wei = xa::ZReg(31);

    inline void prepare_output(int ur_w);
    inline void store_output(int ur_w);
    inline void compute_loop_fma(int ur_w, int l_overflow, int r_overflow);
    inline void compute_loop_fma_core(
            int ur_w, int l_overflow, int r_overflow, int k_offset);
    inline void compute_loop(
            int ur_w, int l_overflow, int r_overflow, int k_offset = 0);
    void generate();

    inline int get_iw_start(int ki, int l_overflow) {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow) {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }

    inline size_t get_diff_src_offset(int iw, int icb) {
        const bool is_nxc_layout = is_dsrc_layout_nxc();
        size_t iw_str = is_nxc_layout ? jcp.ngroups * jcp.ic : jcp.ic_block;
        size_t icb_str = is_nxc_layout
                ? jcp.ic_block
                : (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ic_block;

        return typesize * (icb * icb_str + iw * iw_str);
    }

    inline ptrdiff_t get_dst_offset(int iw, int oc, int kw) {
        ptrdiff_t ow
                = (iw + jcp.l_pad - kw * (jcp.dilate_w + 1)) / jcp.stride_w;
        ptrdiff_t ow_str
                = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;

        return typesize * (ow * ow_str + oc);
    };

    inline bool is_dsrc_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
};

struct jit_aarch64_sve_512_conv_bwd_data_kernel_f32 {

    jit_aarch64_sve_512_conv_bwd_data_kernel_f32(const jit_conv_conf_t &ajcp)
        : jit_ker(nullptr), sve_512_kernel_(nullptr) {
        switch (ajcp.ic_block) {
            case 16:
                sve_512_kernel_
                        = new _jit_aarch64_sve_512_conv_bwd_data_kernel_f32<
                                xa::ZReg>(ajcp);
                jit_ker = sve_512_kernel_->jit_ker_;
                return;
            default: assert(!"invalid channel blocking");
        }
    }

    ~jit_aarch64_sve_512_conv_bwd_data_kernel_f32() { delete sve_512_kernel_; }

    enum { typesize = sizeof(float) };

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &diff_src_d,
            memory_desc_t &weights_d, memory_desc_t &diff_dst_d, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    void (*jit_ker)(jit_conv_call_s *);
    _jit_aarch64_sve_512_conv_bwd_data_kernel_f32<xa::ZReg> *sve_512_kernel_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_aarch64_sve_512_conv_bwd_data_kernel_f32);
};

struct jit_aarch64_sve_512_conv_bwd_weights_kernel_f32 : public jit_generator {

    jit_aarch64_sve_512_conv_bwd_weights_kernel_f32(const jit_conv_conf_t &ajcp)
        : jit_generator(nullptr, 1024 * 1024), jcp(ajcp) {
        if (jcp.harness != harness_nxc) {
            generate();
            jit_ker = (void (*)(jit_conv_call_s *))getCode32();
        } else {
#if 0
            generate_microkernel();
            jit_microker = (void (*)(float *, const float *, const float *,
                    int64_t, int64_t))getCode32();
#else
            //[info]取り敢えずマスク
            assert(!"none microkernel");
#endif
        }
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_aarch64_sve_512_conv_bwd_weights_kernel_f32)

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &diff_weights_md, memory_desc_t &diff_bias_md,
            memory_desc_t &diff_dst_md, int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);
    void (*jit_microker)(
            float *, const float *, const float *, int64_t, int64_t);

private:
    using reg64_t = const xa::XReg;
    enum { typesize = sizeof(float) };
    static const int max_ur_w;
    static const int min_oh_reduce;

    reg64_t param = abi_param1_aarch64;
    reg64_t reg_input = x1;
    reg64_t reg_kernel = x2;
    reg64_t reg_output = x3;
    reg64_t b_ic = x20;
    reg64_t kj = x5;
    reg64_t reg_kh = x6;
    reg64_t reg_ur_w_trips = x7;
    reg64_t reg_oj = x8;
    reg64_t reg_tmp = x10;
    reg64_t reg_icb = x9;

    reg64_t ki = x11;
    reg64_t reg_kd_count = x12;
    reg64_t reg_oi = x12;
    reg64_t reg_d_index = x13;
    reg64_t reg_input_d = x8;
    reg64_t reg_output_d = x9;
    reg64_t aux_reg_input = x12;
    reg64_t aux_reg_kernel = x13;
    reg64_t reg_bias = x9;
    reg64_t reg_oc_tail = x10;

    /* Temporary registers */
    reg64_t reg_add_tmp = x14;
    reg64_t reg_tmp_imm = x15;

    reg64_t reg_kd_count_org = x16;
    reg64_t reg_input_d_org = x17;
    reg64_t reg_output_d_org = x18;
    reg64_t reg_d_index_org = x19;

    reg64_t reg_input_org = x24;
    reg64_t reg_kernel_org = x22;
    reg64_t reg_output_org = x23;

    reg64_t reg_pre_addr_input = x25;
    reg64_t reg_pre_addr_out = x26;
    reg64_t reg_pre_addr_ker = x26;
    reg64_t reg_ker_start_addr = x27;

    const xa::PReg reg_p_all_ones = p2;

    void prefetch(
            const std::string prfop, int level, reg64_t in, long long int ofs) {
        bool for_load;
        if (prfop == "LD") {
            for_load = true;
        } else if (prfop == "ST") {
            for_load = false;
        } else {
            assert(!"invalid prfop");
        }

        bool cacheline_alinged = ((ofs & 0xFF) == 0) ? true : false;
        if (cacheline_alinged == true) {
            xa::Prfop op;
            switch (level) {
                case 1:
                    op = (for_load == true) ? xa::PLDL1KEEP : xa::PSTL1KEEP;
                    break;
                case 2:
                    op = (for_load == true) ? xa::PLDL2KEEP : xa::PSTL2KEEP;
                    break;
                case 3:
                    op = (for_load == true) ? xa::PLDL3KEEP : xa::PSTL3KEEP;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            if ((ofs <= PRFMMAX) && (ofs >= 0)) {
                CGA64::prfm(op, xa::ptr(in, static_cast<int32_t>(ofs)));
            } else {
                CGA64::add_imm(reg_add_tmp, in, ofs, reg_tmp_imm);
                CGA64::prfm(op, xa::ptr(reg_add_tmp));
            }
        } else {
            xa::PrfopSve op_sve;
            switch (level) {
                case 1:
                    op_sve = (for_load == true) ? xa::PLDL1KEEP_SVE
                                                : xa::PSTL1KEEP_SVE;
                    break;
                case 2:
                    op_sve = (for_load == true) ? xa::PLDL2KEEP_SVE
                                                : xa::PSTL2KEEP_SVE;
                    break;
                case 3:
                    op_sve = (for_load == true) ? xa::PLDL3KEEP_SVE
                                                : xa::PSTL3KEEP_SVE;
                    break;
                default: assert(!"invalid prfop"); break;
            }

            if ((VL_OFS(ofs) <= PRFWMAX)
                    && (VL_OFS(ofs) >= (-1 * PRFWMAX - 1))) {
                CGA64::prfw(op_sve, reg_p_all_ones,
                        xa::ptr(in, static_cast<int32_t>(VL_OFS(ofs))));
            } else {
                CGA64::add_imm(reg_add_tmp, in, ofs, reg_tmp_imm);
                CGA64::prfw(op_sve, reg_p_all_ones, xa::ptr(reg_add_tmp));
            }
        }
    }

    inline void bias_kernel_2d();
    inline void bias_kernel_3d();
    inline void maybe_zero_kernel();
    inline void compute_oh_step_unroll_ow_icblock(
            int ic_block_step, int max_ur_w);
    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset, bool input_wraparound = false);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_disp();
    inline void compute_oh_loop_common();
    inline void compute_oh_loop_partial();
    inline void compute_od_loop_partial();

    inline bool compute_full_spat_loop();
    inline bool flat_4ops_compute();

    inline void compute_loop();
    inline bool is_src_layout_nxc() {
        return utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }
    inline bool is_ddst_layout_nxc() {
        return utils::one_of(jcp.dst_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc);
    }

    inline ptrdiff_t get_full_src_offset(
            int i_iw, int i_ic, ptrdiff_t input_offset) {
        const bool is_nxc_layout = is_src_layout_nxc();
        const size_t w_shift_st
                = (jcp.is_hw_transp ? jcp.iw : 1) * jcp.ic_block;
        ptrdiff_t w_shift = is_nxc_layout ? jcp.ngroups * jcp.ic
                                          : (jcp.is_1stconv ? 1 : w_shift_st);
        ptrdiff_t ic_shift = (jcp.is_1stconv && !is_nxc_layout
                        ? (ptrdiff_t)jcp.ih * jcp.iw * jcp.id
                        : 1);

        ptrdiff_t local_input_offset = i_iw * w_shift + i_ic * ic_shift;
        return input_offset + typesize * local_input_offset;
    };

    inline int get_iw_idx(int ow, int kw, int l_pad) {
        return ow * jcp.stride_w + kw * (jcp.dilate_w + 1) - l_pad;
    }

    void generate();
#if 0
//[info]取り敢えずマスク
    void generate_microkernel();
#endif

    static void balance(const jit_conv_conf_t &j, int &nthr, int &nthr_mb,
            int &nthr_g, int &nthr_oc_b, int &nthr_ic_b, int nthreads);
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
