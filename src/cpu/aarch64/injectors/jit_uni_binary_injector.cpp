/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <algorithm>
#include <bitset>
#include <cmath>

#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"

#ifdef DNNL_AARCH64
#ifdef CG
#undef CG
#endif
#define CG host_->Xbyak_aarch64::CodeGenerator
#define IDX(a) static_cast<uint32_t>(a.getIdx())
#endif //#ifdef DNNL_AARCH64

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace binary_injector {

namespace xa = Xbyak_aarch64;

std::vector<const void *> prepare_binary_args(const post_ops_t &post_ops,
        const exec_ctx_t &ctx, const unsigned first_arg_idx_offset) {
    std::vector<const void *> post_ops_binary_rhs_arg_vec;
    post_ops_binary_rhs_arg_vec.reserve(post_ops.entry_.size());

    unsigned idx = first_arg_idx_offset;
    for (const auto &post_op : post_ops.entry_) {
        if (post_op.is_binary()) {
            post_ops_binary_rhs_arg_vec.emplace_back(CTX_IN_MEM(const void *,
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
        }
        ++idx;
    }

    post_ops_binary_rhs_arg_vec.shrink_to_fit();

    return post_ops_binary_rhs_arg_vec;
}

static broadcasting_strategy_t get_rhs_arg_broadcasting_strategy(
        const memory_desc_t &rhs_arg_md, const memory_desc_wrapper &dst_d,
        bool use_per_oc_spatial_strategy = true) {
    const int ndims = rhs_arg_md.ndims;
    const auto output_dims = injector_utils::make_output_dims(dst_d);

    bool all_ones = true;
    std::bitset<5> mask(0);
    for (int d = 0; d < ndims; d++) {
        const auto &rhs_arg_dim = rhs_arg_md.dims[d];

        if (rhs_arg_dim != 1) all_ones = false;

        if (output_dims[d] != rhs_arg_md.dims[d] || output_dims[d] == 1)
            mask.set(d);
    }

    if (all_ones) {
        //unsupported
    } else if (mask.none()) {
        //unsupported
    }

    const auto &mb_rhs = rhs_arg_md.dims[0];
    const bool broadcast_per_mb = !mask.test(0);
    const bool broadcast_per_oc = !mask.test(1);

    if (broadcast_per_mb && broadcast_per_oc && mb_rhs != 1) {
        //unsupported
    } else if (broadcast_per_oc) {
        if (use_per_oc_spatial_strategy && dst_d.is_blocking_desc()) {
            const auto &strides = dst_d.blocking_desc().strides;
            return broadcasting_strategy_t::per_oc;
        } else {
            return broadcasting_strategy_t::per_oc;
        }
    }
}

bool binary_args_broadcast_supported(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return bcast_type == broadcasting_strategy_t::unsupported;
                }
                return false;
            });
}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_vmm_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_vmm_helper,
            abi_param_offset, dst_d, tail_size, tail_opmask,
            use_exact_tail_scalar_bcast, true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_vmm_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_vmm_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast, bool is_opmask_set)
    : rhs_dt_helper_vmm_idx(rhs_dt_helper_vmm_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_vmm_helper(preserve_vmm_helper)
    , abi_param_offset(abi_param_offset)
    , dst_d(dst_d)
    , tail_size(tail_size)
    , tail_opmask(tail_opmask)
    , use_exact_tail_scalar_bcast(use_exact_tail_scalar_bcast)
    , is_opmask_set_(is_opmask_set) {}

template <cpu_isa_t isa>
jit_uni_binary_injector_t<isa>::jit_uni_binary_injector_t(
        jit_generator *host, const static_params_t &static_params)
    : host_(host)
    , rhs_arg_static_params_(static_params.rhs_arg_static_params)
    , param1_(static_params.param1)
    , use_per_oc_spatial_strategy_(static_params.use_per_oc_spatial_strategy) {}

bool operator!=(
        const Xbyak_aarch64::XReg &lhs, const Xbyak_aarch64::XReg &rhs) {
    return lhs.getIdx() != rhs.getIdx();
}

bool operator!=(const Xbyak_aarch64::AdrNoOfs &lhs,
        const Xbyak_aarch64::AdrNoOfs &rhs) {
    return lhs.getXn() != rhs.getXn();
}

template <typename ParamsMap>
static bool params_differ(ParamsMap &params,
        const typename ParamsMap::key_type key1,
        const typename ParamsMap::key_type key2) {
    const auto &it1 = params.find(key1);
    const auto &it2 = params.find(key2);
    if (utils::one_of(params.end(), it1, it2)) return it1 != it2;
    return it1->second != it2->second;
}

static bool rhs_arg_params_differ(size_t vmm_idx1, size_t vmm_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_elem_off_addr = rhs_arg_params.vmm_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.vmm_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.vmm_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.vmm_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.vmm_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.vmm_idx_to_oc_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        //unsupported
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        //unsupported
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(oc_off_addr, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_val, vmm_idx1, vmm_idx2)
                || params_differ(oc_off_oprnd, vmm_idx1, vmm_idx2);
    }
}

template <cpu_isa_t isa>
int jit_uni_binary_injector_t<isa>::adjust_temp_vmm_hint(
        int user_hint, int start_idx, int end_idx, int max_vmm_idx) const {
    const bool user_hint_in_vector_range
            = user_hint >= start_idx && user_hint <= end_idx;
    const bool user_hint_exceeded_limit = user_hint > max_vmm_idx;
    const bool user_hint_invalid
            = user_hint_in_vector_range || user_hint_exceeded_limit;

    if (user_hint_invalid) {
        //unsupported
    }

    return user_hint;
}

template <cpu_isa_t isa>
std::pair<bool, int> jit_uni_binary_injector_t<isa>::should_preserve_vmm(
        int curr_idx, int vmm_hint, int max_vmm_idx,
        bool dt_helper_vmm_needed) const {
    if (dt_helper_vmm_needed && vmm_hint == curr_idx) {
        //unsupported
    }
    return std::make_pair(false, vmm_hint);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        const injector_utils::vmm_index_set_t &vmm_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {

    if (vmm_idxs.empty()) return;
    const auto start_idx = *(vmm_idxs.begin());
    const auto end_idx = *(vmm_idxs.rbegin());

    // Phase 1 Validate temporary vmm user hint
    static constexpr int max_vmm_idx = cpu_isa_traits<isa>::n_vregs - 1;
    auto &vmm_hint = rhs_arg_static_params_.rhs_dt_helper_vmm_idx;
    vmm_hint = adjust_temp_vmm_hint(vmm_hint, start_idx, end_idx, max_vmm_idx);

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy(post_op.binary.src1_desc,
                    rhs_arg_static_params_.dst_d, use_per_oc_spatial_strategy_);
    const auto rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const auto &vmm_tail_idx = rhs_arg_params.vmm_tail_idx_;
    const bool tail_exists_in_range = !vmm_tail_idx.empty();
    const bool bcast_f32_non_avx512 = !is_sve_512_
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc_spatial)
            && rhs_arg_data_type == data_type::f32;
    const bool should_preserve_vmm_tail = tail_exists_in_range
            && (!is_sve_512_
                    || !utils::one_of(rhs_broadcasting_strategy,
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc_spatial)
                    || rhs_arg_data_type != data_type::f32);
    const bool dt_helper_vmm_needed
            = !binary_op_with_unaligned_mem_operand_allowed_
            || rhs_arg_data_type != data_type::f32 || bcast_f32_non_avx512
            || should_preserve_vmm_tail;

    // Phase 2 Protect temporary registers content.
    const injector_utils::register_preserve_guard_t register_guard {host_,
            (rhs_arg_static_params_.preserve_gpr_helpers
                            ? std::initializer_list<Xbyak_aarch64::XReg>(
                                    {rhs_arg_static_params_.rhs_addr_reg,
                                            rhs_arg_static_params_
                                                    .rhs_helper_reg})
                            : std::initializer_list<Xbyak_aarch64::XReg>()),
            (rhs_arg_static_params_.preserve_vmm_helper && dt_helper_vmm_needed
                            ? std::initializer_list<Xbyak_aarch64::VReg>(
                                    {xa::VReg(vmm_hint)})
                            : std::initializer_list<Xbyak_aarch64::VReg>())};

    bool vmm0_was_preserved = false;
    static const TReg zero_vmm(0);

    Xbyak_aarch64::AdrNoOfs rhs_arg_addr(xa::XReg(0));

    // Phase 3 Apply binary post-op over all vmms.
    for (const auto vmm_idx : vmm_idxs) {
        if (vmm_idx == start_idx
                || rhs_arg_params_differ(vmm_idx, vmm_idx - 1, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(vmm_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy);
        }

        const auto local_vmm_preservation = should_preserve_vmm(
                vmm_idx, vmm_hint, max_vmm_idx, dt_helper_vmm_needed);
        const bool &vmm_preservation_needed = local_vmm_preservation.first;
        const TReg dst_vmm(vmm_idx);
        const bool with_tail = rhs_arg_static_params_.tail_size
                && vmm_tail_idx.find(vmm_idx) != vmm_tail_idx.cend()
                && IMPLICATION(rhs_broadcasting_strategy
                                == broadcasting_strategy_t::scalar,
                        rhs_arg_static_params_.use_exact_tail_scalar_bcast);

        if (vmm_preservation_needed) {
            //unsupported
        } else
            inject_binary(post_op, dst_vmm, rhs_arg_addr, with_tail);
    }
    // ...and restored afterwards
    //unsupported
}

template <cpu_isa_t isa>
Xbyak_aarch64::AdrNoOfs jit_uni_binary_injector_t<isa>::prepare_rhs_arg_addr(
        std::size_t vmm_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto rhs_arg_elem_size
            = types::data_type_size(post_op.binary.src1_desc.data_type);

    CG::add_imm(xa::XReg(28), param1_, abi_param_offset, xa::XReg(23));
    CG::ldr(rhs_addr_reg, ptr(xa::XReg(28)));
    CG::add_imm(xa::XReg(28), rhs_addr_reg, rhs_arg_idx * rhs_arg_ptr_size,
            xa::XReg(23));
    CG::ldr(rhs_addr_reg, ptr(xa::XReg(28)));

    switch (rhs_broadcasting_strategy) {
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_offset_from_operand(rhs_arg_params.vmm_idx_to_oc_off_oprnd,
                    vmm_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.vmm_idx_to_oc_elem_off_addr, vmm_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.vmm_idx_to_oc_elem_off_val,
                    vmm_idx, rhs_addr_reg, rhs_arg_elem_size);
            return xa::ptr(rhs_addr_reg);
        }
        default: assert(false && "Broadcasting type not supported");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_from_operand(
        const std::map<int, xa::XReg> &vmm_idx_to_elem_operand_off, int vmm_idx,
        const xa::XReg &addr_reg, const xa::XReg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_operand_off = vmm_idx_to_elem_operand_off.find(vmm_idx);
    if (it_operand_off != vmm_idx_to_elem_operand_off.end()) {
        if (elem_size_bytes == 1) {
            //unsupported
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            CG::mov(tmp_reg, xa::XReg(it_operand_off->second.getIdx()));
            CG::lsl(tmp_reg, tmp_reg, shift_val);
            CG::add(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_under_mem_addr(
        const std::map<int, Xbyak_aarch64::AdrNoOfs> &vmm_idx_to_elem_addr_off,
        int vmm_idx, const xa::XReg &addr_reg, const xa::XReg &tmp_reg,
        std::size_t elem_size_bytes) const {

    const auto it_off_addr = vmm_idx_to_elem_addr_off.find(vmm_idx);
    if (it_off_addr != vmm_idx_to_elem_addr_off.end()) {
        if (elem_size_bytes == 1) {
            //unsupported
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            CG::ldr(tmp_reg, xa::ptr(it_off_addr->second.getXn()));
            CG::lsl(tmp_reg, tmp_reg, shift_val);
            CG::add(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_value_offset(
        const std::map<int, int> &vmm_idx_to_elem_val_off, int vmm_idx,
        const xa::XReg &addr_reg, std::size_t elem_size_bytes) const {

    const auto it_off_val = vmm_idx_to_elem_val_off.find(vmm_idx);
    if (it_off_val != vmm_idx_to_elem_val_off.end()) {
        CG::add_imm(addr_reg, addr_reg, it_off_val->second * elem_size_bytes,
                xa::XReg(23));
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, TReg dst,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr, bool with_tail) const {

    const auto &alg = post_op.binary.alg;
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const bool scalar_f32 = false;
    const bool with_tail_not_fusable_to_binary_op
            = with_tail && !(scalar_f32 && is_sve_512_);
    const bool process_rhs_arg_using_tmp_vmm
            = rhs_arg_data_type != data_type::f32
            || (scalar_f32 && !is_sve_512_)
            || with_tail_not_fusable_to_binary_op
            || !binary_op_with_unaligned_mem_operand_allowed_;

    if (process_rhs_arg_using_tmp_vmm) {

        const TReg tmp_vmm = TReg(rhs_arg_static_params_.rhs_dt_helper_vmm_idx);

        if (false) {
            //unsupported
        } else
            load_rhs(rhs_arg_data_type, tmp_vmm, rhs_addr, with_tail);

        if (rhs_arg_data_type != data_type::bf16
                && rhs_arg_data_type != data_type::f32) {
            //unsupported
        }

        execute_binary(alg, dst, dst, tmp_vmm);
    } else {
        const auto lhs = dst;
        const bool with_tail_fusable_to_binary_op
                = with_tail && scalar_f32 && is_sve_512_;
        if (with_tail_fusable_to_binary_op) {
            assert(rhs_arg_static_params_.is_opmask_set()
                    && "Opmask is not set for tail loading avx512");
        }

        execute_binary(alg, dst, lhs, rhs_addr);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs(const dnnl_data_type_t &data_type,
        const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr,
        bool with_tail) const {
    if (with_tail)
        load_rhs_tail(data_type, tmp_reg, rhs_addr);
    else {
        //unsupported
    }
}

static constexpr int xmm_size_elem = 4;

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_tail(
        const dnnl_data_type_t &data_type, const TReg &tmp_vmm,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");

    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        CG::eor(xa::ZReg(IDX(tmp_vmm)).d, xa::ZReg(IDX(tmp_vmm)).d,
                xa::ZReg(IDX(tmp_vmm)).d);
    } else if (vlen == 32) {
        CG::ptrue(xa::PRegB(1), xa::VL32);
        CG::eor(xa::ZReg(IDX(tmp_vmm)).d, xa::ZReg(IDX(tmp_vmm)).d,
                xa::ZReg(IDX(tmp_vmm)).d);
        CG::mov(xa::ZReg(IDX(tmp_vmm)).s, xa::PReg(1) / xa::T_m, 0);
    } else if (vlen == 16) {
        CG::eor(xa::VReg16B(IDX(tmp_vmm)), xa::VReg16B(IDX(tmp_vmm)),
                xa::VReg16B(IDX(tmp_vmm)));
    } else {
        assert(!"unreachable");
    }
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            if (vlen == 64) {
                CG::pfalse(xa::PRegB(9));
                CG::zip1(xa::PRegB(1), xa::PRegB(IDX(tail_opmask)),
                        xa::PRegB(9));
                CG::zip1(xa::PRegH(1), xa::PRegH(1), xa::PRegH(9));
                CG::ld1w(xa::ZRegS(IDX(tmp_vmm)), xa::PReg(1) / xa::T_z,
                        xa::ptr(rhs_addr.getXn()));
            } else if (vlen == 32) {
                CG::sub(xa::XReg(22), xa::XReg(22), 8);
                CG::str(xa::PReg(7), xa::ptr(xa::XReg(22)));
                CG::ptrue(xa::PRegB(7));
                CG::ptrue(xa::PRegB(14), xa::VL32);
                CG::bic(xa::PRegB(7), xa::PReg(7) / xa::T_z,
                        xa::PRegB(IDX(tail_opmask)), xa::PRegB(14));
                CG::ld1w(xa::ZRegS(IDX(tmp_vmm)), xa::PReg(7) / xa::T_z,
                        xa::ptr(rhs_addr.getXn()));
                CG::ldr(xa::PReg(7), xa::ptr(xa::XReg(22)));
                CG::add(xa::XReg(22), xa::XReg(22), 8);
            } else if (vlen == 16) {
                CG::sub(xa::XReg(22), xa::XReg(22), 8);
                CG::str(xa::PReg(7), xa::ptr(xa::XReg(22)));
                CG::ptrue(xa::PRegB(7));
                CG::ptrue(xa::PRegB(13), xa::VL16);
                CG::bic(xa::PRegB(7), xa::PReg(7) / xa::T_z,
                        xa::PRegB(IDX(tail_opmask)), xa::PRegB(13));
                CG::ld1w(xa::ZRegS(IDX(tmp_vmm)), xa::PReg(7) / xa::T_z,
                        xa::ptr(rhs_addr.getXn()));
                CG::ldr(xa::PReg(7), xa::ptr(xa::XReg(22)));
                CG::add(xa::XReg(22), xa::XReg(22), 8);
            } else {
                assert(!"unreachable");
            }
            break;
        case data_type::s8:
            //unsupported
            break;
        case data_type::u8:
            //unsupported
            break;
        case data_type::bf16:
            //unsupported
            break;
        default: assert(!"unsupported data type"); break;
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_binary(alg_kind_t binary_alg,
        const TReg &dst, const TReg &lhs, const TReg &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add:
            CG::fadd(xa::ZReg(IDX(dst)).s, xa::ZReg(IDX(lhs)).s,
                    xa::ZReg(IDX(rhs)).s);
            break;
        case alg_kind::binary_mul: /*unsupported*/ break;
        case alg_kind::binary_max: /*unsupported*/ break;
        case alg_kind::binary_min: /*unsupported*/ break;
        case alg_kind::binary_div: /*unsupported*/ break;
        case alg_kind::binary_sub: /*unsupported*/ break;
        default: assert(!"unsupported algorithm"); break;
    }
}
template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_binary(alg_kind_t binary_alg,
        const TReg &dst, const TReg &lhs,
        const Xbyak_aarch64::AdrNoOfs &rhs) const {
    switch (binary_alg) {
        case alg_kind::binary_add:
            CG::ptrue(xa::PReg(1).b);
            CG::sub_imm(xa::XReg(22), xa::XReg(22), 64, xa::XReg(23));
            CG::st1w(xa::ZRegS(1), xa::PReg(1), xa::ptr(xa::XReg(22)));
            CG::ldr(xa::ZReg(1), xa::ptr(rhs.getXn()));
            CG::fadd(xa::ZReg(IDX(dst)).s, xa::ZReg(IDX(lhs)).s, xa::ZReg(1).s);
            CG::ld1w(xa::ZRegS(1), xa::PReg(1), xa::ptr(xa::XReg(22)));
            CG::add_imm(xa::XReg(22), xa::XReg(22), 64, xa::XReg(23));
            break;
        case alg_kind::binary_mul: /*unsupported*/ break;
        case alg_kind::binary_max: /*unsupported*/ break;
        case alg_kind::binary_min: /*unsupported*/ break;
        case alg_kind::binary_div: /*unsupported*/ break;
        case alg_kind::binary_sub: /*unsupported*/ break;
        default: assert(!"unsupported algorithm"); break;
    }
}

template class jit_uni_binary_injector_t<sve_512>;

} // namespace binary_injector
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
