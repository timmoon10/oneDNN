/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
* Copyright 2020-2021 FUJITSU LIMITED
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
#include <cmath>

#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"

#define IDX(a) static_cast<uint32_t>(a.getIdx())

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace binary_injector {

bool is_data_supported(cpu_isa_t isa, data_type_t data_type) {
    return data_type == data_type::bf16 ? false : true;
}

bool is_bcast_supported(const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
            src1_desc, dst_d, supported_strategy_set);
    return bcast_type != broadcasting_strategy_t::unsupported;
}

bool is_supported(cpu_isa_t isa, const dnnl::impl::memory_desc_t &src1_desc,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {
    return is_data_supported(isa, src1_desc.data_type)
            && is_bcast_supported(src1_desc, dst_d, supported_strategy_set);
}

bool binary_args_broadcast_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const bcast_set_t &supported_strategy_set) {

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d,
                            supported_strategy_set);
                    return bcast_type == broadcasting_strategy_t::unsupported;
                }
                return false;
            });
}

bool binary_args_tail_supported(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d, int vlen,
        const bcast_set_t &supported_strategy_set) {
    const auto channels = dst_d.dims()[1];
    const int treg_l_len = vlen / 4;

    return std::none_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return utils::one_of(bcast_type,
                                   broadcasting_strategy_t::per_oc,
                                   broadcasting_strategy_t::per_oc_spatial)
                            && (channels % treg_l_len != 0);
                }
                return false;
            });
}

bool binary_args_matches_tag(format_tag_t tag, const post_ops_t &post_ops) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) {
                if (entry.is_binary()) {
                    const memory_desc_wrapper rhs_arg_d(entry.binary.src1_desc);
                    return rhs_arg_d.matches_tag(tag);
                }
                return true;
            });
}

bool any_binary_postop_rhs_per_oc_broadcast(
        const post_ops_t &post_ops, const memory_desc_wrapper &dst_d) {
    return std::any_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    return bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                            == broadcasting_strategy_t::per_oc_spatial;
                }
                return false;
            });
}

bool all_binary_postop_rhs_per_oc_broadcast(const post_ops_t &post_ops,
        const memory_desc_wrapper &dst_d,
        const std::function<bool(const memory_desc_wrapper &)> predicate) {
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [&](const post_ops_t::entry_t &entry) -> bool {
                if (entry.is_binary()) {
                    const auto bcast_type = get_rhs_arg_broadcasting_strategy(
                            entry.binary.src1_desc, dst_d);
                    if (bcast_type == broadcasting_strategy_t::per_oc
                            || bcast_type
                                    == broadcasting_strategy_t::per_oc_spatial)
                        return predicate(
                                memory_desc_wrapper(entry.binary.src1_desc));
                }
                return true;
            });
}

static_params_t::static_params_t(const Xbyak_aarch64::XReg &param1,
        const bcast_set_t &supported_strategy_set,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : param1(param1)
    , supported_strategy_set(supported_strategy_set)
    , rhs_arg_static_params(rhs_arg_static_params) {}

static_params_t::static_params_t(const Xbyak_aarch64::XReg &param1,
        const rhs_arg_static_params_t &rhs_arg_static_params)
    : static_params_t(param1,
            bcast_set_t {broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::no_broadcast},
            rhs_arg_static_params) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_treg_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_treg_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_treg_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_treg_helper,
            abi_param_offset, dst_d, tail_size, Xbyak_aarch64::PReg(2),
            use_exact_tail_scalar_bcast, false /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_treg_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_treg_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast)
    : rhs_arg_static_params_t(rhs_dt_helper_treg_idx, rhs_addr_reg,
            rhs_helper_reg, preserve_gpr_helpers, preserve_treg_helper,
            abi_param_offset, dst_d, tail_size, tail_opmask,
            use_exact_tail_scalar_bcast, true /*is_opmask_set*/) {}

rhs_arg_static_params_t::rhs_arg_static_params_t(
        std::size_t rhs_dt_helper_treg_idx,
        const Xbyak_aarch64::XReg &rhs_addr_reg,
        const Xbyak_aarch64::XReg &rhs_helper_reg, bool preserve_gpr_helpers,
        bool preserve_treg_helper, std::size_t abi_param_offset,
        const memory_desc_wrapper &dst_d, std::size_t tail_size,
        const Xbyak_aarch64::PReg &tail_opmask,
        bool use_exact_tail_scalar_bcast, bool is_opmask_set)
    : rhs_dt_helper_treg_idx(rhs_dt_helper_treg_idx)
    , rhs_addr_reg(rhs_addr_reg)
    , rhs_helper_reg(rhs_helper_reg)
    , preserve_gpr_helpers(preserve_gpr_helpers)
    , preserve_treg_helper(preserve_treg_helper)
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
    , supported_strategy_set_(static_params.supported_strategy_set) {}

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

static bool rhs_arg_params_differ(size_t treg_idx1, size_t treg_idx2,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        broadcasting_strategy_t rhs_broadcasting_strategy) {

    const auto &out_elem_off_addr
            = rhs_arg_params.treg_idx_to_out_elem_off_addr;
    const auto &out_elem_off_val = rhs_arg_params.treg_idx_to_out_elem_off_val;
    const auto &out_off_oprnd = rhs_arg_params.treg_idx_to_out_off_oprnd;
    const auto &oc_off_addr = rhs_arg_params.treg_idx_to_oc_elem_off_addr;
    const auto &oc_off_val = rhs_arg_params.treg_idx_to_oc_elem_off_val;
    const auto &oc_off_oprnd = rhs_arg_params.treg_idx_to_oc_off_oprnd;

    if (rhs_broadcasting_strategy == broadcasting_strategy_t::scalar) {
        return false;
    } else if (rhs_broadcasting_strategy
            == broadcasting_strategy_t::no_broadcast) {
        return params_differ(out_elem_off_addr, treg_idx1, treg_idx2)
                || params_differ(out_elem_off_val, treg_idx1, treg_idx2)
                || params_differ(out_off_oprnd, treg_idx1, treg_idx2);
    } else if (rhs_broadcasting_strategy == broadcasting_strategy_t::per_oc
            || rhs_broadcasting_strategy
                    == broadcasting_strategy_t::per_oc_spatial) {
        return params_differ(oc_off_addr, treg_idx1, treg_idx2)
                || params_differ(oc_off_val, treg_idx1, treg_idx2)
                || params_differ(oc_off_oprnd, treg_idx1, treg_idx2);
    }
    return true;
}

template <cpu_isa_t isa>
int jit_uni_binary_injector_t<isa>::adjust_temp_treg_hint(
        int user_hint, int start_idx, int end_idx, int max_treg_idx) const {
    const bool user_hint_in_vector_range
            = user_hint >= start_idx && user_hint <= end_idx;
    const bool user_hint_exceeded_limit = user_hint > max_treg_idx;
    const bool user_hint_invalid
            = user_hint_in_vector_range || user_hint_exceeded_limit;

    if (user_hint_invalid) {
        const bool max_treg_idx_in_vector_range
                = max_treg_idx >= start_idx && max_treg_idx <= end_idx;

        if (max_treg_idx_in_vector_range || user_hint_exceeded_limit
                || user_hint == max_treg_idx)
            return 0;
        else
            return max_treg_idx;
    }

    return user_hint;
}

template <typename TReg>
static void push_treg(jit_generator *host, const TReg &treg) {
    host->sub_imm(host->X_SP, host->X_SP,
            injector_utils::treg_size_t<TReg>::bytes, host->X_TMP_0);

    if (treg.getBit() != 16) /* TReg is ZReg. */
        host->str(treg, Xbyak_aarch64::ptr(host->X_SP));
    else /* TReg is VReg. */
        host->str(Xbyak_aarch64::QReg(treg.getIdx()),
                Xbyak_aarch64::ptr(host->X_SP));
}

template <typename TReg>
static void pop_treg(jit_generator *host, const TReg &treg) {
    if (treg.getBit() != 16) /* TReg is ZReg. */
        host->ldr(treg, Xbyak_aarch64::ptr(host->X_SP));
    else /* TReg is VReg. */
        host->ldr(Xbyak_aarch64::QReg(treg.getIdx()),
                Xbyak_aarch64::ptr(host->X_SP));

    host->add_imm(host->X_SP, host->X_SP,
            injector_utils::treg_size_t<TReg>::bytes, host->X_TMP_0);
}

static void pop_opmask(jit_generator *host, const Xbyak_aarch64::PReg &p) {
    static constexpr int p_mask_size = 8;
    if (mayiuse(sve_512)) {
        host->ldr(p, Xbyak_aarch64::ptr(host->X_SP));
        host->add(host->X_SP, host->X_SP, p_mask_size);
    } else {
        assert(!"unreachable");
    }
}

template <typename TReg>
static void restore_stack(jit_generator *host, const TReg &treg) {
    host->add(host->X_SP, injector_utils::treg_size_t<TReg>::bytes);
}

template <cpu_isa_t isa>
std::pair<bool, int> jit_uni_binary_injector_t<isa>::should_preserve_treg(
        int curr_idx, int treg_hint, int max_treg_idx,
        bool dt_helper_treg_needed) const {
    if (dt_helper_treg_needed && treg_hint == curr_idx) {
        if (curr_idx == 0)
            return std::make_pair(true, max_treg_idx);
        else
            return std::make_pair(true, 0);
    }
    return std::make_pair(false, treg_hint);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(size_t start_idx,
        size_t end_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    injector_utils::treg_index_set_t treg_idxs;
    for (size_t i = start_idx; i < end_idx; i++)
        treg_idxs.emplace(i);
    compute_vector_range(treg_idxs, rhs_arg_idx, post_op, rhs_arg_params);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::compute_vector_range(
        const injector_utils::treg_index_set_t &treg_idxs,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {

    if (treg_idxs.empty()) return;
    const auto start_idx = *(treg_idxs.begin());
    const auto end_idx = *(treg_idxs.rbegin());

    // Phase 1 Validate temporary treg user hint
    static constexpr int max_treg_idx = cpu_isa_traits<isa>::n_vregs - 1;
    auto &treg_hint = rhs_arg_static_params_.rhs_dt_helper_treg_idx;
    treg_hint = adjust_temp_treg_hint(
            treg_hint, start_idx, end_idx, max_treg_idx);

    const auto rhs_broadcasting_strategy
            = get_rhs_arg_broadcasting_strategy(post_op.binary.src1_desc,
                    rhs_arg_static_params_.dst_d, supported_strategy_set_);
    const auto rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    const auto &treg_tail_idx = rhs_arg_params.treg_tail_idx_;
    const bool tail_exists_in_range = !treg_tail_idx.empty();
    const bool bcast_f32_non_sve_512 = !is_sve_512_
            && utils::one_of(rhs_broadcasting_strategy,
                    broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::per_oc_spatial)
            && rhs_arg_data_type == data_type::f32;
    const bool should_preserve_treg_tail = tail_exists_in_range
            && (!is_sve_512_
                    || !utils::one_of(rhs_broadcasting_strategy,
                            broadcasting_strategy_t::scalar,
                            broadcasting_strategy_t::per_oc_spatial)
                    || rhs_arg_data_type != data_type::f32);
    const bool dt_helper_treg_needed
            = !binary_op_with_unaligned_mem_operand_allowed_
            || rhs_arg_data_type != data_type::f32 || bcast_f32_non_sve_512
            || should_preserve_treg_tail;

    // Phase 2 Protect temporary registers content.
    const injector_utils::register_preserve_guard_t<TReg> register_guard {host_,
            (rhs_arg_static_params_.preserve_gpr_helpers
                            ? std::initializer_list<Xbyak_aarch64::XReg>(
                                    {rhs_arg_static_params_.rhs_addr_reg,
                                            rhs_arg_static_params_
                                                    .rhs_helper_reg})
                            : std::initializer_list<Xbyak_aarch64::XReg>()),
            (rhs_arg_static_params_.preserve_treg_helper
                                    && dt_helper_treg_needed
                            ? std::initializer_list<TReg>({TReg(treg_hint)})
                            : std::initializer_list<TReg>())};

    bool treg0_was_preserved = false;
    static const TReg zero_treg(0);

    Xbyak_aarch64::AdrNoOfs rhs_arg_addr(Xbyak_aarch64::XReg(0));

    // Phase 3 Apply binary post-op over all tregs.
    for (const auto treg_idx : treg_idxs) {
        if (treg_idx == start_idx
                || rhs_arg_params_differ(treg_idx, treg_idx - 1, rhs_arg_params,
                        rhs_broadcasting_strategy)) {
            rhs_arg_addr = prepare_rhs_arg_addr(treg_idx, rhs_arg_idx, post_op,
                    rhs_arg_params, rhs_broadcasting_strategy);
        }

        const auto local_treg_preservation = should_preserve_treg(
                treg_idx, treg_hint, max_treg_idx, dt_helper_treg_needed);
        const bool &treg_preservation_needed = local_treg_preservation.first;
        const TReg dst_treg(treg_idx);
        const bool with_tail = rhs_arg_static_params_.tail_size
                && treg_tail_idx.find(treg_idx) != treg_tail_idx.cend()
                && IMPLICATION(rhs_broadcasting_strategy
                                == broadcasting_strategy_t::scalar,
                        rhs_arg_static_params_.use_exact_tail_scalar_bcast);

        if (treg_preservation_needed) {
            const TReg treg_to_preserve(local_treg_preservation.second);
            push_treg(host_, treg_to_preserve);
            inject_binary(post_op, dst_treg, rhs_arg_addr, with_tail);
            pop_treg(host_, treg_to_preserve);
            // in case all TReg are occupied, TReg(0) is chosen for tmp by default,
            // so it's content needs to be preserved...

            push_treg(host_, zero_treg);
            treg0_was_preserved = true;
        } else
            inject_binary(post_op, dst_treg, rhs_arg_addr, with_tail);
    }
    // ...and restored afterwards
    if (treg0_was_preserved) pop_treg(host_, zero_treg);
}

template <cpu_isa_t isa>
Xbyak_aarch64::AdrNoOfs jit_uni_binary_injector_t<isa>::prepare_rhs_arg_addr(
        std::size_t treg_idx, std::size_t rhs_arg_idx,
        const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params,
        const broadcasting_strategy_t rhs_broadcasting_strategy) const {

    static constexpr auto rhs_arg_ptr_size = sizeof(const void *);
    const auto &rhs_addr_reg = rhs_arg_static_params_.rhs_addr_reg;
    const auto &abi_param_offset = rhs_arg_static_params_.abi_param_offset;
    const auto &rhs_helper_reg = rhs_arg_static_params_.rhs_helper_reg;
    const auto rhs_arg_elem_size
            = types::data_type_size(post_op.binary.src1_desc.data_type);

    host_->add_imm(
            host_->X_DEFAULT_ADDR, param1_, abi_param_offset, host_->X_TMP_0);
    host_->ldr(rhs_addr_reg, Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));
    host_->add_imm(host_->X_DEFAULT_ADDR, rhs_addr_reg,
            rhs_arg_idx * rhs_arg_ptr_size, host_->X_TMP_0);
    host_->ldr(rhs_addr_reg, Xbyak_aarch64::ptr(host_->X_DEFAULT_ADDR));

    switch (rhs_broadcasting_strategy) {
            //        case broadcasting_strategy_t::scalar: return host_->ptr_b[rhs_addr_reg];
        case broadcasting_strategy_t::no_broadcast: {
            append_offset_from_operand(rhs_arg_params.treg_idx_to_out_off_oprnd,
                    treg_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.treg_idx_to_out_elem_off_addr, treg_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.treg_idx_to_out_elem_off_val,
                    treg_idx, rhs_addr_reg, rhs_arg_elem_size);

            return Xbyak_aarch64::ptr(rhs_addr_reg);
        }
            /*
        case broadcasting_strategy_t::per_oc:
        case broadcasting_strategy_t::per_oc_spatial: {
            append_offset_from_operand(rhs_arg_params.treg_idx_to_oc_off_oprnd,
                    treg_idx, rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_offset_under_mem_addr(
                    rhs_arg_params.treg_idx_to_oc_elem_off_addr, treg_idx,
                    rhs_addr_reg, rhs_helper_reg, rhs_arg_elem_size);
            append_value_offset(rhs_arg_params.treg_idx_to_oc_elem_off_val,
                    treg_idx, rhs_addr_reg, rhs_arg_elem_size);

            return rhs_broadcasting_strategy
                            == broadcasting_strategy_t::per_oc_spatial
                    ? host_->ptr_b[rhs_addr_reg]
                    : host_->ptr[rhs_addr_reg];
        } */
        default: assert(false && "Broadcasting type not supported");
    }

    return Xbyak_aarch64::ptr(rhs_addr_reg);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_from_operand(
        const std::map<int, Xbyak_aarch64::XReg> &treg_idx_to_elem_operand_off,
        int treg_idx, const Xbyak_aarch64::XReg &addr_reg,
        const Xbyak_aarch64::XReg &tmp_reg, std::size_t elem_size_bytes) const {

    const auto it_operand_off = treg_idx_to_elem_operand_off.find(treg_idx);
    if (it_operand_off != treg_idx_to_elem_operand_off.end()) {
        if (elem_size_bytes == 1) {
            //            host_->add(addr_reg, it_operand_off->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->mov(tmp_reg,
                    Xbyak_aarch64::XReg(it_operand_off->second.getIdx()));
            host_->lsl(tmp_reg, tmp_reg, shift_val);
            host_->add(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_offset_under_mem_addr(
        const std::map<int, Xbyak_aarch64::AdrNoOfs> &treg_idx_to_elem_addr_off,
        int treg_idx, const Xbyak_aarch64::XReg &addr_reg,
        const Xbyak_aarch64::XReg &tmp_reg, std::size_t elem_size_bytes) const {

    const auto it_off_addr = treg_idx_to_elem_addr_off.find(treg_idx);
    if (it_off_addr != treg_idx_to_elem_addr_off.end()) {
        if (elem_size_bytes == 1) {
            //	  host_->add(addr_reg, addr_reg, it_off_addr->second);
        } else {
            const int shift_val = std::log2(elem_size_bytes);
            host_->ldr(
                    tmp_reg, Xbyak_aarch64::ptr(it_off_addr->second.getXn()));
            host_->lsl(tmp_reg, tmp_reg, shift_val);
            host_->add(addr_reg, addr_reg, tmp_reg);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::append_value_offset(
        const std::map<int, int> &treg_idx_to_elem_val_off, int treg_idx,
        const Xbyak_aarch64::XReg &addr_reg,
        std::size_t elem_size_bytes) const {

    const auto it_off_val = treg_idx_to_elem_val_off.find(treg_idx);
    if (it_off_val != treg_idx_to_elem_val_off.end()) {
        host_->add_imm(addr_reg, addr_reg, it_off_val->second * elem_size_bytes,
                host_->X_TMP_0);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::inject_binary(
        const dnnl_post_ops::entry_t &post_op, TReg dst,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr, bool with_tail) const {

    const auto &alg = post_op.binary.alg;
    const auto &rhs_arg_data_type = post_op.binary.src1_desc.data_type;
    /*    const bool scalar_f32
	  = rhs_addr.isBroadcast() && rhs_arg_data_type == data_type::f32; */
    const bool scalar_f32 = false;
    const bool with_tail_not_fusable_to_binary_op
            = with_tail && !(scalar_f32 && is_sve_512_);
    const bool process_rhs_arg_using_tmp_treg
            = rhs_arg_data_type != data_type::f32
            || (scalar_f32 && !is_sve_512_)
            || with_tail_not_fusable_to_binary_op
            || !binary_op_with_unaligned_mem_operand_allowed_;

    if (process_rhs_arg_using_tmp_treg) {

        const TReg tmp_treg
                = TReg(rhs_arg_static_params_.rhs_dt_helper_treg_idx);

        /*        if (rhs_addr.isBroadcast())
            execute_broadcast(rhs_arg_data_type, tmp_treg,
                    remove_bcast_bit(rhs_addr), with_tail);
		    else */
        load_rhs(rhs_arg_data_type, tmp_treg, rhs_addr, with_tail);

        if (rhs_arg_data_type != data_type::bf16
                && rhs_arg_data_type != data_type::f32)
            cvt_to_f32(tmp_treg);

        execute_binary(alg, dst, dst, tmp_treg);
    } else {
        const auto lhs = dst;
        const bool with_tail_fusable_to_binary_op
                = with_tail && scalar_f32 && is_sve_512_;
        if (with_tail_fusable_to_binary_op) {
            assert(rhs_arg_static_params_.is_opmask_set()
                    && "Opmask is not set for tail loading sve_512");
            const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
            //            dst = dst | tail_opmask | host_->T_z;
        }

        execute_binary(alg, dst, lhs, rhs_addr);
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast(
        const dnnl_data_type_t &data_type, const TReg &tmp_reg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr, bool with_tail) const {
    if (with_tail)
        execute_broadcast_tail(data_type, tmp_reg, rhs_addr);
    else
        execute_broadcast_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs(const dnnl_data_type_t &data_type,
        const TReg &tmp_reg, const Xbyak_aarch64::AdrNoOfs &rhs_addr,
        bool with_tail) const {
    if (with_tail)
        load_rhs_tail(data_type, tmp_reg, rhs_addr);
    else
        load_rhs_no_tail(data_type, tmp_reg, rhs_addr);
}

template <cpu_isa_t isa>
Xbyak_aarch64::AdrNoOfs jit_uni_binary_injector_t<isa>::remove_bcast_bit(
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {
    //    return Xbyak_aarch64::AdrNoOfs(
    //            rhs_addr.getBit(), false, rhs_addr.getRegExp());
    return rhs_addr;
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::cvt_to_f32(const TReg &tmp_treg) const {
    //    host_->vcvtdq2ps(tmp_treg, tmp_treg);
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_no_tail(
        const dnnl_data_type_t &data_type, const TReg &tmp_treg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {
    /*    switch (data_type) {
        case data_type::f32: host_->uni_vbroadcastss(tmp_treg, rhs_addr); break;
        case data_type::s32: host_->uni_vpbroadcastd(tmp_treg, rhs_addr); break;
        case data_type::s8:
        case data_type::u8:
            execute_broadcast_s8u8_no_tail(data_type, tmp_treg, rhs_addr);
            break;
        case data_type::bf16:
            if (is_sve_512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpbroadcastw(tmp_treg, rhs_addr);
                host_->vpslld(tmp_treg, tmp_treg, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
	} */
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const TReg &tmp_treg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {
    /*    const TReg treg(tmp_treg.getIdx());
	switch (data_type) {
	case data_type::s8:
            host_->vpbroadcastb(xmm, rhs_addr);
            host_->vpmovsxbd(tmp_treg, xmm);
            break;
        case data_type::u8:
            host_->vpbroadcastb(xmm, rhs_addr);
            host_->vpmovzxbd(tmp_treg, xmm);
            break;
        default: assert(!"unsupported data type");
	}*/
}

template <>
void jit_uni_binary_injector_t<asimd>::execute_broadcast_s8u8_no_tail(
        const data_type_t &data_type, const TReg &tmp_treg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {
    /*
    if (data_type == data_type::s8 || data_type == data_type::u8) {
        const auto tmp_reg64_idx
                = rhs_arg_static_params_.rhs_helper_reg.getIdx();
        const Xbyak::Reg8 tmp_reg8 = Xbyak::Reg8(tmp_reg64_idx);
        const Xbyak::Reg32 tmp_reg32 = Xbyak::Reg32(tmp_reg64_idx);
        const auto tmp_xmm = Xbyak::Xmm(tmp_treg.getIdx());
        host_->mov(tmp_reg8, rhs_addr);
        host_->vmovd(tmp_xmm, tmp_reg32);
        host_->vpunpcklbw(tmp_xmm, tmp_xmm, tmp_xmm);
        host_->vpshuflw(tmp_xmm, tmp_xmm, 0);
        if (data_type == data_type::s8)
            host_->vpmovsxbd(tmp_xmm, tmp_xmm);
        else
            host_->vpmovzxbd(tmp_xmm, tmp_xmm);

        host_->vinsertf128(tmp_treg, tmp_treg, tmp_xmm, 1);
	} else */
    assert(!"unsupported data type");
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::execute_broadcast_tail(
        const dnnl_data_type_t &data_type, const TReg &tmp_treg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");
    /*    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;

    host_->uni_vxorps(tmp_treg, tmp_treg, tmp_treg);
    switch (data_type) {
        case data_type::f32:
            host_->vbroadcastss(tmp_treg | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s32:
            host_->vpbroadcastd(tmp_treg | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::s8:
            host_->vpbroadcastb(tmp_treg | tail_opmask | host_->T_z,
                    rhs_addr); // broadcast to tmm_xmm should be enough ?
            host_->vpmovsxbd(tmp_treg | tail_opmask | host_->T_z, tmp_treg);
            break;
        case data_type::u8:
            host_->vpbroadcastb(tmp_treg | tail_opmask | host_->T_z, rhs_addr);
            host_->vpmovzxbd(tmp_treg | tail_opmask | host_->T_z, tmp_treg);
            break;
        case data_type::bf16:
            if (is_avx512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpbroadcastw(tmp_treg, rhs_addr);
                host_->vpslld(
                        tmp_treg | tail_opmask | host_->T_z, tmp_treg, 0x10);
                break;
            }
        default: assert(!"unsupported data type");
	} */
    assert(!"kawakami");
}

static constexpr int xmm_size_elem = 4;

static void load_tail_avx(jit_generator *host, std::size_t ymm_idx,
        std::size_t tail_size, const std::function<void()> &init_op,
        const std::function<void(int, bool)> &ymm_upper_half_op,
        const std::function<void(int)> &ymm_lower_half_op) {
    /*
    if (init_op) init_op();

    const auto res = std::div(tail_size, xmm_size_elem);
    const auto &ymm_upper_half_op_data_size = res.rem;
    const bool should_load_lower_half = res.quot;

    if (ymm_upper_half_op_data_size && ymm_upper_half_op)
        ymm_upper_half_op(ymm_upper_half_op_data_size, should_load_lower_half);

    if (should_load_lower_half) {
        const auto tmp_xmm = Xbyak::Xmm(ymm_idx);

        if (ymm_upper_half_op_data_size) push_treg(host, tmp_xmm);

        if (ymm_lower_half_op) ymm_lower_half_op(ymm_upper_half_op_data_size);

        if (ymm_upper_half_op_data_size) {
            const auto tmp_ymm = Xbyak::Ymm(ymm_idx);
            host->vinsertf128(tmp_ymm, tmp_ymm, host->ptr[host->rsp], 1);
            restore_stack(host, tmp_xmm);
        }
	} */
}

static void load_tail_avx(jit_generator *host, std::size_t ymm_idx,
        std::size_t tail_size,
        const std::function<void(int, bool)> &ymm_upper_half_op,
        const std::function<void(int)> &ymm_lower_half_op) {
    load_tail_avx(host, ymm_idx, tail_size, nullptr, ymm_upper_half_op,
            ymm_lower_half_op);
}

static uint8_t MM_SHUFFLE(uint8_t z, uint8_t y, uint8_t x, uint8_t w) {
    return (((z) << 6) | ((y) << 4) | ((x) << 2) | (w));
}

static void execute_broadcast_f32_tail_avx(jit_generator *host,
        std::size_t ymm_idx, const Xbyak_aarch64::AdrNoOfs &rhs_addr,
        std::size_t tail_size) {

    /*    const auto tmp_xmm = Xbyak::Xmm(ymm_idx);
    static const std::array<uint8_t, 2> imms {
            {MM_SHUFFLE(3, 2, 0, 0), MM_SHUFFLE(3, 0, 0, 0)}};

    const auto init_op = [&] { host->vmovss(tmp_xmm, rhs_addr); };
    const auto upper_half_op
            = [&](int upper_half_data_size, bool should_load_lower_half) {
                  // one element is already loaded
                  if (upper_half_data_size > 1)
                      host->vshufps(tmp_xmm, tmp_xmm, tmp_xmm,
                              imms.at(upper_half_data_size - 2));
              };
    const auto lower_half_op = [&](int upper_half_data_size) {
        host->vshufps(tmp_xmm, tmp_xmm, tmp_xmm, 0);
    };

    load_tail_avx(
    host, ymm_idx, tail_size, init_op, upper_half_op, lower_half_op); */
}
template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_no_tail(
        const dnnl_data_type_t &data_type, const TReg &tmp_treg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            host_->ldr(tmp_treg, rhs_addr);
            break;
            /*        case data_type::s8: host_->uni_vpmovsxbd(tmp_treg, rhs_addr); break;
        case data_type::u8: host_->uni_vpmovzxbd(tmp_treg, rhs_addr); break;
        case data_type::bf16:
            if (is_avx512_
                    && utils::one_of(isa, avx512_core_bf16, avx512_core)) {
                host_->vpmovzxwd(tmp_treg, rhs_addr);
                host_->vpslld(tmp_treg, tmp_treg, 0x10);
                break;
		} */
        default: assert(!"unsupported data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_binary_injector_t<isa>::load_rhs_tail(
        const dnnl_data_type_t &data_type, const TReg &tmp_treg,
        const Xbyak_aarch64::AdrNoOfs &rhs_addr) const {

    assert(rhs_arg_static_params_.is_opmask_set()
            && "Opmask is not set for tail loading avx512");

    const auto &tail_opmask = rhs_arg_static_params_.tail_opmask;
    int vlen = cpu_isa_traits<isa>::vlen;
    if (vlen == 64) {
        host_->eor(Xbyak_aarch64::ZReg(IDX(tmp_treg)).d,
                Xbyak_aarch64::ZReg(IDX(tmp_treg)).d,
                Xbyak_aarch64::ZReg(IDX(tmp_treg)).d);
    } else if (vlen == 32) {
        host_->ptrue(Xbyak_aarch64::PRegB(1), Xbyak_aarch64::VL32);
        host_->eor(Xbyak_aarch64::ZReg(IDX(tmp_treg)).d,
                Xbyak_aarch64::ZReg(IDX(tmp_treg)).d,
                Xbyak_aarch64::ZReg(IDX(tmp_treg)).d);
        host_->mov(Xbyak_aarch64::ZReg(IDX(tmp_treg)).s,
                Xbyak_aarch64::PReg(1) / Xbyak_aarch64::T_m, 0);
    } else if (vlen == 16) {
        host_->eor(Xbyak_aarch64::VReg16B(IDX(tmp_treg)),
                Xbyak_aarch64::VReg16B(IDX(tmp_treg)),
                Xbyak_aarch64::VReg16B(IDX(tmp_treg)));
    } else {
        assert(!"unreachable");
    }
    switch (data_type) {
        case data_type::f32:
        case data_type::s32:
            if (vlen == 64) {
                host_->pfalse(Xbyak_aarch64::PRegB(9));
                host_->zip1(Xbyak_aarch64::PRegB(1), tail_opmask.b,
                        Xbyak_aarch64::PRegB(9));
                host_->zip1(Xbyak_aarch64::PRegH(1), Xbyak_aarch64::PRegH(1),
                        Xbyak_aarch64::PRegH(9));
                host_->ld1w(Xbyak_aarch64::ZRegS(IDX(tmp_treg)),
                        Xbyak_aarch64::PReg(1) / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(rhs_addr.getXn()));
            } else if (vlen == 32) {
                host_->sub(Xbyak_aarch64::XReg(22), Xbyak_aarch64::XReg(22), 8);
                host_->str(Xbyak_aarch64::PReg(7),
                        Xbyak_aarch64::ptr(Xbyak_aarch64::XReg(22)));
                host_->ptrue(Xbyak_aarch64::PRegB(7));
                host_->ptrue(Xbyak_aarch64::PRegB(14), Xbyak_aarch64::VL32);
                host_->bic(Xbyak_aarch64::PRegB(7),
                        Xbyak_aarch64::PReg(7) / Xbyak_aarch64::T_z,
                        tail_opmask.b, Xbyak_aarch64::PRegB(14));
                host_->ld1w(Xbyak_aarch64::ZRegS(IDX(tmp_treg)),
                        Xbyak_aarch64::PReg(7) / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(rhs_addr.getXn()));
                host_->ldr(Xbyak_aarch64::PReg(7),
                        Xbyak_aarch64::ptr(Xbyak_aarch64::XReg(22)));
                host_->add(Xbyak_aarch64::XReg(22), Xbyak_aarch64::XReg(22), 8);
            } else if (vlen == 16) {
                host_->sub(Xbyak_aarch64::XReg(22), Xbyak_aarch64::XReg(22), 8);
                host_->str(Xbyak_aarch64::PReg(7),
                        Xbyak_aarch64::ptr(Xbyak_aarch64::XReg(22)));
                host_->ptrue(Xbyak_aarch64::PRegB(7));
                host_->ptrue(Xbyak_aarch64::PRegB(13), Xbyak_aarch64::VL16);
                host_->bic(Xbyak_aarch64::PRegB(7),
                        Xbyak_aarch64::PReg(7) / Xbyak_aarch64::T_z,
                        tail_opmask.b, Xbyak_aarch64::PRegB(13));
                host_->ld1w(Xbyak_aarch64::ZRegS(IDX(tmp_treg)),
                        Xbyak_aarch64::PReg(7) / Xbyak_aarch64::T_z,
                        Xbyak_aarch64::ptr(rhs_addr.getXn()));
                host_->ldr(Xbyak_aarch64::PReg(7),
                        Xbyak_aarch64::ptr(Xbyak_aarch64::XReg(22)));
                host_->add(Xbyak_aarch64::XReg(22), Xbyak_aarch64::XReg(22), 8);
            } else {
                assert(!"unreachable");
            }
            break;
        case data_type::s8:
            //            host_->vpmovsxbd(tmp_treg | tail_opmask | host_->T_z, rhs_addr);
            break;
        case data_type::u8:
            //            host_->vpmovzxbd(tmp_treg | tail_opmask | host_->T_z, rhs_addr);
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
            host_->fadd(Xbyak_aarch64::ZReg(IDX(dst)).s,
                    Xbyak_aarch64::ZReg(IDX(lhs)).s,
                    Xbyak_aarch64::ZReg(IDX(rhs)).s);
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
            host_->ptrue(Xbyak_aarch64::PReg(1).b);
            host_->sub_imm(Xbyak_aarch64::XReg(22), Xbyak_aarch64::XReg(22), 64,
                    host_->X_TMP_0);
            host_->st1w(Xbyak_aarch64::ZRegS(1), Xbyak_aarch64::PReg(1),
                    Xbyak_aarch64::ptr(Xbyak_aarch64::XReg(22)));
            host_->ldr(Xbyak_aarch64::ZReg(1), Xbyak_aarch64::ptr(rhs.getXn()));
            host_->fadd(Xbyak_aarch64::ZReg(IDX(dst)).s,
                    Xbyak_aarch64::ZReg(IDX(lhs)).s, Xbyak_aarch64::ZReg(1).s);
            host_->ld1w(Xbyak_aarch64::ZRegS(1), Xbyak_aarch64::PReg(1),
                    Xbyak_aarch64::ptr(Xbyak_aarch64::XReg(22)));
            host_->add_imm(Xbyak_aarch64::XReg(22), Xbyak_aarch64::XReg(22), 64,
                    host_->X_TMP_0);
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
void jit_uni_binary_injector_t<isa>::compute_vector(size_t idx,
        std::size_t rhs_arg_idx, const dnnl_post_ops::entry_t &post_op,
        const rhs_arg_dynamic_params_t &rhs_arg_params) const {
    compute_vector_range({idx}, rhs_arg_idx, post_op, rhs_arg_params);
}

template class jit_uni_binary_injector_t<sve_512>;

} // namespace binary_injector
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
