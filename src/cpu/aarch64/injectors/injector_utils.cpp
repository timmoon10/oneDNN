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
#include <numeric>
#include "common/broadcast_strategy.hpp"
#include "cpu/aarch64/injectors/injector_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector_utils {

template <typename TReg>
static std::size_t get_treg_size_bytes(const TReg &treg) {
    static constexpr int byte_size_bits = 8;
    return treg.getBit() / byte_size_bits;
}

template <typename TReg>
static std::size_t calc_treg_to_preserve_size_bytes(
        const std::initializer_list<TReg> &treg_to_preserve) {

    return std::accumulate(treg_to_preserve.begin(), treg_to_preserve.end(),
            std::size_t(0u), [](std::size_t accum, const TReg &treg) {
                return accum + get_treg_size_bytes(treg);
            });
}

template <typename TReg>
register_preserve_guard_t<TReg>::register_preserve_guard_t(jit_generator *host,
        std::initializer_list<Xbyak_aarch64::XReg> xreg_to_preserve,
        std::initializer_list<TReg> treg_to_preserve)
    : host_(host)
    , xreg_stack_(xreg_to_preserve)
    , treg_stack_(treg_to_preserve)
    , treg_to_preserve_size_bytes_(
              calc_treg_to_preserve_size_bytes(treg_to_preserve)) {

    for (const auto &reg : xreg_to_preserve)
        host_->str(reg, Xbyak_aarch64::pre_ptr(host_->X_TRANSLATOR_STACK, -8));

    if (!treg_stack_.empty()) {
        host_->sub_imm(host_->X_SP, host_->X_SP, treg_to_preserve_size_bytes_,
                host_->X_TMP_0);

        auto stack_offset = treg_to_preserve_size_bytes_;
        for (const auto &treg : treg_to_preserve) {
            stack_offset -= get_treg_size_bytes(treg);
            const auto idx = treg.getIdx();

            if (treg.getBit() != 16) { /* TReg is ZReg */
                host_->str(Xbyak_aarch64::ZReg(idx),
                        Xbyak_aarch64::ptr(host_->X_SP, (int32_t)stack_offset));
            } else /* TReg is VReg */
                host_->str(Xbyak_aarch64::QReg(idx),
                        Xbyak_aarch64::ptr(host_->X_SP, (int32_t)stack_offset));
        }
    }
}

template <typename TReg>
register_preserve_guard_t<TReg>::~register_preserve_guard_t() {

    auto tmp_stack_offset = 0;

    while (!treg_stack_.empty()) {
        const TReg &treg = treg_stack_.top();
        const auto idx = treg.getIdx();
        if (treg.getBit() != 16) /* TReg is ZReg. */
            host_->ldr(Xbyak_aarch64::ZReg(idx),
                    Xbyak_aarch64::ptr(host_->X_SP, (int32_t)tmp_stack_offset));
        else /* TReg is VReg. */
            host_->ldr(Xbyak_aarch64::QReg(idx),
                    Xbyak_aarch64::ptr(host_->X_SP, (int32_t)tmp_stack_offset));

        tmp_stack_offset += get_treg_size_bytes(treg);
        treg_stack_.pop();
    }

    if (treg_to_preserve_size_bytes_)
        host_->add_imm(host_->X_SP, host_->X_SP, treg_to_preserve_size_bytes_,
                host_->X_TMP_0);

    while (!xreg_stack_.empty()) {
        host_->ldr(xreg_stack_.top(),
                Xbyak_aarch64::post_ptr(host_->X_TRANSLATOR_STACK, 8));
        xreg_stack_.pop();
    }
}

template <typename TReg>
size_t register_preserve_guard_t<TReg>::stack_space_occupied() const {
    constexpr static size_t xreg_size = 8;
    const size_t stack_space_occupied
            = treg_to_preserve_size_bytes_ + xreg_stack_.size() * xreg_size;

    return stack_space_occupied;
};

template <typename TReg>
conditional_register_preserve_guard_t<
        TReg>::conditional_register_preserve_guard_t(bool condition_to_be_met,
        jit_generator *host,
        std::initializer_list<Xbyak_aarch64::XReg> xreg_to_preserve,
        std::initializer_list<TReg> treg_to_preserve)
    : register_preserve_guard_t<TReg> {condition_to_be_met
                    ? register_preserve_guard_t<TReg> {host, xreg_to_preserve,
                            treg_to_preserve}
                    : register_preserve_guard_t<TReg> {nullptr, {}, {}}} {};

template class register_preserve_guard_t<Xbyak_aarch64::ZReg>;
template class conditional_register_preserve_guard_t<Xbyak_aarch64::ZReg>;

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
