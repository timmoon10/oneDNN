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
#include <numeric>
#include "cpu/aarch64/injectors/injector_utils.hpp"

#ifdef DNNL_AARCH64
#define IDX(a) static_cast<uint32_t>(a.getIdx())
#endif //#ifdef DNNL_AARCH64

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace injector_utils {

namespace xa = Xbyak_aarch64;

static std::size_t get_vmm_size_bytes(const Xbyak_aarch64::VReg &vmm) {
    static constexpr int byte_size_bits = 8;
    if (mayiuse(sve_512))
        return Xbyak_aarch64::ZReg(vmm.getIdx()).getBit() / byte_size_bits;
    else
        return vmm.getBit() / byte_size_bits;
}

static std::size_t calc_vmm_to_preserve_size_bytes(
        const std::initializer_list<Xbyak_aarch64::VReg> &vmm_to_preserve) {
    return std::accumulate(vmm_to_preserve.begin(), vmm_to_preserve.end(),
            std::size_t(0u),
            [](std::size_t accum, const Xbyak_aarch64::VReg &vmm) {
                return accum + get_vmm_size_bytes(vmm);
            });
}

register_preserve_guard_t::register_preserve_guard_t(jit_generator *host,
        std::initializer_list<Xbyak_aarch64::XReg> reg64_to_preserve,
        std::initializer_list<Xbyak_aarch64::VReg> vmm_to_preserve)
    : host_(host)
    , reg64_stack_(reg64_to_preserve)
    , vmm_stack_(vmm_to_preserve)
    , vmm_to_preserve_size_bytes_(
              calc_vmm_to_preserve_size_bytes(vmm_to_preserve)) {

    for (const auto &reg : reg64_to_preserve)
        host_->str(xa::XReg(reg), xa::pre_ptr(host_->X_TRANSLATOR_STACK, -8));

    if (!vmm_stack_.empty()) {
        host_->sub_imm(host_->X_SP, host_->X_SP, vmm_to_preserve_size_bytes_,
                host_->X_TMP_0);

        auto stack_offset = vmm_to_preserve_size_bytes_;
        for (const auto &vmm : vmm_to_preserve) {
            const auto idx = vmm.getIdx();
            stack_offset -= get_vmm_size_bytes(vmm);

            if (mayiuse(sve_512)) {
                host_->str(xa::ZReg(idx),
                        xa::ptr(host_->X_SP, (int32_t)stack_offset));
            } else {
                host_->str(xa::QReg(idx),
                        xa::ptr(host_->X_SP, (int32_t)stack_offset));
            }
        }
    }
}

register_preserve_guard_t::~register_preserve_guard_t() {

    auto tmp_stack_offset = 0;
    int i = 0;

    while (!vmm_stack_.empty()) {
        const xa::VReg &vmm = vmm_stack_.top();
        const auto idx = vmm.getIdx();
        if (mayiuse(sve_512))
            host_->ldr(xa::ZReg(idx),
                    xa::ptr(host_->X_SP, (int32_t)tmp_stack_offset));
        else
            host_->ldr(xa::QReg(idx),
                    xa::ptr(host_->X_SP, (int32_t)tmp_stack_offset));

        tmp_stack_offset += get_vmm_size_bytes(vmm);
        vmm_stack_.pop();
    }

    if (vmm_to_preserve_size_bytes_) {
        host_->add_imm(host_->X_SP, host_->X_SP, vmm_to_preserve_size_bytes_,
                host_->X_TMP_0);
    }

    while (!reg64_stack_.empty()) {
        host_->ldr(xa::XReg(IDX(reg64_stack_.top())),
                xa::post_ptr(host_->X_TRANSLATOR_STACK, 8));
        reg64_stack_.pop();
    }
}

size_t register_preserve_guard_t::stack_space_occupied() const {
    constexpr static size_t reg64_size = 8;
    const size_t stack_space_occupied
            = vmm_to_preserve_size_bytes_ + reg64_stack_.size() * reg64_size;

    return stack_space_occupied;
};

output_dims_t make_output_dims(const memory_desc_wrapper &dst_d) {

    const dim_t n_dims = dst_d.ndims();
    const auto dims = dst_d.dims();
    const dim_t &mb = dims[0];
    const dim_t &oc = n_dims >= 2 ? dims[1] : 1;
    const dim_t &ow = n_dims >= 3 ? dims[n_dims - 1] : 1;
    const dim_t &oh = n_dims >= 4 ? dims[n_dims - 2] : 1;
    const dim_t &od = n_dims >= 5 ? dims[n_dims - 3] : 1;

    switch (n_dims) {
        case 1: return output_dims_t {{mb, 0, 0, 0, 0}};
        case 2: return output_dims_t {{mb, 0, 0, 0, 0}};
        case 3: return output_dims_t {{mb, oc, ow, 0, 0}};
        case 4: return output_dims_t {{mb, oc, oh, ow, 0}};
        case 5: return output_dims_t {{mb, oc, od, oh, ow}};
        default: assert(!"dimension count error"); break;
    }

    return output_dims_t();
}

} // namespace injector_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
