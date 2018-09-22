// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/compute/kernels/match.h"

#include "arrow/compute/context.h"
#include "arrow/compute/kernels/util-internal.h"
#include "arrow/util/logging.h"

#include <vector>
#include <map>
#include <arrow/io/api.h>
#include <sstream>

namespace arrow {
    namespace compute {

        class MatchKernel : public BinaryKernel {
            Status Call(FunctionContext* ctx, const Datum& left, const Datum& right,
                        Datum* out) override {
                DCHECK_EQ(Datum::ARRAY, right.kind());
                DCHECK_EQ(Datum::ARRAY, left.kind());

                const ArrayData& left_data = *left.array();
                const ArrayData& right_data = *right.array();
                ArrayData* result;
                out->value = ArrayData::Make(int64(), left_data.length);

                result = out->array().get();

                return Compute(ctx, left_data, right_data, result);
            }

            Status Compute(FunctionContext* ctx, const ArrayData& left,
                           const ArrayData& right, ArrayData* out) {
                DCHECK_EQ(left.type, right.type);

                constexpr uint8_t empty_value = 0;
                const uint8_t *left_data;
                const uint8_t *right_data;
                const int32_t* left_offsets = GetValues<int32_t>(left, 1);
                const int32_t* right_offsets = GetValues<int32_t>(right, 1);

                if (left.buffers[2].get() == nullptr) {
                    left_data = &empty_value;
                } else {
                    left_data = GetValues<uint8_t>(left, 2);
                }

                if (right.buffers[2].get() == nullptr) {
                    right_data = &empty_value;
                } else {
                    right_data = GetValues<uint8_t>(right, 2);
                }

                out->null_count = right.null_count;
                
                for (int64_t i = 0; i < left.length; i++) {
                    const int32_t left_current_position = left_offsets[i];                                              \
                    const int32_t left_current_length = left_offsets[i + 1] - left_current_position;                                 \
                    const uint8_t* left_current_value = left_data + left_current_position;
                    std::shared_ptr<Buffer> out_data = nullptr;

                    for (int64_t j = 0; j < right.length; j++) {
                        const int32_t right_current_position = right_offsets[j];                                              \
                        const int32_t right_current_length = right_offsets[j + 1] - right_current_position;                                 \
                        const uint8_t* right_current_value = right_data + right_current_position;

                        if (left_current_length == right_current_length &&
                        (std::memcmp(left_current_value, right_current_value, left_current_length)) == 0) {
                            if (out_data != nullptr) {
                                std::stringstream ss;
                                ss << "Invalid State: Data should not have been pre-allocated " << out->type->ToString();
                                return Status::Invalid(ss.str());
                            }

                            RETURN_NOT_OK(ctx->Allocate(sizeof(int64_t), &out_data));
                            memset(out_data->mutable_data(), 0, sizeof(int64_t));
                            auto current_buffer_data = reinterpret_cast<int32_t*>(out_data->mutable_data());

                            current_buffer_data[0] = j;
                            break;
                        }
                    }

                    out->buffers.push_back(out_data);
                }

                return Status::OK();
            }
        };

        Status MatchArrays(FunctionContext* context, const Datum& left, const Datum& right, Datum* out) {
            MatchKernel kernel;

            return detail::InvokeBinaryArrayKernel(context, &kernel, left, right, out);
        }
    }  // namespace compute
}  // namespace arrow
