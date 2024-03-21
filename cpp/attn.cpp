
// TODO: import exposed file
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <synapse_common_types.hpp>

enum DTYPE{
  FP8 = 0,
  BFLOAT16 = 1,
  FLOAT = 2
};

bool register_custom_matmul(DTYPE type) {
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      auto self_rank = self.sizes().size();

      auto b = inputs[1].toTensor();
      auto b_rank = b.sizes().size();

      std::vector<int64_t> result_sizes = self.sizes().vec();
      result_sizes[self_rank - 1] = b.sizes().vec()[b_rank - 1];
      return result_sizes;
    };
    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    if (type == DTYPE::BFLOAT16) {
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_bf16", //schema name
          "custom_matrix_multiply_fwd_bf16_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_bf16\n";
    }

    if (type == DTYPE::FLOAT) {
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_fp32", //schema name
          "custom_matrix_multiply_fwd_fp32_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_fp32\n";
    }    

    return true;
}

at::Tensor custom_matmul_execute_bf16(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul(DTYPE::BFLOAT16);
  TORCH_CHECK(registered, "custom_matmul_bf16 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_execute_fp32(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul(DTYPE::FLOAT);
  TORCH_CHECK(registered, "custom_matmul_fp32 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_fp32");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_matmul_bf16(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_fp32(Tensor self, Tensor b) -> Tensor");

}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_matmul_bf16", custom_matmul_execute_bf16);
  m.impl("custom_matmul_fp32", custom_matmul_execute_fp32);
}
