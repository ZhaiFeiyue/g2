
// TODO: import exposed file
#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <synapse_common_types.hpp>

enum DTYPE{
  FP8FP8 = 0,
  FP8BF16 = 1,
  FP8FP32 = 2,
  BFLOAT16 = 3,
  FLOAT = 4,
  FP8BF16SCALE = 5
};

bool register_custom_matmul(DTYPE type) {
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::TENSOR, 3};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc};
    std::vector<habana::custom_op::InputDesc> inputs_4_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

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

    if (type == DTYPE::BFLOAT16) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::BFloat16, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_bf16", //schema name
          "custom_matrix_multiply_fwd_bf16_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_bf16\n";
    }

    if (type == DTYPE::FLOAT) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::Float, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_fp32", //schema name
          "custom_matrix_multiply_fwd_fp32_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_fp32\n";
    }    

    if (type == DTYPE::FP8FP8) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::Float8_e4m3fn, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_fp8fp8", //schema name
          "custom_matrix_multiply_fwd_fp8_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_fp8fp8\n";
    }
    
    if (type == DTYPE::FP8BF16) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::BFloat16, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_fp8bf16", //schema name
          "custom_matrix_multiply_fwd_fp8_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_fp8bf16\n";
    }
    
    if (type == DTYPE::FP8BF16SCALE) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::BFloat16, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_fp8bf16scale", //schema name
          "custom_matrix_multiply_fwd_with_scale_fp8_gaudi2", // guid
          inputs_4_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_fp8bf16scale\n";
    }
    
    if (type == DTYPE::FP8FP32) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::Float, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_fp8fp32", //schema name
          "custom_matrix_multiply_fwd_fp8_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_fp8fp32\n";
    }
    
    return true;
}

bool register_custom_matmul_idx(DTYPE type) {
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc};

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

    if (type == DTYPE::BFLOAT16) {
	habana::custom_op::OutputDesc output_desc{0, c10::ScalarType::BFloat16, output_size_lambda};
	std::vector<habana::custom_op::OutputDesc> outputs_desc{output_desc};
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_idx_bf16", //schema name
          "custom_matrix_multiply_fwd_with_index_bf16_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_idx_bf16\n";
    }
    return true;
}

bool register_custom_matmul_t(DTYPE type) {
  //inputs desc
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
    result_sizes[self_rank - 1] = 1;
    result_sizes[self_rank - 2] = b.sizes().vec()[b_rank - 3];
    result_sizes[self_rank - 3] = self.sizes().vec()[self_rank - 2];
    return result_sizes;
  };
  habana::custom_op::OutputDesc output_desc{
  0, c10::ScalarType::BFloat16, output_size_lambda};
  
  std::vector<habana::custom_op::OutputDesc> outputs_desc{
  output_desc};
  
  if (type == DTYPE::BFLOAT16) {
    REGISTER_CUSTOM_OP_ATTRIBUTES(
    "custom_op::custom_matmul_t_bf16", //schema name
    "custom_matrix_multiply_t_fwd_bf16_gaudi2", // guid
    inputs_desc,
    outputs_desc,
    nullptr);
    std::cout << "cpp registered custom_op::custom_matmul_t_bf16\n";
  }

  return true;
}

bool register_custom_matmul_sv(DTYPE type) {
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
        0, c10::ScalarType::BFloat16, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    if (type == DTYPE::BFLOAT16) {
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_matmul_sv_bf16", //schema name
          "custom_matrix_multiply_sv_fwd_bf16_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_matmul_sv_bf16\n";
    }  

    return true;
}

bool register_custom_gemm(DTYPE type) {
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
        0, c10::ScalarType::BFloat16, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    if (type == DTYPE::BFLOAT16) {
      REGISTER_CUSTOM_OP_ATTRIBUTES(
          "custom_op::custom_gemm_bf16", //schema name
          "custom_gemm_fwd_bf16_gaudi2", // guid
          inputs_desc,
          outputs_desc,
          nullptr);
      std::cout << "cpp registered custom_op::custom_gemm_bf16" << std::endl;
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

at::Tensor custom_matmul_execute_fp8fp8(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul(DTYPE::FP8FP8);
  TORCH_CHECK(registered, "custom_matmul_fp8 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_fp8fp8");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_execute_fp8bf16(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul(DTYPE::FP8BF16);
  TORCH_CHECK(registered, "custom_matmul_fp8 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_fp8bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_execute_fp8bf16scale(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor input_c,
    torch::Tensor input_d) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul(DTYPE::FP8BF16SCALE);
  TORCH_CHECK(registered, "custom_matmul_fp8bf16scale kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b, input_c, input_d};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_fp8bf16scale");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_execute_fp8fp32(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul(DTYPE::FP8FP32);
  TORCH_CHECK(registered, "custom_matmul_fp8 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_fp8fp32");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_idx_execute_bf16(
    torch::Tensor input_a,
    torch::Tensor input_b,
    torch::Tensor input_c) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul_idx(DTYPE::BFLOAT16);
  TORCH_CHECK(registered, "custom_matmul_idx_bf16 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b, input_c};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_idx_bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_t_execute_bf16(
		    torch::Tensor input_a,
		        torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul_t(DTYPE::BFLOAT16);
  TORCH_CHECK(registered, "custom_matmul_t_bf16 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_t_bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_matmul_sv_execute_bf16(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_matmul_sv(DTYPE::BFLOAT16);
  TORCH_CHECK(registered, "custom_matmul_sv_bf16 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_matmul_sv_bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor custom_gemm_execute_bf16(
    torch::Tensor input_a,
    torch::Tensor input_b) {
  // Registering the custom op, need to be called only once
  static bool registered = register_custom_gemm(DTYPE::BFLOAT16);
  TORCH_CHECK(registered, "custom_gemm_bf16 kernel not registered" );
  std::vector<c10::IValue> inputs{input_a, input_b};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::custom_gemm_bf16");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("custom_matmul_bf16(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_fp32(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_fp8fp8(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_fp8bf16(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_fp8bf16scale(Tensor self, Tensor b, Tensor c, Tensor d) -> Tensor");
  m.def("custom_matmul_fp8fp32(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_idx_bf16(Tensor self, Tensor b, Tensor c) -> Tensor");
  m.def("custom_matmul_t_bf16(Tensor self, Tensor b) -> Tensor");
  m.def("custom_matmul_sv_bf16(Tensor self, Tensor b) -> Tensor");
  m.def("custom_gemm_bf16(Tensor self, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("custom_matmul_bf16", custom_matmul_execute_bf16);
  m.impl("custom_matmul_fp32", custom_matmul_execute_fp32);
  m.impl("custom_matmul_fp8fp8", custom_matmul_execute_fp8fp8);
  m.impl("custom_matmul_fp8bf16", custom_matmul_execute_fp8bf16);
  m.impl("custom_matmul_fp8bf16scale", custom_matmul_execute_fp8bf16scale);
  m.impl("custom_matmul_fp8fp32", custom_matmul_execute_fp8fp32);
  m.impl("custom_matmul_idx_bf16", custom_matmul_idx_execute_bf16);
  m.impl("custom_matmul_t_bf16", custom_matmul_t_execute_bf16);
  m.impl("custom_matmul_sv_bf16", custom_matmul_sv_execute_bf16);
  m.impl("custom_gemm_bf16", custom_gemm_execute_bf16);
}
