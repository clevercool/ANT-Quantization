  
#include <torch/extension.h>
#include <iostream>
using namespace std;

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> quant_forward_cuda(
    torch::Tensor x,
    torch::Tensor y);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> quant_forward_cpu(torch::Tensor x,
                            torch::Tensor y) {
                              
    // CHECK_INPUT(x);
    // CHECK_INPUT(y);
    auto z = quant_forward_cuda(x, y);
    return z;
}

// pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quant", &quant_forward_cpu, "Quantization function");
}