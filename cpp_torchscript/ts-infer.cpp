/*
#include <torch/script.h>
#include <torch/nn/functional/activation.h>


int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ts-infer <path-to-exported-model>\n";
        return -1;
    }

    std::cout << "Loading model...\n";

    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        std::cerr << e.msg_without_backtrace();
        return -1;
    }

    std::cout << "Model loaded successfully\n";

    torch::NoGradGuard no_grad; // ensures that autograd is off
    module.eval(); // turn off dropout and other training-time layers/functions

    // create an input "image"
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}));

    // execute model and package output as tensor
    at::Tensor output = module.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
    at::Tensor top5 = std::get<1>(top5_tensor);

    std::cout << top5[0] << "\n";

    std::cout << "\nDONE\n";
    return 0;
} */
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";


// Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
//inputs.push_back(torch::tensor({101, 1996, 4248, 2829,4419,8275,7632,1996,13971,3899,102}));
inputs.push_back(torch::ones({1,11},at::kInt));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/10) << '\n';
}
