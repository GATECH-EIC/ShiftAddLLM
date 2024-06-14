#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <vector>

void lutgemm_cuda(torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor q_bias,
  torch::Tensor d_input, int mSize, int kSize, int nb,  int num_groups);

void lutgemm_block_cuda(torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input, int mSize, int kSize, int nb,  int num_groups);

void lutgemm_block_shift_cuda(torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input, int mSize, int kSize, int nb,  int num_groups, int num_apot);

void lutgemm_block_shiftInt8_cuda(torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input, int mSize, int kSize, int nb,  int num_groups, int num_apot);

void lutgemm_compute_shift_scale_cuda(torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input, int mSize, int kSize, int nb,  int num_groups);

void lutgemm_compute_shift_scale(
  torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  lutgemm_compute_shift_scale_cuda(output, bWeight, alpha, d_input, mSize,  kSize,  nb,   num_groups);
}

void lutgemm_compute_block(
  torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  lutgemm_block_cuda(output, bWeight, alpha, d_input, mSize,  kSize,  nb,   num_groups);
}

void lutgemm_compute_block_shift(
  torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups, int num_apot
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  lutgemm_block_shift_cuda(output, bWeight, alpha, d_input, mSize,  kSize,  nb,   num_groups, num_apot);
}

void lutgemm_compute_block_shiftInt8(
  torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups, int num_apot
) {
  //const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  lutgemm_block_shiftInt8_cuda(output, bWeight, alpha, d_input, mSize,  kSize,  nb,   num_groups, num_apot);
}


void lutgemm_compute(
  torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor q_bias,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  lutgemm_cuda(output, bWeight, alpha, q_bias, d_input, mSize,  kSize,  nb,   num_groups);
}

void random_seed(){
  time_t t;
  srand((unsigned int)time(&t));
}
bool rand_bool(){
  return rand()>(RAND_MAX/2);
}
double rand_fp64(double max=1.0){
  double sign[] = {-1.0,1.0};
  return (double)sign[rand_bool()]*rand()/RAND_MAX*rand()/RAND_MAX*max;
}

float rand_fp32(float max=1.0){
  return rand_fp64()*max;
}

void makeRandomInput(torch::Tensor inp, int M, int K){
    for(int m=0;m<M;m++)
        for(int k=0;k<K;k++)
          inp.index_put_({m, k}, rand_fp32());
//            inp[m][k] = rand_fp32(); // (-1.0, 1.0) / 2^b
}

void makeRandomWeight(torch::Tensor qW, torch::Tensor bW, int N, int NUM_BITS, int K){
    for(int n=0;n<N;n++){
        for(int b=0;b<NUM_BITS;b++){
            for(int k=0;k<K;k+=32){  //32 단위
                int32_t s=0;
                for(int t=0;t<32;t++){
                    if(rand_bool()){
                            s |= 1<<t;
                            qW.index_put_({k + t, b, n}, 1);
                            //qW[k + t][b][n] = +1;
                    } else  qW.index_put_({k + t, b, n}, -1); //qW[k + t][b][n] = -1;
                }
                //at::Tensor t = CPU(at::kInt).ones();
                bW.index_put_({k/32, b, n}, s);
                //bW[k/32][b][n] = s;
            }
        }
    }
}

void makeRandomWeight_int3(torch::Tensor weight_int3, int N, int K_new){
    for(int n=0;n<N;n++){
        for(int k=0;k<K_new;k++){
          weight_int3.index_put_({k, n}, rand());
//            weight_int3[k][n] = rand();
        }
    }
}
void makeRandomAlpha(torch::Tensor alpha, torch::Tensor q_bias, int num_groups, int NUM_BITS, int N){
    for(int g=0;g<num_groups;g++)
        for(int n=0;n<N;n++){
          q_bias.index_put_({g, n}, rand_fp32()/(1<< NUM_BITS));
            //q_bias[g][n] = rand_fp32()/(1<< NUM_BITS);
            for(int b=0;b<NUM_BITS;b++)
            alpha.index_put_({g,b, n}, rand_fp32()/(1<<b));
//                alpha[g][b][n] = rand_fp32()/(1<<b); // (-1.0, 1.0) / 2^b
        }
}
void makeRandomScale(torch::Tensor scale, int N){
    for(int n=0;n<N;n++)
      scale.index_put_({n}, rand_fp32());
//        scale[n] = rand_fp32();
}

void makeRandomBias(torch::Tensor bias, int N){
    for(int n=0;n<N;n++)
      bias.index_put_({n}, rand_fp32());
//        bias[n] = rand_fp32();
}


std::vector<torch::Tensor> parsing(torch::Tensor bW, torch::Tensor A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups, torch::Tensor  q_bias, int cuda){
    int num_groups = num_alpha_groups;

    //__half* p_alpha;
    //__half* p_q_bias;
    int nb=num_bits;
    int kSize = 0, mSize = 0;
    if(is_row_wise_quantize){
        mSize = row; 
        kSize = col; 
    }
    else{
        mSize = col; 
        kSize = row;             
    }
    auto options_p =
  torch::TensorOptions()
    .dtype(at::kHalf)
    .device(torch::kCUDA, cuda);

  torch::Tensor p_q_bias, p_alpha;
    //if(q_bias == nullptr) {} //p_q_bias = nullptr;
    //else{
      p_q_bias = torch::empty(num_groups * mSize, options_p);
//        cudaMallocManaged(&p_q_bias    ,sizeof(__half  ) * num_groups * mSize);
        for(int i=0;i<num_groups*mSize;i++) {
            __half tmp = __float2half(q_bias.index({i}).item<float>());
          p_q_bias.index_put_({i},  *reinterpret_cast<at::Half*>(&tmp) );
        }
    //}
    p_alpha = torch::empty(num_groups * mSize * nb, options_p);
//    cudaMallocManaged(&p_alpha    ,sizeof(__half  ) * num_groups * mSize * nb);
    for(int i=0;i<num_groups*mSize*nb;i++) {
      __half tmp = __float2half(A.index({i}).item<float>());
      p_alpha.index_put_({i}, *reinterpret_cast<at::Half*>(&tmp )     );
    }
    //cudaMallocManaged(&bWeight  ,sizeof(uint32_t) * kSize * mSize * nb / 32);
    //int32
    auto options_bweight =
  torch::TensorOptions()
    .dtype(at::kInt)
    .device(torch::kCUDA, cuda);

    torch::Tensor bWeight = torch::empty({kSize * mSize * nb / 32}, options_bweight);
    cudaMemcpy(bWeight.data<int>() ,bW.data<int>(), sizeof(uint32_t) * kSize * mSize * nb / 32,    cudaMemcpyHostToDevice);
    //this->alpha = (void*)p_alpha;
    //this->q_bias = (void*)p_q_bias;
    return std::vector<torch::Tensor>({bWeight, p_q_bias, p_alpha});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lutgemm_compute", &lutgemm_compute, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("lutgemm_compute_block", &lutgemm_compute_block, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("lutgemm_compute_block_shift", &lutgemm_compute_block_shift, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("lutgemm_compute_block_shiftInt8", &lutgemm_compute_block_shiftInt8, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("lutgemm_compute_shift_scale", &lutgemm_compute_shift_scale, "pass");
  m.def("makeRandomInput", &makeRandomInput, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("makeRandomAlpha", &makeRandomAlpha, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("makeRandomBias", &makeRandomBias, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("makeRandomScale", &makeRandomScale, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("makeRandomWeight_int3", &makeRandomWeight_int3, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("makeRandomWeight", &makeRandomWeight, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("parsing", &parsing, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
}
