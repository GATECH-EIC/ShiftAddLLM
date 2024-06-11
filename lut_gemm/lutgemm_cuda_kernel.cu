#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "mv_fp16_bias.hpp"

class timer {
public:
    std::vector<double> arr;
    bool sort_flag = false;
    double s;
    void start(){
        s = omp_get_wtime();
    }
    double end(){
        double l = (omp_get_wtime() - s) * 1000.0;
        arr.push_back(l);
        sort_flag = false;
        return l;
    }

    double mean(){
        double sum=0;
        for(auto it : arr)
            sum += it;
        return sum/arr.size();
    }

    void sort(){
        if(sort_flag) return;
        std::sort(arr.begin(), arr.end());
        sort_flag = true;
    }

    
    double pile(float p){
        sort();
        int idx = (arr.size() - 1) * p;
        return arr[idx];
    }

    double max(){
        sort();
        return arr[arr.size() - 1];
    }
    double min(){
        sort();
        return arr[0];
    }

    timer(/* args */){}
    ~timer(){}
};
/*
void parsing(unsigned int *bW, float *A, int row, int col, int num_bits, 
        bool is_row_wise_quantize, int num_alpha_groups, float* q_bias){
    int num_groups = num_alpha_groups;
    int group_size =  kSize/num_alpha_groups;

    __half* p_alpha;
    __half* p_q_bias;
    int nb=num_bits;
    bool is_row_wise_quantize = is_row_wise_quantize;
    if(is_row_wise_quantize){
        mSize = row; 
        kSize = col; 
    }
    else{
        mSize = col; 
        kSize = row;             
    }

    if(q_bias == nullptr) p_q_bias = nullptr;
    else{
        cudaMallocManaged(&p_q_bias    ,sizeof(__half  ) * num_groups * mSize);
        for(int i=0;i<num_groups*mSize;i++) p_q_bias[i] = __float2half(q_bias[i]);
    }
    
    cudaMallocManaged(&p_alpha    ,sizeof(__half  ) * num_groups * mSize * nb);
    for(int i=0;i<num_groups*mSize*nb;i++) p_alpha[i] = __float2half(A[i]);

    cudaMallocManaged(&bWeight  ,sizeof(uint32_t) * kSize * mSize * nb / 32);
    cudaMemcpy(bWeight ,bW      ,sizeof(uint32_t) * kSize * mSize * nb / 32,    cudaMemcpyHostToDevice);
   void* alpha = (void*)p_alpha;
   void* q_bias = (void*)p_q_bias;
}*/

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  __half* __restrict__ vec,
    const       uint32_t* __restrict__ mat,
           __half* __restrict__ mul,
    const  __half* __restrict__ scales,
    const  __half* __restrict__ zeros,
    int height,
    int width
);

const int BLOCKWIDTH  = 1024;
const int BLOCKHEIGHT =   96; 

void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH 
  );
  dim3 threads(BLOCKWIDTH);
  //size: 49152 1152 12 48 1024
  //this: 49152 1152 12 48 1024
  //printf("size: %d %d %d %d %d\n",width, height, blocks.x, blocks.y, threads.x);
  int M = 1;
  int K = width;
  int N = width;
  int K_new = height;

    __half*  d_input;
    uint32_t*    d_weight_int3;
    __half* d_scale;
    __half* d_bias;
    __half* d_gptq_output;

        cudaMallocManaged(&d_input    , sizeof(float) * M * K);   
        cudaMallocManaged(&d_weight_int3, sizeof(int) * K_new * N);   
        cudaMallocManaged(&d_scale, sizeof(float) * N);   
        cudaMallocManaged(&d_bias, sizeof(float) * N);   
        cudaMallocManaged(&d_gptq_output, sizeof(float) * M * N);


  __half* v = (__half*)vec.data<at::Half>();
  uint32_t* m = (uint32_t*)mat.data<int32_t>();
  __half*mu = (__half*)mul.data<at::Half>();
  __half* s = (__half*)scales.data<at::Half>();
  __half*z = (__half*)zeros.data<at::Half>();
        timer tm;
        cudaDeviceSynchronize();
        for(int i=0;i<2;i++){
            tm.start();
         //   lutGEMM::matmul_gptq(m, n, k, (void*)scale, (void*)bias,
          //              (void*)A, (void*)B, (void*)C);
  //AT_DISPATCH_FLOATING_TYPES(
   // vec.type(), "vecquant3matmul_cuda", ([&] {
        VecQuant3MatMulKernel<__half> <<<blocks, threads>>>(
        d_input,d_weight_int3, d_gptq_output,
        d_scale, d_bias,
        height, width
      );

          //uint32_t *mat, __half *scales, __half *zeros,
    //__half *vec, __half *mul
      /*
      VecQuant3MatMulKernel<__half> <<<blocks, threads>>>(
        v,m, mu,
        s, z,
        height, width
      );*/
                  cudaDeviceSynchronize();
            tm.end();
        }

        printf("latencyx min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());
    //})
 // );
clock_t end = clock();
 //1152 49152
  //printf("cap: %d %d %d %d %d %d %lf\n", height, width, vec.size(0), vec.size(1), mul.size(0), mul.size(1), scales.size(0), zeros.size(0), (double)(end - start) / CLOCKS_PER_SEC);
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  __half* __restrict__ vec,
    const       uint32_t* __restrict__ mat,
           __half* __restrict__ mul,
    const  __half* __restrict__ scales,
    const  __half* __restrict__ zeros,
    int height,
    int width
) {
  int row = BLOCKHEIGHT * blockIdx.x;
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t scale = scales[col];
  scalar_t zero = zeros[col];

  scalar_t res = 0;
  int i = width * row + col;
  int k = 0;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    k += 10;
  }

  atomicAdd(&mul[col], res);
}


void lutgemm_compute_shift_scale_cuda(
   torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups
) {

      __half * d_input_ptr = (__half*)d_input.data<at::Half>();
     __half * output_ptr = (__half*)output.data<at::Half>();
      uint32_t*bWeight_ptr = (uint32_t*)bWeight.data<int32_t>();
       __half* alpha_ptr = (__half*)alpha.data<at::Half>();
//output, uint32_t*bWeight, __half* alpha,__half*q_bias, 
//int mSize, int kSize, int nb,  int num_groups, __half *input, int algo
    nqmv_bias_shift_mulq(output_ptr, bWeight_ptr, alpha_ptr,
mSize, kSize, nb,  num_groups, d_input_ptr, 0);

}


void lutgemm_cuda(
  torch::Tensor output,
  torch::Tensor bWeight,
  torch::Tensor alpha,
  torch::Tensor q_bias,
  torch::Tensor d_input,
  int mSize, int kSize, int nb,  int num_groups
) {

      __half * d_input_ptr = (__half*)d_input.data<at::Half>();
     __half * output_ptr = (__half*)output.data<at::Half>();
      uint32_t*bWeight_ptr = (uint32_t*)bWeight.data<int32_t>();
       __half* alpha_ptr = (__half*)alpha.data<at::Half>();
       __half*q_bias_ptr = (__half*)q_bias.data<at::Half>();
//output, uint32_t*bWeight, __half* alpha,__half*q_bias, 
//int mSize, int kSize, int nb,  int num_groups, __half *input, int algo
    nqmv_bias(output_ptr, bWeight_ptr, alpha_ptr,q_bias_ptr, 
mSize, kSize, nb,  num_groups, d_input_ptr, 0);


          //        cudaDeviceSynchronize();
        //    tm.end();
        //}

       // printf("latencyx min : %.5fms, max : %.5fms, avg:%.5f\n", tm.min(), tm.max(), tm.mean());
    //})
 // );
//clock_t end = clock();
}