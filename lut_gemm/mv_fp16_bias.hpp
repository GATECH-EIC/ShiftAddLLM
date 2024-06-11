#ifndef KERNELS_MV_FP16_BIAS_HPP
#define KERNELS_MV_FP16_BIAS_HPP




#include <stdio.h>
template<int K_TILE_SIZE>
__global__ void _nqmv_bias(uint32_t *W, __half *alpha, __half *q_bias, __half *input, __half *output,
            int M, int K, int NUM_BITS, int M_TILE_SIZE, int group_size){
    __shared__ __half lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
 
    int lut_y = threadIdx.x/lut_x_size;
    int lut_x = threadIdx.x%lut_x_size;

    __half *_inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
    __half base =    + __float2half((2 * ((lut_x>>0) & 1) - 1)) * _inp[0]
                     + __float2half((2 * ((lut_x>>1) & 1) - 1)) * _inp[1]
                     + __float2half((2 * ((lut_x>>2) & 1) - 1)) * _inp[2]
                     + __float2half((2 * ((lut_x>>3) & 1) - 1)) * _inp[3]
                     + __float2half((2 * ((lut_x>>4) & 1) - 1)) * _inp[4]
                     + __float2half((2 * ((lut_x>>5) & 1) - 1)) * _inp[5]
                     + __float2half((2 * ((lut_x>>6) & 1) - 1)) * _inp[6]
                     + __float2half((2 * ((lut_x>>7) & 1) - 1)) * _inp[7] ;
             
    lut[lut_y][lut_x] = base;

    int s = (lut_x_size==1)  ?0:
            (lut_x_size==2)  ?1:
            (lut_x_size==4)  ?2:
            (lut_x_size==8)  ?3:
            (lut_x_size==16) ?4:
            (lut_x_size==32) ?5:
            (lut_x_size==64) ?6: 
            (lut_x_size==128)?7: 8;  

    for(;s<8;s++){
        __half iValue =  __float2half(2)*_inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] =  lut[lut_y][i +  lut_x - (1 << s)] + iValue;
        }
    }
    __syncthreads();

    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x*2;
    int m_end = (blockIdx.x + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = blockDim.x * 2;

    uint32_t *bW = &W[blockIdx.y * K_TILE_SIZE/32 * NUM_BITS * M];
    int group_idx = (blockIdx.y * K_TILE_SIZE)/group_size;
    for(int m = m_start;m < m_end;m += m_step){
        __half reg_o0 = 0;
        __half reg_o1 = 0;

        {
            __half   reg_a0 = q_bias[group_idx*M + m + 0];
            __half   reg_a1 = q_bias[group_idx*M + m + 1];
            __half   reg_t_o0 = 0;
            __half   reg_t_o1 = 0;
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                reg_t_o0 +=  + lut[kt*4 + 0][255];
                reg_t_o0 +=  + lut[kt*4 + 1][255];
                reg_t_o0 +=  + lut[kt*4 + 2][255];
                reg_t_o0 +=  + lut[kt*4 + 3][255]; 

                reg_t_o1 +=  + lut[kt*4 + 0][255];
                reg_t_o1 +=  + lut[kt*4 + 1][255];
                reg_t_o1 +=  + lut[kt*4 + 2][255];
                reg_t_o1 +=  + lut[kt*4 + 3][255]; 
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }   
        __half   reg_a0 = alpha[group_idx*M + m + 0];
        __half   reg_a1 = alpha[group_idx*M + m + 1];     
        for(int b=0;b < NUM_BITS;b++){
            __half   reg_t_o0 = 0;
            __half   reg_t_o1 = 0;

            reg_a0 /= 2;
            reg_a1 /= 2;
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 
                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + lut[kt*4 + 3][reg_w3]; 
            }
            reg_o0 += reg_a0 * reg_t_o0;
            reg_o1 += reg_a1 * reg_t_o1;
        }
        atomicAdd((half2*)&output[m], __halves2half2(reg_o0, reg_o1));
    }
}

template<int K_TILE_SIZE>
__global__ void _nqmv_bias_shift_mulq(uint32_t *W, __half *alpha, __half *input, __half *output,
            int M, int K, int NUM_BITS, int M_TILE_SIZE, int group_size){
    __shared__ __half lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
 
    int lut_y = threadIdx.x/lut_x_size;
    int lut_x = threadIdx.x%lut_x_size;

    __half *_inp = &input[blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
    __half base =    + __float2half((2 * ((lut_x>>0) & 1) - 1)) * _inp[0]
                     + __float2half((2 * ((lut_x>>1) & 1) - 1)) * _inp[1]
                     + __float2half((2 * ((lut_x>>2) & 1) - 1)) * _inp[2]
                     + __float2half((2 * ((lut_x>>3) & 1) - 1)) * _inp[3]
                     + __float2half((2 * ((lut_x>>4) & 1) - 1)) * _inp[4]
                     + __float2half((2 * ((lut_x>>5) & 1) - 1)) * _inp[5]
                     + __float2half((2 * ((lut_x>>6) & 1) - 1)) * _inp[6]
                     + __float2half((2 * ((lut_x>>7) & 1) - 1)) * _inp[7] ;
             
    lut[lut_y][lut_x] = base;

    int s = (lut_x_size==1)  ?0:
            (lut_x_size==2)  ?1:
            (lut_x_size==4)  ?2:
            (lut_x_size==8)  ?3:
            (lut_x_size==16) ?4:
            (lut_x_size==32) ?5:
            (lut_x_size==64) ?6: 
            (lut_x_size==128)?7: 8;  

    for(;s<8;s++){
        __half iValue =  __float2half(2)*_inp[s];
        for (int i = (1 << s); i < (1 << (s + 1)); i += lut_x_size) {
            lut[lut_y][i + lut_x] =  lut[lut_y][i +  lut_x - (1 << s)] + iValue;
        }
    }
    __syncthreads();

    int m_start = blockIdx.x * M_TILE_SIZE + threadIdx.x*2;
    int m_end = (blockIdx.x + 1) * M_TILE_SIZE;
    m_end = (m_end < M) ? m_end : M;
    int m_step = blockDim.x * 2;

    uint32_t *bW = &W[blockIdx.y * K_TILE_SIZE/32 * NUM_BITS * M];
    int group_idx = (blockIdx.y * K_TILE_SIZE)/group_size;
    for(int m = m_start;m < m_end;m += m_step){
        __half reg_o0 = 0;
        __half reg_o1 = 0;

        //__half   reg_a0 = alpha[group_idx*M + m + 0]; // * q_bias[group_idx*M + m + 0];
        //__half   reg_a1 = alpha[group_idx*M + m + 1];// * q_bias[group_idx*M + m + 1];     
        for(int b=0;b < NUM_BITS;b++){
            __half   reg_a0 = alpha[group_idx*NUM_BITS*M + b * M + m + 0];
            __half   reg_a1 = alpha[group_idx*NUM_BITS*M + b * M + m + 1];

            __half   reg_t_o0 = 0;
            __half   reg_t_o1 = 0;

            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 
                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + lut[kt*4 + 3][reg_w3]; 
            }

            //*((int16_t*)&reg_t_o0) += (b << (10));
            //*((int16_t*)&reg_t_o1) += (b << (10));

            reg_o0 += reg_t_o0 * reg_a0;
            reg_o1 += reg_t_o1 * reg_a1;
        }
        atomicAdd((half2*)&output[m], __halves2half2(reg_o0 , reg_o1 ));
    }
}

inline int div_roundup(int x , int y){return (x + y - 1)/ y;}
template<int k_tile_size>
inline void _excute_nqmv_bias(__half *output,uint32_t*bWeight, __half* alpha,__half*q_bias, 
int mSize, int kSize, int nb,  int num_groups,
/*nQWeight_fp16 &nqW,*/ __half *input, int num_thraeds, int m_tile_size){
    dim3 grid(
        div_roundup(mSize, m_tile_size), 
        div_roundup(kSize, k_tile_size)); 
    dim3 block(num_thraeds);
    //printf("size: %d %d %d %d %d\n", nqW.mSize, nqW.kSize, nqW.nb, nqW.kSize, nqW.num_groups);
    _nqmv_bias<k_tile_size><<<grid, block>>>(bWeight, (__half*)alpha, (__half*)q_bias, 
            input, output, mSize, kSize, nb, m_tile_size, kSize/num_groups);
}

template<int k_tile_size>
inline void _excute_nqmv_bias_shift_mulq(__half *output,uint32_t*bWeight, __half* alpha,
int mSize, int kSize, int nb,  int num_groups,
/*nQWeight_fp16 &nqW,*/ __half *input, int num_thraeds, int m_tile_size){
    dim3 grid(
        div_roundup(mSize, m_tile_size), 
        div_roundup(kSize, k_tile_size)); 
    dim3 block(num_thraeds);
    _nqmv_bias_shift_mulq<k_tile_size><<<grid, block>>>(bWeight, (__half*)alpha, 
            input, output, mSize, kSize, nb, m_tile_size, kSize/num_groups);
}


inline auto _get_excute_nqmv_bias(int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(__half *output, uint32_t*bWeight, __half* alpha,__half*q_bias, 
int mSize, int kSize, int nb,  int num_groups, __half *input, int num_thraeds, int m_tile_size)= {
        _excute_nqmv_bias< 32*1>, 
        _excute_nqmv_bias< 32*2>, 
        _excute_nqmv_bias< 32*3>, 
        _excute_nqmv_bias< 32*4>, 
        _excute_nqmv_bias< 32*5>, 
        _excute_nqmv_bias< 32*6>, 
        _excute_nqmv_bias< 32*7>, 
        _excute_nqmv_bias< 32*8>, 
    };
    return funcs[k_tile_idx];
}

inline auto _get_excute_nqmv_bias_shift_mulq(int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(__half *output, uint32_t*bWeight, __half* alpha, 
int mSize, int kSize, int nb,  int num_groups, __half *input, int num_thraeds, int m_tile_size)= {
        _excute_nqmv_bias_shift_mulq< 32*1>, 
        _excute_nqmv_bias_shift_mulq< 32*2>, 
        _excute_nqmv_bias_shift_mulq< 32*3>, 
        _excute_nqmv_bias_shift_mulq< 32*4>, 
        _excute_nqmv_bias_shift_mulq< 32*5>, 
        _excute_nqmv_bias_shift_mulq< 32*6>, 
        _excute_nqmv_bias_shift_mulq< 32*7>, 
        _excute_nqmv_bias_shift_mulq< 32*8>, 
    };
    return funcs[k_tile_idx];
}



inline void nqmv_bias(__half *output, uint32_t*bWeight, __half* alpha,__half*q_bias, 
int mSize, int kSize, int nb,  int num_groups, __half *input, int algo){
    int k_tile_idx   =     0;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void  (*func)(__half *output, uint32_t*bWeight, __half* alpha,__half*q_bias, 
int mSize, int kSize, int nb,  int num_groups, __half *input, int num_thraeds, int m_tile_size) = _get_excute_nqmv_bias(num_thraeds, nb, k_tile_idx);
    func(output, bWeight, alpha, q_bias, mSize, kSize, nb, num_groups, input, num_thraeds, m_tile_size);
}

inline void nqmv_bias_shift_mulq(__half *output, uint32_t*bWeight, __half* alpha,
int mSize, int kSize, int nb,  int num_groups, __half *input, int algo){
    int k_tile_idx   =     0;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void  (*func)(__half *output, uint32_t*bWeight, __half* alpha,
int mSize, int kSize, int nb,  int num_groups, __half *input, int num_thraeds, int m_tile_size) = _get_excute_nqmv_bias_shift_mulq(num_thraeds, nb, k_tile_idx);
    func(output, bWeight, alpha, mSize, kSize, nb, num_groups, input, num_thraeds, m_tile_size);
}


#endif //KERNELS_MV_FP16_HPP
