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
__global__ void _nqmv_bias_block(uint32_t *W, __half *alpha, __half *input, __half *output,
            int M, int K, int NUM_BITS, int M_TILE_SIZE, int group_size, int in_size, int out_size){
    __shared__ __half lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
    const int ngroups = M / group_size;
    int lut_y = threadIdx.x/lut_x_size;
    int lut_x = threadIdx.x%lut_x_size;
    int ib = blockIdx.z;

    //for (int ib = 0;ib < batch;ib++){

    //if(ib > 0){
    //    __syncthreads();
    //}

    __half *_inp = &input[ib * in_size + blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
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
    
    __half loaded_alpha[12];
    int loaded_idx = -1;
    int group_idx = ((blockIdx.y) * K_TILE_SIZE)/8;

    for(int m = m_start;m < m_end;m += m_step){
        __half reg_o0 = 0;
        __half reg_o1 = 0;

        int N_group_idx = m / group_size;
        if(N_group_idx != loaded_idx){
            loaded_idx = N_group_idx;
            for(int b=0;b < NUM_BITS;b++){
                loaded_alpha[b * 4]  = alpha[group_idx*NUM_BITS*ngroups + b * ngroups + N_group_idx];
                loaded_alpha[b * 4 + 1] = alpha[(group_idx + 1)*NUM_BITS*ngroups + b * ngroups + N_group_idx];
                loaded_alpha[b * 4 + 2] = alpha[(group_idx + 2)*NUM_BITS*ngroups + b * ngroups + N_group_idx];
                loaded_alpha[b * 4 + 3] = alpha[(group_idx + 3)*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            }   
        }
        //int N_p1_group_idx = (m + 1) / group_size;
        for(int b=0;b < NUM_BITS;b++){
            __half   reg_t_o0 = 0;
            __half   reg_t_o1 = 0;

            __half  & reg_a00 = loaded_alpha[b * 4] ; //alpha[group_idx*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            __half  & reg_a01 = loaded_alpha[b * 4 + 1] ; //alpha[(group_idx + 1)*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            __half  & reg_a02 = loaded_alpha[b * 4 + 2] ; //alpha[(group_idx + 2)*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            __half  & reg_a03 = loaded_alpha[b * 4 + 3] ; //alpha[(group_idx + 3)*NUM_BITS*ngroups + b * ngroups + N_group_idx];


            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
            
                reg_t_o0 = 0;
                reg_t_o1 = 0;

                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 


                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + reg_a00 * lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + reg_a01 * lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + reg_a02 * lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + reg_a03 * lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + reg_a00 * lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + reg_a01 * lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + reg_a02 * lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + reg_a03 * lut[kt*4 + 3][reg_w3]; 

                reg_o0 += reg_t_o0 ;
                reg_o1 += reg_t_o1 ;

            }
        }
        
        atomicAdd((half2*)&output[ib * out_size + m], __halves2half2(reg_o0, reg_o1));
    }

    //}
}


template<int K_TILE_SIZE>
__global__ void _nqmv_bias_block_shift(uint32_t *W, uint16_t *alpha, __half *input, __half *output,
            int M, int K, int NUM_BITS, int M_TILE_SIZE, int group_size, int num_apot, int batch, int in_size, int out_size){
    __shared__ __half lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
    const int ngroups = M / group_size;
    int lut_y = threadIdx.x/lut_x_size;
    int lut_x = threadIdx.x%lut_x_size;

    for (int ib = 0;ib < batch;ib++){
    
    if(ib > 0){
        __syncthreads();
    }

    __half *_inp = &input[ib * in_size + blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
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
    
    for(int m = m_start;m < m_end;m += m_step){
        __half reg_o0 = 0;
        __half reg_o1 = 0;

        int N_group_idx = m / group_size;
        //int N_p1_group_idx = (m + 1) / group_size;
        for(int b=0;b < NUM_BITS;b++){
            __half   reg_t_o0 = 0;
            __half   reg_t_o1 = 0;

            int group_idx = ((blockIdx.y) * K_TILE_SIZE)/8;
            uint16_t   reg_a00 =  alpha[group_idx*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            uint16_t   reg_a01 = alpha[(group_idx + 1)*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            uint16_t   reg_a02 = alpha[(group_idx + 2)*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            uint16_t   reg_a03 = alpha[(group_idx + 3)*NUM_BITS*ngroups + b * ngroups + N_group_idx];

            __half m0 = (1 << (reg_a00 & 7)) + (1 << ((reg_a00 & 56) >> 3)) + (1 << ((reg_a00 & 448) >> 6));
            __half m1 = (1 << (reg_a01 & 7)) + (1 << ((reg_a01 & 56) >> 3)) + (1 << ((reg_a01 & 448) >> 6));
            __half m2 = (1 << (reg_a02 & 7)) + (1 << ((reg_a02 & 56) >> 3)) + (1 << ((reg_a02 & 448) >> 6));
            __half m3 = (1 << (reg_a03 & 7)) + (1 << ((reg_a03 & 56) >> 3)) + (1 << ((reg_a03 & 448) >> 6));
            
            //printf("set: %f %f %f %f %d\n", float(m0), float(m1), float(m2), float(m3), reg_a00);
            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
            
                reg_t_o0 = 0;
                reg_t_o1 = 0;

                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 


                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + m0 * lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + m1 * lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + m2 * lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + m3 * lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + m0 * lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + m1 * lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + m2 * lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + m3 * lut[kt*4 + 3][reg_w3]; 

                reg_o0 += reg_t_o0 ;
                reg_o1 += reg_t_o1 ;

            }
        }
        atomicAdd((half2*)&output[ib * out_size + m], __halves2half2(reg_o0, reg_o1));
    }

    }
}

//__device__ __inline__ __half shift_func(__half & inp, int8_t & shift)
//{
//    return inp * __half(1 << shift);
//}

__device__ __inline__ __half shift_func(__half inp, int8_t & shift)
{   
    *((int16_t*)&inp) += (shift << (10));

    return inp;//  + (shift << 10); // __half(1 << shift);
}

template<int K_TILE_SIZE>
__global__ void _nqmv_bias_block_shiftInt8(uint32_t *W, int8_t *alpha, __half *input, __half *output,
            int M, int K, int NUM_BITS, int M_TILE_SIZE, int group_size, int num_apot, int batch, int in_size, int out_size){
    __shared__ __half lut[K_TILE_SIZE/8][256];
    const int lut_x_size = blockDim.x / (K_TILE_SIZE/8);
    const int ngroups = M / group_size;
    int lut_y = threadIdx.x/lut_x_size;
    int lut_x = threadIdx.x%lut_x_size;

    for (int ib = 0;ib < batch;ib++){
    
    if(ib > 0){
        __syncthreads();
    }

    __half *_inp = &input[ib * in_size + blockIdx.y * K_TILE_SIZE + lut_y * 8];
    
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
    int8_t loaded_alpha[36];
    int loaded_idx = -1;
    int group_idx = ((blockIdx.y) * K_TILE_SIZE)/8;

    for(int m = m_start;m < m_end;m += m_step){
        __half reg_o0 = 0;
        __half reg_o1 = 0;

        int N_group_idx = m / group_size;
        if(N_group_idx != loaded_idx){
            loaded_idx = N_group_idx;
            for(int b=0;b < NUM_BITS;b++){

                loaded_alpha[b * 12]  = alpha[(group_idx*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3];
                loaded_alpha[b * 12 + 1] = alpha[(group_idx*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 1];
                loaded_alpha[b * 12 + 2] = alpha[(group_idx*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 2];

                loaded_alpha[b * 12 + 3] = alpha[((group_idx + 1)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3];
                loaded_alpha[b * 12 + 4] = alpha[((group_idx + 1)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 1];
                loaded_alpha[b * 12 + 5] = alpha[((group_idx + 1)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 2];

                loaded_alpha[b * 12 + 6] = alpha[((group_idx + 2)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3];
                loaded_alpha[b * 12 + 7] = alpha[((group_idx + 2)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 1];
                loaded_alpha[b * 12 + 8] = alpha[((group_idx + 2)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 2];

                loaded_alpha[b * 12 + 9] = alpha[((group_idx + 3)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3];
                loaded_alpha[b * 12 + 10] = alpha[((group_idx + 3)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 1];
                loaded_alpha[b * 12 + 11] = alpha[((group_idx + 3)*NUM_BITS*ngroups + b * ngroups + N_group_idx) * 3 + 2];
            }   
        }

        //int N_p1_group_idx = (m + 1) / group_size;
        for(int b=0;b < NUM_BITS;b++){
            __half   reg_t_o0 = 0;
            __half   reg_t_o1 = 0;

            __half m0 = 0, m1 = 0, m2 = 0, m3 = 0;
            int8_t &  reg_a00 = loaded_alpha[b * 12], & reg_a10 = loaded_alpha[b * 12 + 1], & reg_a20 = loaded_alpha[b * 12 + 2];
            int8_t &  reg_a01 = loaded_alpha[b * 12 + 3], & reg_a11 = loaded_alpha[b * 12 + 4], & reg_a21 = loaded_alpha[b * 12 + 5];
            int8_t &  reg_a02 = loaded_alpha[b * 12 + 6], & reg_a12 = loaded_alpha[b * 12 + 7], & reg_a22 = loaded_alpha[b * 12 + 8];
            int8_t &  reg_a03 = loaded_alpha[b * 12 + 9], & reg_a13 = loaded_alpha[b * 12 + 10], & reg_a23 = loaded_alpha[b * 12 + 11];

            /*
            m0 += 1 << reg_a00; 
            m0 += 1 << reg_a10;
            m0 += 1 << reg_a20;

            m1 += 1 << reg_a01;
            m1 += 1 << reg_a11;
            m1 += 1 << reg_a21;

            m2 += 1 << reg_a02;
            m2 += 1 << reg_a12;
            m2 += 1 << reg_a22;

            m3 += 1 << reg_a03;
            m3 += 1 << reg_a13;
            m3 += 1 << reg_a23;
*/

            for(int kt=0;kt < K_TILE_SIZE/32;kt++){
            
                reg_t_o0 = 0;
                reg_t_o1 = 0;
/*
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 

                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  + m0 * lut[kt*4 + 0][reg_w0];
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  + m1 * lut[kt*4 + 1][reg_w1];
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  + m2 * lut[kt*4 + 2][reg_w2];
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  + m3 * lut[kt*4 + 3][reg_w3]; 

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + m0 * lut[kt*4 + 0][reg_w0];
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  + m1 * lut[kt*4 + 1][reg_w1];
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  + m2 * lut[kt*4 + 2][reg_w2];
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  + m3 * lut[kt*4 + 3][reg_w3]; 
*/
                uint32_t reg_w = bW[kt * NUM_BITS * M + b * M + m]; 

                int reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o0 +=  shift_func(lut[kt*4 + 0][reg_w0], reg_a00) + shift_func(lut[kt*4 + 0][reg_w0], reg_a10) + shift_func(lut[kt*4 + 0][reg_w0], reg_a20);
                int reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o0 +=  shift_func(lut[kt*4 + 1][reg_w1], reg_a01) + shift_func(lut[kt*4 + 1][reg_w1], reg_a11) + shift_func(lut[kt*4 + 1][reg_w1], reg_a21);
                int reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o0 +=  shift_func(lut[kt*4 + 2][reg_w2], reg_a02) + shift_func(lut[kt*4 + 2][reg_w2], reg_a12) + shift_func(lut[kt*4 + 2][reg_w2], reg_a22);
                int reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o0 +=  shift_func(lut[kt*4 + 3][reg_w3], reg_a03) + shift_func(lut[kt*4 + 3][reg_w3], reg_a13) + shift_func(lut[kt*4 + 3][reg_w3], reg_a23);

                reg_w = bW[kt * NUM_BITS * M + b * M + m + 1]; 
                reg_w0 = (reg_w >> 8 * 0) & 255;        reg_t_o1 +=  + shift_func(lut[kt*4 + 0][reg_w0], reg_a00) + shift_func(lut[kt*4 + 0][reg_w0], reg_a10) + shift_func(lut[kt*4 + 0][reg_w0], reg_a20);
                reg_w1 = (reg_w >> 8 * 1) & 255;        reg_t_o1 +=  shift_func(lut[kt*4 + 1][reg_w1], reg_a01) + shift_func(lut[kt*4 + 1][reg_w1], reg_a11) + shift_func(lut[kt*4 + 1][reg_w1], reg_a21);
                reg_w2 = (reg_w >> 8 * 2) & 255;        reg_t_o1 +=  shift_func(lut[kt*4 + 2][reg_w2], reg_a02) + shift_func(lut[kt*4 + 2][reg_w2], reg_a12) + shift_func(lut[kt*4 + 2][reg_w2], reg_a22);
                reg_w3 = (reg_w >> 8 * 3) & 255;        reg_t_o1 +=  shift_func(lut[kt*4 + 3][reg_w3], reg_a03) + shift_func(lut[kt*4 + 3][reg_w3], reg_a13) + shift_func(lut[kt*4 + 3][reg_w3], reg_a23);



                reg_o0 += reg_t_o0 ;
                reg_o1 += reg_t_o1 ;

            }
        }
        atomicAdd((half2*)&output[ib * out_size + m], __halves2half2(reg_o0, reg_o1));
    }

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
inline void _excute_nqmv_bias_block(__half *output,uint32_t*bWeight, __half* alpha,
int mSize, int kSize, int nb,  int num_groups,
/*nQWeight_fp16 &nqW,*/ __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size){
    dim3 grid(
        div_roundup(mSize, m_tile_size), 
        div_roundup(kSize, k_tile_size),
        batch); 
    dim3 block(num_thraeds);
    //printf("size: %d %d %d %d %d\n", nqW.mSize, nqW.kSize, nqW.nb, nqW.kSize, nqW.num_groups);
    _nqmv_bias_block<k_tile_size><<<grid, block>>>(bWeight, (__half*)alpha,
            input, output, mSize, kSize, nb, m_tile_size, mSize/num_groups, in_size, out_size);
}

template<int k_tile_size>
inline void _excute_nqmv_bias_block_shift(__half *output,uint32_t*bWeight, uint16_t* alpha,
int mSize, int kSize, int nb,  int num_groups, int num_apot,
/*nQWeight_fp16 &nqW,*/ __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size){
    dim3 grid(
        div_roundup(mSize, m_tile_size), 
        div_roundup(kSize, k_tile_size)); 
    dim3 block(num_thraeds);
    //printf("size: %d %d %d %d %d\n", nqW.mSize, nqW.kSize, nqW.nb, nqW.kSize, nqW.num_groups);
    _nqmv_bias_block_shift<k_tile_size><<<grid, block>>>(bWeight, (uint16_t*)alpha,
            input, output, mSize, kSize, nb, m_tile_size, mSize/num_groups, num_apot, batch, in_size, out_size);
}


template<int k_tile_size>
inline void _excute_nqmv_bias_block_shiftInt8(__half *output,uint32_t*bWeight, int8_t* alpha,
int mSize, int kSize, int nb,  int num_groups, int num_apot,
/*nQWeight_fp16 &nqW,*/ __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size){
    dim3 grid(
        div_roundup(mSize, m_tile_size), 
        div_roundup(kSize, k_tile_size)); 
    dim3 block(num_thraeds);
    //printf("size: %d %d %d %d %d\n", nqW.mSize, nqW.kSize, nqW.nb, nqW.kSize, nqW.num_groups);
    _nqmv_bias_block_shiftInt8<k_tile_size><<<grid, block>>>(bWeight, (int8_t*)alpha,
            input, output, mSize, kSize, nb, m_tile_size, mSize/num_groups, num_apot, batch, in_size, out_size);
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

inline auto _get_excute_nqmv_bias_block(int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(__half *output, uint32_t*bWeight, __half* alpha, 
int mSize, int kSize, int nb,  int num_groups, __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size)= {
        _excute_nqmv_bias_block< 32*1>, 
        _excute_nqmv_bias_block< 32*2>, 
        _excute_nqmv_bias_block< 32*3>, 
        _excute_nqmv_bias_block< 32*4>, 
        _excute_nqmv_bias_block< 32*5>, 
        _excute_nqmv_bias_block< 32*6>, 
        _excute_nqmv_bias_block< 32*7>, 
        _excute_nqmv_bias_block< 32*8>, 
    };
    return funcs[k_tile_idx];
}

inline auto _get_excute_nqmv_bias_block_shift(int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(__half *output, uint32_t*bWeight, uint16_t* alpha, 
int mSize, int kSize, int nb,  int num_groups, int num_apot, __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size)= {
        _excute_nqmv_bias_block_shift< 32*1>, 
        _excute_nqmv_bias_block_shift< 32*2>, 
        _excute_nqmv_bias_block_shift< 32*3>, 
        _excute_nqmv_bias_block_shift< 32*4>, 
        _excute_nqmv_bias_block_shift< 32*5>, 
        _excute_nqmv_bias_block_shift< 32*6>, 
        _excute_nqmv_bias_block_shift< 32*7>, 
        _excute_nqmv_bias_block_shift< 32*8>, 
    };
    return funcs[k_tile_idx];
}
inline auto _get_excute_nqmv_bias_block_shiftInt8(int num_thraeds, size_t bits, size_t k_tile_idx){
    void (*funcs[])(__half *output, uint32_t*bWeight, int8_t* alpha, 
int mSize, int kSize, int nb,  int num_groups, int num_apot, __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size)= {
        _excute_nqmv_bias_block_shiftInt8< 32*1>, 
        _excute_nqmv_bias_block_shiftInt8< 32*2>, 
        _excute_nqmv_bias_block_shiftInt8< 32*3>, 
        _excute_nqmv_bias_block_shiftInt8< 32*4>, 
        _excute_nqmv_bias_block_shiftInt8< 32*5>, 
        _excute_nqmv_bias_block_shiftInt8< 32*6>, 
        _excute_nqmv_bias_block_shiftInt8< 32*7>, 
        _excute_nqmv_bias_block_shiftInt8< 32*8>, 
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

inline void nqmv_bias_block(__half *output, uint32_t*bWeight, __half* alpha,
int mSize, int kSize, int nb,  int num_groups, __half *input, int algo, int batch, int in_size, int out_size){
    int k_tile_idx   =     0;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void  (*func)(__half *output, uint32_t*bWeight, __half* alpha, 
int mSize, int kSize, int nb,  int num_groups, __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size) = _get_excute_nqmv_bias_block(num_thraeds, nb, k_tile_idx);
    func(output, bWeight, alpha, mSize, kSize, nb, num_groups, input, num_thraeds, m_tile_size, batch, in_size, out_size);
}

inline void nqmv_bias_block_shift(__half *output, uint32_t*bWeight, uint16_t* alpha,
int mSize, int kSize, int nb,  int num_groups, int num_apot, __half *input, int algo, int batch, int in_size, int out_size){
    int k_tile_idx   =     0;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void  (*func)(__half *output, uint32_t*bWeight, uint16_t* alpha, 
int mSize, int kSize, int nb,  int num_groups, int num_apot, __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size) = _get_excute_nqmv_bias_block_shift(num_thraeds, nb, k_tile_idx);
    func(output, bWeight, alpha, mSize, kSize, nb, num_groups, num_apot, input, num_thraeds, m_tile_size, batch, in_size, out_size);
}

inline void nqmv_bias_block_shiftInt8(__half *output, uint32_t*bWeight, int8_t* alpha,
int mSize, int kSize, int nb,  int num_groups, int num_apot, __half *input, int algo, int batch, int in_size, int out_size){
    int k_tile_idx   =     0;
    int m_tile_size  =  2048;
    int num_thraeds  =   256;
    void  (*func)(__half *output, uint32_t*bWeight, int8_t* alpha, 
int mSize, int kSize, int nb,  int num_groups, int num_apot, __half *input, int num_thraeds, int m_tile_size, int batch, int in_size, int out_size) = _get_excute_nqmv_bias_block_shiftInt8(num_thraeds, nb, k_tile_idx);
    func(output, bWeight, alpha, mSize, kSize, nb, num_groups, num_apot, input, num_thraeds, m_tile_size, batch, in_size, out_size);
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

