/**
 *  Linear algebra methods modified from: https://github.com/ericjang/svd3
 */

#include "registration_utils_gpu.h"

#include <stdio.h>

#include <float.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

#include <curand_kernel.h>
#include <pcl/gpu/utils/safe_call.hpp>
#include <boost/chrono.hpp>

#include <locale.h>         // Radix character

#define _gamma 5.828427124  // FOUR_GAMMA_SQUARED = sqrt(8)+3;
#define _cstar 0.923879532  // cos(pi/8)
#define _sstar 0.3826834323 // sin(p/8)
#define EPSILON 1e-6

using namespace pcl;

namespace cudaransac
{

float RegistrationUtilsGPU::ransac_threshold= -1.0;
float RegistrationUtilsGPU::ransac_radius   = -1.0;
float RegistrationUtilsGPU::ransac_poly     = -1.0;
int   RegistrationUtilsGPU::ransac_converged= -1;
int   RegistrationUtilsGPU::ransac_inliers  = -1;

//=========================================================================================================
//=================================== CALCULATE NEAREST NEIGHBOUR INDEX ===================================
//=========================================================================================================

/**
 *  @brief      gpuCalculateNeighbors   For every point in source calculate index of the nearest point in target in the features
 *                                      space. The result save in neighbors array.
 *  @param[in]  source_features         Source feature cloud.
 *  @param[in]  source_step             Source feature step in bytes.
 *  @param[in]  target_features         Target feature cloud.
 *  @param[in]  target_rows             Number of target features.
 *  @param[out] neighbors               Array of the nearest neighbor of corresponding point in target cloud.
 */
__global__ void gpuCalculateNeighbors(float* source_features, size_t source_step,
                                      float* target_features, int target_rows,
                                      int *neighbors)
{
    //Shared memory declaration
    __shared__ int min_idx[1024];
    __shared__ float min_distance[1024];

    //Some help variables
    unsigned int source_point_index = blockIdx.x;
    unsigned int source_mem_index = blockIdx.x*(source_step/sizeof(float));
    unsigned int loop_limit = target_rows/gridDim.x + ((bool)(target_rows%gridDim.x));

    //Share memory initialization
    min_idx[threadIdx.x] = 0;
    min_distance[threadIdx.x] = FLT_MAX;

    //Loop after every block fit in target cloud
    for (unsigned int loop_idx=0; loop_idx<loop_limit; loop_idx++)
    {
        //Local distance
        float distance = 0;

        //Count target point index
        unsigned int target_point_index = (threadIdx.x + loop_idx*blockDim.x);
        unsigned int target_mem_index = target_point_index*(source_step/sizeof(float));

        //If target point index is not in bounds break the lopp
        if (target_point_index>=target_rows) break;

        //Loop for every bin
        for (unsigned int bin_idx=0; bin_idx<33; bin_idx++)
        {
            float dist_help = source_features[source_mem_index+bin_idx] - target_features[target_mem_index+bin_idx];
            distance += dist_help*dist_help;
        }

        //Remember min distance & index
        if (distance < min_distance[threadIdx.x])
        {
            min_idx[threadIdx.x] = target_point_index;
            min_distance[threadIdx.x] = distance;
        }
    }

    //Barier
    __syncthreads();

    //Reduce
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        //Remember distance in first half
        if (threadIdx.x < s)
        {
            if (min_distance[threadIdx.x] > min_distance[threadIdx.x + s])
            {
                min_idx[threadIdx.x] = min_idx[threadIdx.x + s];
                min_distance[threadIdx.x] = min_distance[threadIdx.x + s];
            }
        }

        //Barier
        __syncthreads();
    }

    //Save result
    if (threadIdx.x==0)
        neighbors[source_point_index] = min_idx[threadIdx.x];
}

__global__ void gpuCalculateNeighborsA(float* source_features, size_t source_step,
                                       float* target_features, int target_rows,
                                       int* indices, float* distances)
{
    // Shared memory declaration
    __shared__ int indices_sh[32];
    __shared__ int distances_sh[1024];

    // Shared idx
    int shared_idx = threadIdx.x + threadIdx.y*blockDim.x;

    //Some help variables
    unsigned int source_point_index = blockIdx.y;
    unsigned int source_mem_index = source_point_index*(source_step/sizeof(float));

    //Count target point index
    unsigned int target_point_index = blockIdx.x*32 + threadIdx.y;
    unsigned int target_mem_index = target_point_index*(source_step/sizeof(float));

    //If target point index is not in bounds break the lopp
    if (target_point_index>=target_rows) return;

    //Share memory initialization
    float src = source_features[source_mem_index+threadIdx.x];
    float dst = target_features[target_mem_index+threadIdx.x];
    distances_sh[shared_idx] = (src-dst)*(src-dst);
    if (threadIdx.x == 0) indices_sh[threadIdx.y] = target_point_index;

    //Barier
    __syncthreads();

    //Reduce X (SUM)
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        //Add distance to the first half
        if (threadIdx.x < s)
        {
            distances_sh[shared_idx] += distances_sh[shared_idx + s];
        }

        //Barier
        __syncthreads();
    }

//    //Reduce Y (MIN)
//    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
//    {
//        //Remember distance in first half
//        if (threadIdx.y < s && threadIdx.x == 0)
//        {
//            if (distances_sh[shared_idx] > distances_sh[shared_idx + s])
//            {
//                indices_sh[threadIdx.y] = indices_sh[threadIdx.y + s];
//                distances_sh[shared_idx] = distances_sh[shared_idx + s];
//            }
//        }

//        //Barier
//        __syncthreads();
//    }

    //Save result
    if (shared_idx==0)
    {
        int global_idx = blockIdx.x + blockIdx.y*gridDim.x;
        indices[global_idx] = indices_sh[shared_idx];
        distances[global_idx] = distances_sh[shared_idx];
    }
}

__global__ void gpuCalculateNeighborsB(float* source_features, size_t source_step,
                                      float* target_features, int target_rows,
                                      int *neighbors)
{
    //Shared memory declaration
    __shared__ int min_idx[1024];
    __shared__ float min_distance[1024];

    //Some help variables
    unsigned int source_point_index = blockIdx.x;
    unsigned int source_mem_index = blockIdx.x*(source_step/sizeof(float));
    unsigned int loop_limit = target_rows/gridDim.x + ((bool)(target_rows%gridDim.x));

    //Share memory initialization
    min_idx[threadIdx.x] = 0;
    min_distance[threadIdx.x] = FLT_MAX;

    unsigned int loop_idx=0;
    //Loop after every block fit in target cloud
    for (unsigned int loop_idx=0; loop_idx<loop_limit; loop_idx++)
    {
        //Local distance
        float distance = 0;

        //Count target point index
        unsigned int target_point_index = (threadIdx.x + loop_idx*blockDim.x);
        unsigned int target_mem_index = target_point_index*(source_step/sizeof(float));

        //If target point index is not in bounds break the lopp
        if (target_point_index>=target_rows) break;

        //Loop for every bin
        for (unsigned int bin_idx=0; bin_idx<33; bin_idx++)
        {
            float dist_help = source_features[source_mem_index+bin_idx] - target_features[target_mem_index+bin_idx];
            distance += dist_help*dist_help;
        }

        //Remember min distance & index
        if (distance < min_distance[threadIdx.x])
        {
            min_idx[threadIdx.x] = target_point_index;
            min_distance[threadIdx.x] = distance;
        }
    }

    //Barier
    __syncthreads();

    //Reduce
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        //Remember distance in first half
        if (threadIdx.x < s)
        {
            if (min_distance[threadIdx.x] > min_distance[threadIdx.x + s])
            {
                min_idx[threadIdx.x] = min_idx[threadIdx.x + s];
                min_distance[threadIdx.x] = min_distance[threadIdx.x + s];
            }
        }

        //Barier
        __syncthreads();
    }

    //Save result
    if (threadIdx.x==0)
        neighbors[source_point_index] = min_idx[threadIdx.x];
}

void RegistrationUtilsGPU::calculateNeighborIndex(float* d_source_features, int source_rows, size_t source_step,
                                                  float* d_target_features, int target_rows, size_t target_step,
                                                  int **d_neighbor_index)
{
    cudaSafeCall(cudaMalloc((void**)d_neighbor_index, source_rows*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)*d_neighbor_index, 0, source_rows*sizeof(int)));

//    // How many blocks per one source point
//    int t = target_rows/32 + (bool)(target_rows%32);
//    dim3 dimGrid(t, source_rows/2, 1);    // One row processing one source point
//    dim3 dimBlock(32, 32, 1);           // One row processing one target point

//    // Allocate memory for min distances and indices
//    float* d_distances;
//    cudaSafeCall(cudaMalloc((void**)&d_distances, t*source_rows*sizeof(float)));
//    cudaSafeCall(cudaMemset((void*)d_distances, 0, t*source_rows*sizeof(float)));
//    int* d_indices;
//    cudaSafeCall(cudaMalloc((void**)&d_indices, t*source_rows*sizeof(int)));
//    cudaSafeCall(cudaMemset((void*)d_indices, 0, t*source_rows*sizeof(int)));


//    std::cout << "SOURCE ROWS = " << source_rows << std::endl;
//    std::cout << "SOURCE STEP = " << source_step << std::endl;
//    std::cout << "TARGET ROWS = " << target_rows << std::endl;
//    std::cout << "TARGET STEP = " << target_step << std::endl;
//    std::cout << "t           = " << t << std::endl;
//    std::cout << "THREADS     = " << min(target_rows, 1024) << std::endl;
//    std::cout << "BLOCKS      = " << t*source_rows << std::endl;

//    gpuCalculateNeighborsA<<< dimGrid, dimBlock >>>(d_source_features, source_step,
//                                                    d_target_features, target_rows,
//                                                    d_indices, d_distances);


//    cudaSafeCall(cudaDeviceSynchronize());
//    cudaSafeCall(cudaFree((void*)d_distances));
//    cudaSafeCall(cudaFree((void*)d_indices));


    gpuCalculateNeighbors <<< source_rows, 1024 >>> (d_source_features, source_step,d_target_features, target_rows,*d_neighbor_index);

    cudaSafeCall(cudaDeviceSynchronize());
}

void RegistrationUtilsGPU::downloadNeighborsIndex(int *d_neighbor_index, int *h_neighbor_index, size_t bytes)
{
    cudaSafeCall(cudaMemcpy((void*)h_neighbor_index, (void*)d_neighbor_index, bytes, cudaMemcpyDeviceToHost));
}

//=========================================================================================================
//============================================== MATH UTILS ===============================================
//=========================================================================================================

__forceinline__ __device__ float gpuMathAbs(float x)
{
    return x > 0 ? x : -x;
}
__forceinline__ __device__ float gpuMathMax(float &x, float &y)
{
    return x > y ? x : y;
}
__forceinline__ __device__ float gpuMathRsqrt(float x)
{
   float xhalf = 0.5f*x;
   int i = *(int *)&x;          // View x as an int.
   i = 0x5f375a82 - (i >> 1);   // Initial guess (slightly better).
   x = *(float *)&i;            // View i as float.
   x = x*(1.5f - xhalf*x*x);    // Newton step.
   return x;
}
__forceinline__ __device__ float gpuAccurateSqrt(float x)
{
    return x * gpuMathRsqrt(x);
}
__forceinline__ __device__ void gpuMathCondSwap(bool c, float &X, float &Y)
{
    // used in step 2
    float Z = X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}
__forceinline__ __device__ void gpuMathCondNegSwap(bool c, float &X, float &Y)
{
    // used in step 2 and 3
    float Z = -X;
    X = c ? Y : X;
    Y = c ? Z : Y;
}
__forceinline__ __device__ float gpuMathDist2(float x, float y, float z)
{
    return x*x+y*y+z*z;
}

//=========================================================================================================
//========================================== LINEAR ALGEBRA UTILS =========================================
//=========================================================================================================


// matrix multiplication M = A * B
__forceinline__ __device__ void gpuLaMultiplicationAB(float* a, float* b, float *m)
{
    m[0]=a[0]*b[0] + a[3]*b[1] + a[6]*b[2]; m[3]=a[0]*b[3] + a[3]*b[4] + a[6]*b[5]; m[6]=a[0]*b[6] + a[3]*b[7] + a[6]*b[8];
    m[1]=a[1]*b[0] + a[4]*b[1] + a[7]*b[2]; m[4]=a[1]*b[3] + a[4]*b[4] + a[7]*b[5]; m[7]=a[1]*b[6] + a[4]*b[7] + a[7]*b[8];
    m[2]=a[2]*b[0] + a[5]*b[1] + a[8]*b[2]; m[5]=a[2]*b[3] + a[5]*b[4] + a[8]*b[5]; m[8]=a[2]*b[6] + a[5]*b[7] + a[8]*b[8];
}

// matrix multiplication M = Transpose[A] * B
__forceinline__ __device__ void gpuLaMultiplicationAtB(float* a, float* b, float *m)
{
  m[0]=a[0]*b[0] + a[3]*b[1] + a[6]*b[2]; m[3]=a[0]*b[3] + a[3]*b[4] + a[6]*b[5]; m[6]=a[0]*b[6] + a[3]*b[7] + a[6]*b[8];
  m[1]=a[1]*b[0] + a[4]*b[1] + a[7]*b[2]; m[4]=a[1]*b[3] + a[4]*b[4] + a[7]*b[5]; m[7]=a[1]*b[6] + a[4]*b[7] + a[7]*b[8];
  m[2]=a[2]*b[0] + a[5]*b[1] + a[8]*b[2]; m[5]=a[2]*b[3] + a[5]*b[4] + a[8]*b[5]; m[8]=a[2]*b[6] + a[5]*b[7] + a[8]*b[8];
}

// matrix multiplication M = Transpose[A] * Transpose[B]
__forceinline__ __device__ void gpuLaMultiplicationAtBt(float* a, float* b, float* m)
{
    m[0] = a[0]*b[0] + a[3]*b[1] + a[6]*b[2];  m[3] = a[0]*b[3] + a[3]*b[4] + a[6]*b[5];  m[6] = a[0]*b[6] + a[3]*b[7] + a[6]*b[8];
    m[1] = a[1]*b[0] + a[4]*b[1] + a[7]*b[2];  m[4] = a[1]*b[3] + a[4]*b[4] + a[7]*b[5];  m[7] = a[1]*b[6] + a[4]*b[7] + a[7]*b[8];
    m[2] = a[2]*b[0] + a[5]*b[1] + a[8]*b[2];  m[5] = a[2]*b[3] + a[5]*b[4] + a[8]*b[5];  m[8] = a[2]*b[6] + a[5]*b[7] + a[8]*b[8];
}

__forceinline__ __device__ void gpuLaQuaternionToMatrix3(const float * qV, float* m)
{
    float w = qV[3];
    float x = qV[0];
    float y = qV[1];
    float z = qV[2];

    float qxx = x*x;
    float qyy = y*y;
    float qzz = z*z;
    float qxz = x*z;
    float qxy = x*y;
    float qyz = y*z;
    float qwx = w*x;
    float qwy = w*y;
    float qwz = w*z;

    m[0]=1 - 2*(qyy + qzz); m[3]=2*(qxy - qwz);     m[6]=2*(qxz + qwy);
    m[1]=2*(qxy + qwz);     m[4]=1 - 2*(qxx + qzz); m[7]=2*(qyz - qwx);
    m[2]=2*(qxz - qwy);     m[5]=2*(qyz + qwx);     m[8]=1 - 2*(qxx + qyy);
}

__forceinline__ __device__ void gpuLaApproximateGivensQuaternion(float a11, float a12, float a22,
                                                                 float &ch, float &sh)
{
    /*
     * Given givens angle computed by approximateGivensAngles,
     * compute the corresponding rotation quaternion.
     */
    ch = 2*(a11-a22);
    sh = a12;
    bool b = _gamma*sh*sh < ch*ch;
    float w = gpuMathRsqrt(ch*ch+sh*sh);
    ch=b?w*ch:_cstar;
    sh=b?w*sh:_sstar;
}

__forceinline__ __device__ void gpuLaJacobiConjugation(const int x, const int y, const int z,
                                                       float* s, float* qV)
{
    float ch,sh;
    gpuLaApproximateGivensQuaternion(s[0],s[1],s[4],ch,sh);

    float scale = ch*ch+sh*sh;
    float a = (ch*ch-sh*sh)/scale;
    float b = (2*sh*ch)/scale;

    // make temp copy of S
    float _s[9] = { s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8] };

    // perform conjugation S = Q'*S*Q
    // Q already implicitly solved from a, b
    s[0] =a*(a*_s[0] + b*_s[1]) + b*(a*_s[1] + b*_s[4]);
    s[1] =a*(-b*_s[0] + a*_s[1]) + b*(-b*_s[1] + a*_s[4]);	s[4]=-b*(-b*_s[0] + a*_s[1]) + a*(-b*_s[1] + a*_s[4]);
    s[2] =a*_s[2] + b*_s[5];								s[5]=-b*_s[2] + a*_s[5];                                s[8]=_s[8];

    // update cumulative rotation qV
    float tmp[3];
    tmp[0]=qV[0]*sh;
    tmp[1]=qV[1]*sh;
    tmp[2]=qV[2]*sh;
    sh *= qV[3];

    qV[0] *= ch;
    qV[1] *= ch;
    qV[2] *= ch;
    qV[3] *= ch;

    // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
    // for (p,q) = ((0,1),(1,2),(0,2))
    qV[z] += sh;
    qV[3] -= tmp[z]; // w
    qV[x] += tmp[y];
    qV[y] -= tmp[x];

    // re-arrange matrix for next iteration
    _s[0] = s[4];
    _s[1] = s[5]; _s[4] = s[8];
    _s[2] = s[1]; _s[5] = s[2]; _s[8] = s[0];
    s[0] = _s[0];
    s[1] = _s[1]; s[4] = _s[4];
    s[2] = _s[2]; s[5] = _s[5]; s[8] = _s[8];
}

// finds transformation that diagonalizes a symmetric matrix
// s  - symmetric matrix
// qV - quaternion representation of V
__forceinline__ __device__ void gpuLaJacobiEigenAnlysis(float* s, float* qV)
{
    qV[3]=1; qV[0]=0;qV[1]=0;qV[2]=0; // follow same indexing convention as GLM
    for (int i=0;i<4;i++)
    {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
        //  asymptotic convergence
        gpuLaJacobiConjugation(0,1,2,s,qV); // p,q = 0,1
        gpuLaJacobiConjugation(1,2,0,s,qV); // p,q = 1,2
        gpuLaJacobiConjugation(2,0,1,s,qV); // p,q = 0,2
    }
}

// b  - matrix we want to decompose
// v  - sort v simulaneously
__forceinline__ __device__ void gpuLaSortSingularValues(float* b, float* v)
{
    float rho1 = gpuMathDist2(b[0],b[1],b[2]);
    float rho2 = gpuMathDist2(b[3],b[4],b[7]);
    float rho3 = gpuMathDist2(b[6],b[7],b[8]);
    bool c;
    c = rho1 < rho2;
    gpuMathCondNegSwap(c,b[0],b[3]); gpuMathCondNegSwap(c,v[0],v[3]);
    gpuMathCondNegSwap(c,b[1],b[4]); gpuMathCondNegSwap(c,v[1],v[4]);
    gpuMathCondNegSwap(c,b[2],b[5]); gpuMathCondNegSwap(c,v[2],v[5]);
    gpuMathCondSwap(c,rho1,rho2);
    c = rho1 < rho3;
    gpuMathCondNegSwap(c,b[0],b[6]); gpuMathCondNegSwap(c,v[0],v[6]);
    gpuMathCondNegSwap(c,b[1],b[7]); gpuMathCondNegSwap(c,v[1],v[7]);
    gpuMathCondNegSwap(c,b[2],b[8]); gpuMathCondNegSwap(c,v[2],v[8]);
    gpuMathCondSwap(c,rho1,rho3);
    c = rho2 < rho3;
    gpuMathCondNegSwap(c,b[3],b[6]); gpuMathCondNegSwap(c,v[3],v[6]);
    gpuMathCondNegSwap(c,b[4],b[7]); gpuMathCondNegSwap(c,v[4],v[7]);
    gpuMathCondNegSwap(c,b[5],b[8]); gpuMathCondNegSwap(c,v[5],v[8]);
}


__forceinline__ __device__ void gpuLaQRGivensQuaternion(float a1, float a2,
                                                        float &ch, float &sh)
{
    // a1 = pivot point on diagonal
    // a2 = lower triangular entry we want to annihilate
    float epsilon = EPSILON;
    float rho = gpuAccurateSqrt(a1*a1 + a2*a2);

    sh = rho > epsilon ? a2 : 0;
    ch = fabs(a1) + fmax(rho,epsilon);
    bool b = a1 < 0;
    gpuMathCondSwap(b,sh,ch);
    float w = gpuMathRsqrt(ch*ch+sh*sh);
    ch *= w;
    sh *= w;
}
// b  - matrix we want to decompose
// q  - output q matrix
// r  - output r matrix
__forceinline__ __device__ void gpuLaQRD(float* b, float* q, float* r)
{
    float ch1,sh1,ch2,sh2,ch3,sh3;
    float x,y;

    // first givens rotation (ch,0,0,sh)
    gpuLaQRGivensQuaternion(b[0],b[1],ch1,sh1);
    x=1-2*sh1*sh1;
    y=2*ch1*sh1;
    // apply B = Q' * B
    r[0]=x*b[0]+y*b[1];     r[3]=x*b[3]+y*b[4];     r[6]=x*b[6]+y*b[7];
    r[1]=-y*b[0]+x*b[1];    r[4]=-y*b[3]+x*b[4];    r[7]=-y*b[6]+x*b[7];
    r[2]=b[2];              r[5]=b[5];              r[8]=b[8];

    // second givens rotation (ch,0,-sh,0)
    gpuLaQRGivensQuaternion(r[0],r[2],ch2,sh2);
    x=1-2*sh2*sh2;
    y=2*ch2*sh2;
    // apply B = Q' * B;
    b[0]=x*r[0]+y*r[2];     b[3]=x*r[3]+y*r[5];     b[6]=x*r[6]+y*r[8];
    b[1]=r[1];              b[4]=r[4];              b[7]=r[7];
    b[2]=-y*r[0]+x*r[2];    b[5]=-y*r[3]+x*r[5];    b[8]=-y*r[6]+x*r[8];

    // third givens rotation (ch,sh,0,0)
    gpuLaQRGivensQuaternion(b[4],b[5],ch3,sh3);
    x=1-2*sh3*sh3;
    y=2*ch3*sh3;

    // R is now set to desired value
    r[0]=b[0];              r[3]=b[3];              r[6]=b[6];
    r[1]=x*b[1]+y*b[2];     r[4]=x*b[4]+y*b[5];     r[7]=x*b[7]+y*b[8];
    r[2]=-y*b[1]+x*b[2];    r[5]=-y*b[4]+x*b[5];    r[8]=-y*b[7]+x*b[8];

    // construct the cumulative rotation Q=Q1 * Q2 * Q3
    // the number of floating point operations for three quaternion multiplications
    // is more or less comparable to the explicit form of the joined matrix.
    // certainly more memory-efficient!
    float sh12=sh1*sh1;
    float sh22=sh2*sh2;
    float sh32=sh3*sh3;

    q[0]=(-1+2*sh12)*(-1+2*sh22);
    q[3]=4*ch2*ch3*(-1+2*sh12)*sh2*sh3+2*ch1*sh1*(-1+2*sh32);
    q[6]=4*ch1*ch3*sh1*sh3-2*ch2*(-1+2*sh12)*sh2*(-1+2*sh32);

    q[1]=2*ch1*sh1*(1-2*sh22);
    q[4]=-8*ch1*ch2*ch3*sh1*sh2*sh3+(-1+2*sh12)*(-1+2*sh32);
    q[7]=-2*ch3*sh3+4*sh1*(ch3*sh1*sh3+ch1*ch2*sh2*(-1+2*sh32));

    q[2]=2*ch2*sh2;
    q[5]=2*ch3*(1-2*sh22)*sh3;
    q[8]=(-1+2*sh22)*(-1+2*sh32);
}

//=========================================================================================================
//================================================ SVD ====================================================
//=========================================================================================================

//EVERYTHING IN COLUMN MAJOR ORDER!

// a  - input matrix
// u  - left matrix u
// s  - singular values matrix
// vt - right matrix v transposed
__forceinline__ __device__ void gpuLaSVD(float* a, float* u, float* s, float* vt)
{
    //Help variables
    float help[9];
    float qV[4];

    gpuLaMultiplicationAtB(a, a, help);
    gpuLaJacobiEigenAnlysis(help,qV);
    gpuLaQuaternionToMatrix3(qV,vt);

    gpuLaMultiplicationAB(a, vt, help);
    gpuLaSortSingularValues(help,vt);

    gpuLaQRD(help,u,s);
}

// a  - input matrix
// r  - rotation matrix
__device__ void gpuRanSaCCalculateRotationMatrix(float* a, float* r)
{
    float u[9], s[9], vt[9];
    gpuLaSVD(a, u, s, vt);
    gpuLaMultiplicationAtBt(vt, u, r);
}

//=========================================================================================================
//============================================== MY TYPES =================================================
//=========================================================================================================

class MyFloat3: public float3
{
public:
    __forceinline__ __device__ MyFloat3()
    {
        x = 0;
        y = 0;
        z = 0;
    }
    __forceinline__ __device__ MyFloat3(const MyFloat3 &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }
    __forceinline__ __device__ MyFloat3(float x_, float y_, float z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }
    __forceinline__ __device__ MyFloat3(const gpu::Feature::PointType &other)
    {
        x = other.data[0];
        y = other.data[1];
        z = other.data[2];
    }
    __forceinline__ __device__ MyFloat3& operator=(const MyFloat3 &other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }
    __forceinline__ __device__ MyFloat3& operator+=(const MyFloat3 &other)
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    __forceinline__ __device__ MyFloat3& operator-=(const MyFloat3 &other)
    {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }
    __forceinline__ __device__ MyFloat3& operator/=(const float &other)
    {
        x /= other;
        y /= other;
        z /= other;
        return *this;
    }
    __forceinline__ __device__ float length()
    {
        return ((float)sqrt(this->x*this->x + this->y*this->y + this->z*this->z));
    }
};

__forceinline__ __device__ MyFloat3 operator+(MyFloat3 this_one, const MyFloat3 &other)
{
    this_one += other;
    return this_one;
}
__forceinline__ __device__ MyFloat3 operator-(MyFloat3 this_one, const MyFloat3 &other)
{
    this_one -= other;
    return this_one;
}
__forceinline__ __device__ MyFloat3 operator/(MyFloat3 this_one, const int &other)
{
    this_one /= other;
    return this_one;
}

//=========================================================================================================
//=========================================== SCAN & COMPACT ==============================================
//=========================================================================================================

/**
 *  @brief      gpuInclusiveSumScan     Perform invlusive sum scan of the input array. If you want to perform segmented scan
 *                                      (input is bigger than size of a single block) just pass non-zero pointer to auxiliary
 *                                      array (of size equals to blocks/2).
 *  @param[in]  input                   Input vector to be scanned.
 *  @param[in]  input_length            Size of input vector.
 *  @param[out] output                  Output scanned array.
 *  @param[out] aux                     Auxiliary output array (for segmented scan).
 */
__global__ void gpuInclusiveSumScan(int* input, int input_length, int* output, int* aux=0)
{
    // Load a segment of the input vector into shared memory
    extern __shared__ int scan_array[];

    // Help variables
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // Copy first values
    if (start + t < input_length)
       scan_array[t] = input[start + t];
    else
       scan_array[t] = 0;

    // Copy second values
    if (start + blockDim.x + t < input_length)
       scan_array[blockDim.x + t] = input[start + blockDim.x + t];
    else
       scan_array[blockDim.x + t] = 0;

    // Barier
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= blockDim.x; stride <<= 1)
    {
       int index = (t + 1) * stride * 2 - 1;
       if (index < 2 * blockDim.x)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    // Post reduction
    for (stride = blockDim.x >> 1; stride; stride >>= 1)
    {
       int index = (t + 1) * stride * 2 - 1;
       if (index + stride < 2 * blockDim.x)
          scan_array[index + stride] += scan_array[index];
       __syncthreads();
    }

    //Copy first output
    if (start + t < input_length)
       output[start + t] = scan_array[t];

    //Copy second output
    if (start + blockDim.x + t < input_length)
       output[start + blockDim.x + t] = scan_array[blockDim.x + t];

    //Set aux
    if (aux && t == 0)
       aux[blockIdx.x] = scan_array[2 * blockDim.x - 1];
}

/**
 *  @brief      gpuExclisiveSumScan     Perform invlusive sum scan of the input array. If you want to perform segmented scan
 *                                      (input is bigger than size of a single block) just pass non-zero pointer to auxiliary
 *                                      array (of size equals to blocks/2).
 *  @param[in]  input                   Input vector to be scanned.
 *  @param[in]  input_length            Size of input vector.
 *  @param[out] output                  Output scanned array.
 *  @param[out] aux                     Auxiliary output array (for segmented scan).
 */
__global__ void gpuExclusiveSumScan(int* input, int input_length, int* output, int* aux=0)
{
    // Load a segment of the input vector into shared memory
    extern __shared__ int scan_array[];

    // Help variables
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // Copy first values
    if (start + t < input_length)
       scan_array[t] = input[start + t];
    else
       scan_array[t] = 0;

    // Copy second values
    if (start + blockDim.x + t < input_length)
       scan_array[blockDim.x + t] = input[start + blockDim.x + t];
    else
       scan_array[blockDim.x + t] = 0;

    // Barier
    __syncthreads();

    // Reduction
    int stride;
    for (stride = 1; stride <= blockDim.x; stride <<= 1)
    {
       int index = (t + 1) * stride * 2 - 1;
       if (index < 2 * blockDim.x)
          scan_array[index] += scan_array[index - stride];
       __syncthreads();
    }

    //Zero for exclusive scan
    if (t==0) scan_array[2*blockDim.x-1] = 0;

    // Post reduction
    for (stride = blockDim.x; stride; stride >>= 1)
    {
       int index = (t + 1) * stride * 2 - 1;
       if (index < 2 * blockDim.x)
       {
           int help = scan_array[index];
           scan_array[index] += scan_array[index - stride];
           scan_array[index-stride] = help;
       }
       __syncthreads();
    }

    //Copy first output
    if (start + t < input_length)
       output[start + t] = scan_array[t];

    //Copy second output
    if (start + blockDim.x + t < input_length)
       output[start + blockDim.x + t] = scan_array[blockDim.x + t];

    //Set aux
    if (aux && t == 0)
       aux[blockIdx.x] = scan_array[2 * blockDim.x - 1];
}

/**
 *  @brief          gpuScanFixup        If you use a segmented scan you have scanned array (input) only in block scope and auxiliary
 *                                      array of scanned blocks. This method is to fix this and as a result you will have fully
 *                                      scanned input without segments.
 *  @param[in,out]  input               Input initially scanned array (only in blocks scope).
 *  @param[in]      input_length        Length of input array.
 *  @param[in]      aux                 Auxiliary array of scanned blocks value.
 */
__global__ void gpuScanFixup(int *input, int input_length, int *aux)
{
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    if (blockIdx.x)
    {
        if (start + t < input_length)
            input[start + t] += aux[blockIdx.x];

        if (start + blockDim.x + t < input_length)
            input[start + blockDim.x + t] += aux[blockIdx.x];
    }
}

/**
 *  @brief      gpuCompact              Compact of sparse array into danse array via sparce_valid and sparse_addresses.
 *  @param[in]  sparse_transformations  Input array of sparse transformations.
 *  @param[in]  sparse_length           Length of sparse transformations.
 *  @param[in]  sparse_valid            Array containing flags of valid transformation (1) or non valid (0).
 *  @param[in]  sparse_addresses        Addresses for densing the sparse data.
 *  @param[out] dense_transformations   Output danse transformations array.
 */
__global__ void gpuCompact(float* sparse_transformations, int sparse_length,
                           int* sparse_valid, int* sparse_addresses,
                           float* dense_transformations)
{
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    if (hypothesis_index<sparse_length && sparse_valid[hypothesis_index])
    {
        int address = 16*sparse_addresses[hypothesis_index];
        dense_transformations[address + 0] =  sparse_transformations[16*hypothesis_index + 0];
        dense_transformations[address + 1] =  sparse_transformations[16*hypothesis_index + 1];
        dense_transformations[address + 2] =  sparse_transformations[16*hypothesis_index + 2];
        dense_transformations[address + 3] =  sparse_transformations[16*hypothesis_index + 3];
        dense_transformations[address + 4] =  sparse_transformations[16*hypothesis_index + 4];
        dense_transformations[address + 5] =  sparse_transformations[16*hypothesis_index + 5];
        dense_transformations[address + 6] =  sparse_transformations[16*hypothesis_index + 6];
        dense_transformations[address + 7] =  sparse_transformations[16*hypothesis_index + 7];
        dense_transformations[address + 8] =  sparse_transformations[16*hypothesis_index + 8];
        dense_transformations[address + 9] =  sparse_transformations[16*hypothesis_index + 9];
        dense_transformations[address + 10] = sparse_transformations[16*hypothesis_index + 10];
        dense_transformations[address + 11] = sparse_transformations[16*hypothesis_index + 11];
        dense_transformations[address + 12] = sparse_transformations[16*hypothesis_index + 12];
        dense_transformations[address + 13] = sparse_transformations[16*hypothesis_index + 13];
        dense_transformations[address + 14] = sparse_transformations[16*hypothesis_index + 14];
        dense_transformations[address + 15] = sparse_transformations[16*hypothesis_index + 15];
    }
}

__global__ void gpuCompactA(int* sparse_transformations, int sparse_length,
                           int* sparse_valid, int* sparse_addresses,
                           int* dense_transformations)
{
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    if (hypothesis_index<sparse_length && sparse_valid[hypothesis_index])
    {
        int address = 3*sparse_addresses[hypothesis_index];
        dense_transformations[address + 0] =  sparse_transformations[3*hypothesis_index + 0];
        dense_transformations[address + 1] =  sparse_transformations[3*hypothesis_index + 1];
        dense_transformations[address + 2] =  sparse_transformations[3*hypothesis_index + 2];
    }
}

/**
 *  @brief      gpuCompactAtomic        Compact of sparse array into very dense array (using atomics).
 *  @param[in]  sparse_transformations  Input array of sparse transformations.
 *  @param[in]  sparse_length           Length of sparse transformations.
 *  @param[in]  sparse_valid            Array containing flags of valid transformation (1) or non valid (0).
 *  @param[in]  sparse_actial_address   Addresses for densing the sparse data (for atomic operations).
 *  @param[out] dense_transformations   Output danse transformations array.
 */
__global__ void gpuCompactAtomic(float* sparse_transformations, int sparse_length,
                                 int* sparse_valid, int* sparse_actual_address,
                                 float* dense_transformations)
{
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    if (hypothesis_index<sparse_length && sparse_valid[hypothesis_index])
    {
        int address = atomicAdd(sparse_actual_address,16);
        dense_transformations[address + 0] =  sparse_transformations[16*hypothesis_index + 0];
        dense_transformations[address + 1] =  sparse_transformations[16*hypothesis_index + 1];
        dense_transformations[address + 2] =  sparse_transformations[16*hypothesis_index + 2];
        dense_transformations[address + 3] =  sparse_transformations[16*hypothesis_index + 3];
        dense_transformations[address + 4] =  sparse_transformations[16*hypothesis_index + 4];
        dense_transformations[address + 5] =  sparse_transformations[16*hypothesis_index + 5];
        dense_transformations[address + 6] =  sparse_transformations[16*hypothesis_index + 6];
        dense_transformations[address + 7] =  sparse_transformations[16*hypothesis_index + 7];
        dense_transformations[address + 8] =  sparse_transformations[16*hypothesis_index + 8];
        dense_transformations[address + 9] =  sparse_transformations[16*hypothesis_index + 9];
        dense_transformations[address + 10] = sparse_transformations[16*hypothesis_index + 10];
        dense_transformations[address + 11] = sparse_transformations[16*hypothesis_index + 11];
        dense_transformations[address + 12] = sparse_transformations[16*hypothesis_index + 12];
        dense_transformations[address + 13] = sparse_transformations[16*hypothesis_index + 13];
        dense_transformations[address + 14] = sparse_transformations[16*hypothesis_index + 14];
        dense_transformations[address + 15] = sparse_transformations[16*hypothesis_index + 15];
    }
}

//=========================================================================================================
//============================================= GPU RANSAC ================================================
//=========================================================================================================

/**
 *  @brief      gpuGenerateTransformations  Generate transformations routine.
 *  @param[in]  source_point_cloud          Source point cloud.
 *  @param[in]  source_neighbors            Vector of neighbors for every source point in feature space expressed as target cloud index.
 *  @param[in]  source_points               Number of source points.
 *  @param[in]  seed                        The seed used for curand random numbers generator - you can use time(0) as parameter.
 *  @param[in]  target_point_cloud          Target point cloud.
 *  @param[out] transformations             Vector of estimated transformations (matrices stored in column major order).
 *  @param[out] transformations_valid       Vector of flags if transformation at this point is valid (1) or not (0).
 */
__global__ void gpuGenerateTransformations(gpu::Feature::PointType* source_point_cloud, int *source_neighbors, int source_points, unsigned long seed,
                                           gpu::Feature::PointType* target_point_cloud,
                                           float *transformations, int* transformations_valid, float ransac_poly)
{
    //Init variables
    curandState states[3];
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    //Initialize random generator
    curand_init(seed, (hypothesis_index+1)*1, 0, states+0);
    curand_init(seed, (hypothesis_index+1)*2, 0, states+1);
    curand_init(seed, (hypothesis_index+1)*3, 0, states+2);

    //Generate 3 random numbers as source points
    int source_index_0 = curand_uniform(states+0)*source_points - 1;
    int source_index_1 = curand_uniform(states+1)*source_points - 1;
    int source_index_2 = curand_uniform(states+2)*source_points - 1;

    //Copy points to local values
    MyFloat3 p_0 = source_point_cloud[source_index_0];
    MyFloat3 p_1 = source_point_cloud[source_index_1];
    MyFloat3 p_2 = source_point_cloud[source_index_2];
    MyFloat3 q_0 = target_point_cloud[source_neighbors[source_index_0]];
    MyFloat3 q_1 = target_point_cloud[source_neighbors[source_index_1]];
    MyFloat3 q_2 = target_point_cloud[source_neighbors[source_index_2]];

    // Prerejector poly
    float pDistance[3];
    pDistance[0] = (p_1 - p_0).length();
    pDistance[1] = (p_2 - p_1).length();
    pDistance[2] = (p_0 - p_2).length();

    float qDistance[3];
    qDistance[0] = (q_1 - q_0).length();
    qDistance[1] = (q_2 - q_1).length();
    qDistance[2] = (q_0 - q_2).length();

    float delta[3];
    delta[0] = gpuMathAbs(pDistance[0] - qDistance[0]) / gpuMathMax(pDistance[0],qDistance[0]);
    delta[1] = gpuMathAbs(pDistance[1] - qDistance[1]) / gpuMathMax(pDistance[1],qDistance[1]);
    delta[2] = gpuMathAbs(pDistance[2] - qDistance[2]) / gpuMathMax(pDistance[2],qDistance[2]);

    if (delta[0] > ransac_poly || delta[1] > ransac_poly || delta[2] > ransac_poly)
        return;

    //Calculate avarage
    MyFloat3 p_av = (p_0 + p_1 + p_2)/3;
    MyFloat3 q_av = (q_0 + q_1 + q_2)/3;

    //Move to centorid
    p_0 -= p_av;
    p_1 -= p_av;
    p_2 -= p_av;
    q_0 -= q_av;
    q_1 -= q_av;
    q_2 -= q_av;

    //Calculate rotation
    float r[9];
    float a[9] = { p_0.x*q_0.x + p_1.x*q_1.x + p_2.x*q_2.x,
                   p_0.x*q_0.y + p_1.x*q_1.y + p_2.x*q_2.y,
                   p_0.x*q_0.z + p_1.x*q_1.z + p_2.x*q_2.z,
                   p_0.y*q_0.x + p_1.y*q_1.x + p_2.y*q_2.x,
                   p_0.y*q_0.y + p_1.y*q_1.y + p_2.y*q_2.y,
                   p_0.y*q_0.z + p_1.y*q_1.z + p_2.y*q_2.z,
                   p_0.z*q_0.x + p_1.z*q_1.x + p_2.z*q_2.x,
                   p_0.z*q_0.y + p_1.z*q_1.y + p_2.z*q_2.y,
                   p_0.z*q_0.z + p_1.z*q_1.z + p_2.z*q_2.z };
    gpuRanSaCCalculateRotationMatrix(a, r);

    //Calculate translation
    float t[3];
    t[0] = q_av.x - (r[0]*p_av.x + r[3]*p_av.y + r[6]*p_av.z);
    t[1] = q_av.y - (r[1]*p_av.x + r[4]*p_av.y + r[7]*p_av.z);
    t[2] = q_av.z - (r[2]*p_av.x + r[5]*p_av.y + r[8]*p_av.z);

    //Fill the data
    transformations_valid[hypothesis_index] = 1;
    transformations[16*hypothesis_index+0] = r[0];
    transformations[16*hypothesis_index+1] = r[1];
    transformations[16*hypothesis_index+2] = r[2];
    transformations[16*hypothesis_index+3] = 0;
    transformations[16*hypothesis_index+4] = r[3];
    transformations[16*hypothesis_index+5] = r[4];
    transformations[16*hypothesis_index+6] = r[5];
    transformations[16*hypothesis_index+7] = 0;
    transformations[16*hypothesis_index+8] = r[6];
    transformations[16*hypothesis_index+9] = r[7];
    transformations[16*hypothesis_index+10] = r[8];
    transformations[16*hypothesis_index+11] = 0;
    transformations[16*hypothesis_index+12] = t[0];
    transformations[16*hypothesis_index+13] = t[1];
    transformations[16*hypothesis_index+14] = t[2];
    transformations[16*hypothesis_index+15] = 0;
}
__global__ void gpuGenerateTransformationsC(gpu::Feature::PointType* source_point_cloud, int *source_neighbors, int source_points, unsigned long seed,
                                           gpu::Feature::PointType* target_point_cloud,
                                           float *transformations, int* transformations_valid, float ransac_poly)
{
    //Init variables
    curandState states[3];
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    //Initialize random generator
    curand_init(seed, (hypothesis_index+1)*1, 0, states+0);
    curand_init(seed, (hypothesis_index+1)*2, 0, states+1);
    curand_init(seed, (hypothesis_index+1)*3, 0, states+2);

    //Generate 3 random numbers as source points
    int source_index_0 = curand_uniform(states+0)*source_points - 1;
    int source_index_1 = curand_uniform(states+1)*source_points - 1;
    int source_index_2 = curand_uniform(states+2)*source_points - 1;

//    //Copy points to local values
//    MyFloat3 p_0 = source_point_cloud[source_index_0];
//    MyFloat3 p_1 = source_point_cloud[source_index_1];
//    MyFloat3 p_2 = source_point_cloud[source_index_2];
//    MyFloat3 q_0 = target_point_cloud[source_neighbors[source_index_0]];
//    MyFloat3 q_1 = target_point_cloud[source_neighbors[source_index_1]];
//    MyFloat3 q_2 = target_point_cloud[source_neighbors[source_index_2]];

//    // Prerejector poly
//    float pDistance[3];
//    pDistance[0] = (p_1 - p_0).length();
//    pDistance[1] = (p_2 - p_1).length();
//    pDistance[2] = (p_0 - p_2).length();

//    float qDistance[3];
//    qDistance[0] = (q_1 - q_0).length();
//    qDistance[1] = (q_2 - q_1).length();
//    qDistance[2] = (q_0 - q_2).length();

//    float delta[3];
//    delta[0] = gpuMathAbs(pDistance[0] - qDistance[0]) / gpuMathMax(pDistance[0],qDistance[0]);
//    delta[1] = gpuMathAbs(pDistance[1] - qDistance[1]) / gpuMathMax(pDistance[1],qDistance[1]);
//    delta[2] = gpuMathAbs(pDistance[2] - qDistance[2]) / gpuMathMax(pDistance[2],qDistance[2]);

//    if (delta[0] > ransac_poly || delta[1] > ransac_poly || delta[2] > ransac_poly)
//        return;

//    //Calculate avarage
//    MyFloat3 p_av = (p_0 + p_1 + p_2)/3;
//    MyFloat3 q_av = (q_0 + q_1 + q_2)/3;

//    //Move to centorid
//    p_0 -= p_av;
//    p_1 -= p_av;
//    p_2 -= p_av;
//    q_0 -= q_av;
//    q_1 -= q_av;
//    q_2 -= q_av;

//    //Calculate rotation
//    float r[9];
//    float a[9] = { p_0.x*q_0.x + p_1.x*q_1.x + p_2.x*q_2.x,
//                   p_0.x*q_0.y + p_1.x*q_1.y + p_2.x*q_2.y,
//                   p_0.x*q_0.z + p_1.x*q_1.z + p_2.x*q_2.z,
//                   p_0.y*q_0.x + p_1.y*q_1.x + p_2.y*q_2.x,
//                   p_0.y*q_0.y + p_1.y*q_1.y + p_2.y*q_2.y,
//                   p_0.y*q_0.z + p_1.y*q_1.z + p_2.y*q_2.z,
//                   p_0.z*q_0.x + p_1.z*q_1.x + p_2.z*q_2.x,
//                   p_0.z*q_0.y + p_1.z*q_1.y + p_2.z*q_2.y,
//                   p_0.z*q_0.z + p_1.z*q_1.z + p_2.z*q_2.z };
//    gpuRanSaCCalculateRotationMatrix(a, r);

//    //Calculate translation
//    float t[3];
//    t[0] = q_av.x - (r[0]*p_av.x + r[3]*p_av.y + r[6]*p_av.z);
//    t[1] = q_av.y - (r[1]*p_av.x + r[4]*p_av.y + r[7]*p_av.z);
//    t[2] = q_av.z - (r[2]*p_av.x + r[5]*p_av.y + r[8]*p_av.z);

//    //Fill the data
//    transformations_valid[hypothesis_index] = 1;
//    transformations[16*hypothesis_index+0] = r[0];
//    transformations[16*hypothesis_index+1] = r[1];
//    transformations[16*hypothesis_index+2] = r[2];
//    transformations[16*hypothesis_index+3] = 0;
//    transformations[16*hypothesis_index+4] = r[3];
//    transformations[16*hypothesis_index+5] = r[4];
//    transformations[16*hypothesis_index+6] = r[5];
//    transformations[16*hypothesis_index+7] = 0;
//    transformations[16*hypothesis_index+8] = r[6];
//    transformations[16*hypothesis_index+9] = r[7];
//    transformations[16*hypothesis_index+10] = r[8];
//    transformations[16*hypothesis_index+11] = 0;
//    transformations[16*hypothesis_index+12] = t[0];
//    transformations[16*hypothesis_index+13] = t[1];
//    transformations[16*hypothesis_index+14] = t[2];
//    transformations[16*hypothesis_index+15] = 0;
}

__global__ void gpuPerfomPolyTest(gpu::Feature::PointType* source_point_cloud, int *source_neighbors, int *source_indices,
                                  gpu::Feature::PointType* target_point_cloud,
                                  int* transformations_valid, float ransac_poly)
{
    //Init variables
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    //Copy points to local values
    MyFloat3 p_0 = source_point_cloud[source_indices[3*hypothesis_index+0]];
    MyFloat3 p_1 = source_point_cloud[source_indices[3*hypothesis_index+1]];
    MyFloat3 p_2 = source_point_cloud[source_indices[3*hypothesis_index+2]];
    MyFloat3 q_0 = target_point_cloud[source_neighbors[source_indices[3*hypothesis_index+0]]];
    MyFloat3 q_1 = target_point_cloud[source_neighbors[source_indices[3*hypothesis_index+1]]];
    MyFloat3 q_2 = target_point_cloud[source_neighbors[source_indices[3*hypothesis_index+2]]];

    // Prerejector poly
    float pDistance[3];
    pDistance[0] = (p_1 - p_0).length();
    pDistance[1] = (p_2 - p_1).length();
    pDistance[2] = (p_0 - p_2).length();

    float qDistance[3];
    qDistance[0] = (q_1 - q_0).length();
    qDistance[1] = (q_2 - q_1).length();
    qDistance[2] = (q_0 - q_2).length();

    float delta[3];
    delta[0] = gpuMathAbs(pDistance[0] - qDistance[0]) / gpuMathMax(pDistance[0],qDistance[0]);
    delta[1] = gpuMathAbs(pDistance[1] - qDistance[1]) / gpuMathMax(pDistance[1],qDistance[1]);
    delta[2] = gpuMathAbs(pDistance[2] - qDistance[2]) / gpuMathMax(pDistance[2],qDistance[2]);

    // Is valid?
    if (delta[0] <= ransac_poly && delta[1] <= ransac_poly && delta[2] <= ransac_poly)
        transformations_valid[hypothesis_index] = 1;
}
__global__ void gpuGenerateTransformationsB(gpu::Feature::PointType* source_point_cloud, int *source_neighbors, int *source_indices,
                                            gpu::Feature::PointType* target_point_cloud,
                                            float *transformations)
{
    //Init variables
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    //Copy points to local values
    MyFloat3 p_0 = source_point_cloud[source_indices[3*hypothesis_index+0]];
    MyFloat3 p_1 = source_point_cloud[source_indices[3*hypothesis_index+1]];
    MyFloat3 p_2 = source_point_cloud[source_indices[3*hypothesis_index+2]];
    MyFloat3 q_0 = target_point_cloud[source_neighbors[source_indices[3*hypothesis_index+0]]];
    MyFloat3 q_1 = target_point_cloud[source_neighbors[source_indices[3*hypothesis_index+1]]];
    MyFloat3 q_2 = target_point_cloud[source_neighbors[source_indices[3*hypothesis_index+2]]];


    //Calculate avarage
    MyFloat3 p_av = (p_0 + p_1 + p_2)/3;
    MyFloat3 q_av = (q_0 + q_1 + q_2)/3;

    //Move to centorid
    p_0 -= p_av;
    p_1 -= p_av;
    p_2 -= p_av;
    q_0 -= q_av;
    q_1 -= q_av;
    q_2 -= q_av;

    //Calculate rotation
    float r[9];
    float a[9] = { p_0.x*q_0.x + p_1.x*q_1.x + p_2.x*q_2.x,
                   p_0.x*q_0.y + p_1.x*q_1.y + p_2.x*q_2.y,
                   p_0.x*q_0.z + p_1.x*q_1.z + p_2.x*q_2.z,
                   p_0.y*q_0.x + p_1.y*q_1.x + p_2.y*q_2.x,
                   p_0.y*q_0.y + p_1.y*q_1.y + p_2.y*q_2.y,
                   p_0.y*q_0.z + p_1.y*q_1.z + p_2.y*q_2.z,
                   p_0.z*q_0.x + p_1.z*q_1.x + p_2.z*q_2.x,
                   p_0.z*q_0.y + p_1.z*q_1.y + p_2.z*q_2.y,
                   p_0.z*q_0.z + p_1.z*q_1.z + p_2.z*q_2.z };
    gpuRanSaCCalculateRotationMatrix(a, r);

    //Calculate translation
    float t[3];
    t[0] = q_av.x - (r[0]*p_av.x + r[3]*p_av.y + r[6]*p_av.z);
    t[1] = q_av.y - (r[1]*p_av.x + r[4]*p_av.y + r[7]*p_av.z);
    t[2] = q_av.z - (r[2]*p_av.x + r[5]*p_av.y + r[8]*p_av.z);

    //Fill the data
    transformations[16*hypothesis_index+0] = r[0];
    transformations[16*hypothesis_index+1] = r[1];
    transformations[16*hypothesis_index+2] = r[2];
    transformations[16*hypothesis_index+3] = 0;
    transformations[16*hypothesis_index+4] = r[3];
    transformations[16*hypothesis_index+5] = r[4];
    transformations[16*hypothesis_index+6] = r[5];
    transformations[16*hypothesis_index+7] = 0;
    transformations[16*hypothesis_index+8] = r[6];
    transformations[16*hypothesis_index+9] = r[7];
    transformations[16*hypothesis_index+10] = r[8];
    transformations[16*hypothesis_index+11] = 0;
    transformations[16*hypothesis_index+12] = t[0];
    transformations[16*hypothesis_index+13] = t[1];
    transformations[16*hypothesis_index+14] = t[2];
    transformations[16*hypothesis_index+15] = 0;
}

/**
 *  @brief      gpuGenerateTransformedPoints    Generate transformed points for futher tests.
 *  @param[in]  source_point_cloud              Source point cloud.
 *  @param[in]  source_points                   Number of source points.
 *  @param[in]  seed                            The seed used for curand random numbers generator - you can use time(0) as parameter.
 *  @param[in]  transformations                 Vector of estimated transformations (matrices stored in column major order).
 *  @param[in]  transformations_number          Number of transformations.
 *  @param[in]  points_per_transformation       Number of points generated per hypothesis.
 *  @param[out] transformed_point_cloud         Transformed point cloud for futher tests.
 */
__global__ void gpuGenerateTransformedPoints(gpu::Feature::PointType* source_point_cloud, int source_points, unsigned long seed,
                                             float* transformations, int transformations_number, int points_per_transformation,
                                             gpu::Feature::PointType* transformed_point_cloud)
{
    //Help variables
    int global_thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int hypothesis_index = (global_thread_idx)/points_per_transformation;

    //Generate random point index of source cloud
    curandState state;
    curand_init(seed, global_thread_idx, 0, &state);
    int source_point_index = curand_uniform(&state)*source_points - 1;

    //Transform
    if (hypothesis_index < transformations_number)
    {
        transformed_point_cloud[global_thread_idx].data[0] = transformations[16*hypothesis_index+0]*source_point_cloud[source_point_index].data[0] +
                                                             transformations[16*hypothesis_index+4]*source_point_cloud[source_point_index].data[1] +
                                                             transformations[16*hypothesis_index+8]*source_point_cloud[source_point_index].data[2] +
                                                             transformations[16*hypothesis_index+12];
        transformed_point_cloud[global_thread_idx].data[1] = transformations[16*hypothesis_index+1]*source_point_cloud[source_point_index].data[0] +
                                                             transformations[16*hypothesis_index+5]*source_point_cloud[source_point_index].data[1] +
                                                             transformations[16*hypothesis_index+9]*source_point_cloud[source_point_index].data[2] +
                                                             transformations[16*hypothesis_index+13];
        transformed_point_cloud[global_thread_idx].data[2] = transformations[16*hypothesis_index+2]*source_point_cloud[source_point_index].data[0] +
                                                             transformations[16*hypothesis_index+6]*source_point_cloud[source_point_index].data[1] +
                                                             transformations[16*hypothesis_index+10]*source_point_cloud[source_point_index].data[2] +
                                                             transformations[16*hypothesis_index+14];
    }
}
__global__ void gpuGenerateTransformedPoints2(gpu::Feature::PointType* source_point_cloud, int source_points, int* source_indices,
                                             float* transformations, int transformations_number, int points_per_transformation,
                                             gpu::Feature::PointType* transformed_point_cloud)
{
    //Help variables
    int global_thread_idx = threadIdx.x + blockIdx.x*blockDim.x;
    int hypothesis_index = (global_thread_idx)/points_per_transformation;
    int source_point_index = source_indices[global_thread_idx];

    //Transform
    if (hypothesis_index < transformations_number)
    {
        transformed_point_cloud[global_thread_idx].data[0] = transformations[16*hypothesis_index+0]*source_point_cloud[source_point_index].data[0] +
                                                             transformations[16*hypothesis_index+4]*source_point_cloud[source_point_index].data[1] +
                                                             transformations[16*hypothesis_index+8]*source_point_cloud[source_point_index].data[2] +
                                                             transformations[16*hypothesis_index+12];
        transformed_point_cloud[global_thread_idx].data[1] = transformations[16*hypothesis_index+1]*source_point_cloud[source_point_index].data[0] +
                                                             transformations[16*hypothesis_index+5]*source_point_cloud[source_point_index].data[1] +
                                                             transformations[16*hypothesis_index+9]*source_point_cloud[source_point_index].data[2] +
                                                             transformations[16*hypothesis_index+13];
        transformed_point_cloud[global_thread_idx].data[2] = transformations[16*hypothesis_index+2]*source_point_cloud[source_point_index].data[0] +
                                                             transformations[16*hypothesis_index+6]*source_point_cloud[source_point_index].data[1] +
                                                             transformations[16*hypothesis_index+10]*source_point_cloud[source_point_index].data[2] +
                                                             transformations[16*hypothesis_index+14];
    }
}

/**
 *  @brief      gpuTddTest                  Perform Tdd test.
 *  @param[in]  neighbors                   How many points are in the neigborhood of every point?
 *  @param[in]  threshold                   Ransac threshold.
 *  @param[in]  transformations_count       How many transformations do we check?
 *  @param[out] transformations_valid       Flag for every transformations if it is valid or not.
 *  @param[out] transformations_valid_count How many valid transformations?
 */
__global__ void gpuTddTest(int* neighbors, float threshold, int transformations_count, int* transformations_valid, int* transformtions_valid_count)
{
    //Help variables
    int hypothesis_index = threadIdx.x + blockIdx.x*blockDim.x;

    //If in ranges
    if (hypothesis_index<transformations_count)
    {
        int valid = 0;
        for (int i=0; i<32; i++)
            if (neighbors[32*hypothesis_index+i]) valid++;

        if ((float)valid/32 > threshold)
        {
            atomicAdd(transformtions_valid_count,1);
            transformations_valid[hypothesis_index] = 1;
        }
    }
}

__global__ void gpuFinalTest(int* neighbors, float threshold, int source_points,
                             int* transformation_inliers)
{
    //Shared memory declaration
    __shared__ int valid_sum[1024];

    //Some help variables
    int hypothesis_index = blockIdx.x;
    int point_index = threadIdx.x;
    unsigned int loop_limit = ceil((float)source_points/gridDim.x);

    //Share memory initialization
    valid_sum[threadIdx.x] = 0;

    //Loop after every 1024 points
    for (unsigned int loop_idx=0; loop_idx<loop_limit; loop_idx++)
    {
        //Calculate current source index
        int source_point_index = loop_idx*blockDim.x+point_index;

        //If target point index is not in bounds break the lopp
        if (source_point_index>=source_points) break;

        //Add
        if (neighbors[hypothesis_index*source_points+source_point_index])
            valid_sum[point_index] ++;
    }

    //Barier
    __syncthreads();

    //Reduce
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        //Add
        if (point_index < s)
            valid_sum[point_index] += valid_sum[point_index+s];

        //Barier
        __syncthreads();
    }

    //Save result
    if (point_index==0 && valid_sum[point_index]>threshold)
        transformation_inliers[hypothesis_index] = valid_sum[point_index];
}


//=========================================================================================================
//============================================ RANSAC INTERFACE ===========================================
//=========================================================================================================

void RegistrationUtilsGPU::performRanSaC(gpu::Feature::PointType* d_source_point_cloud, int source_points, int* d_source_neighbors,
                                         gpu::Feature::PointType* d_target_point_cloud, gpu::Octree &d_target_octree,
                                         float* transformation_matrix)
{

    #define NEW     1

    //========================================================================================
    //============================== Initial ransac parameters ===============================
    //========================================================================================

    if (ransac_radius<0 || ransac_threshold<0 || ransac_poly<0)
    {
        std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Ransac parameters not set up!" << "\x1B[0m" << std::endl;
        return;
    }

    boost::chrono::high_resolution_clock::time_point time_start = boost::chrono::high_resolution_clock::now();


#if NEW

    //========================================================================================
    //=============================== GENERATE RANDOM INDICES ================================
    //========================================================================================

    // Rand indices
    std::srand(std::time(0));
    int source_indices[3*BLOCKS*THREADS];
    for (int i=0; i<3*BLOCKS*THREADS; ++i) source_indices[i] = std::rand() % source_points;

    //========================================================================================
    //===================================== CHECK POLY =======================================
    //========================================================================================

    int *d_transformations_sparse_valid;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_sparse_valid, BLOCKS*THREADS*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformations_sparse_valid, 0, BLOCKS*THREADS*sizeof(int)));

    int *d_source_indices;
    cudaSafeCall(cudaMalloc((void**)&d_source_indices, 3*BLOCKS*THREADS*sizeof(int)));
    cudaSafeCall(cudaMemcpy((void*)d_source_indices, (void*)source_indices, 3*BLOCKS*THREADS*sizeof(int), cudaMemcpyHostToDevice));

    gpuPerfomPolyTest<<<BLOCKS,THREADS>>>(d_source_point_cloud, d_source_neighbors, d_source_indices,
                                          d_target_point_cloud,
                                          d_transformations_sparse_valid, ransac_poly);
    cudaSafeCall(cudaDeviceSynchronize());
    boost::chrono::high_resolution_clock::time_point time_generation = boost::chrono::high_resolution_clock::now();

    //========================================================================================
    //=================================== COMPACT INDICES ====================================
    //========================================================================================

    int *d_addresses;
    cudaSafeCall(cudaMalloc((void**)&d_addresses, BLOCKS*THREADS*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_addresses, 0, BLOCKS*THREADS*sizeof(int)));

    int *d_addresses_aux;
    cudaSafeCall(cudaMalloc((void**)&d_addresses_aux, BLOCKS/2*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_addresses_aux, 0, BLOCKS/2*sizeof(int)));

    int *d_addresses_aux_scanned;
    cudaSafeCall(cudaMalloc((void**)&d_addresses_aux_scanned, BLOCKS/2*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_addresses_aux_scanned, 0, BLOCKS/2*sizeof(int)));

    int numBlocks = ceil((float)BLOCKS*THREADS/(THREADS<<1));
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(THREADS, 1, 1);

    gpuExclusiveSumScan<<<dimGrid,dimBlock,THREADS*2*sizeof(int)>>>(d_transformations_sparse_valid, BLOCKS*THREADS, d_addresses, d_addresses_aux);
    gpuExclusiveSumScan<<<dim3(1,1,1), dimGrid,BLOCKS*sizeof(int)>>>(d_addresses_aux, BLOCKS/2, d_addresses_aux_scanned);
    gpuScanFixup<<<dimGrid,dimBlock>>>(d_addresses, BLOCKS*THREADS, d_addresses_aux_scanned);
    cudaSafeCall(cudaDeviceSynchronize());

    int h_transformations_dense_count;
    cudaSafeCall(cudaMemcpy((void*)&h_transformations_dense_count, (void*)&(d_addresses[BLOCKS*THREADS-1]), sizeof(int), cudaMemcpyDeviceToHost));

    int *d_source_indices_dense;
    cudaSafeCall(cudaMalloc((void**)&d_source_indices_dense, 3*h_transformations_dense_count*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_source_indices_dense, 0, 3*h_transformations_dense_count*sizeof(int)));

    gpuCompactA<<<BLOCKS,THREADS>>>(d_source_indices, BLOCKS*THREADS,
                                    d_transformations_sparse_valid, d_addresses,
                                    d_source_indices_dense);
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaFree((void*)d_source_indices));
    cudaSafeCall(cudaFree((void*)d_transformations_sparse_valid));
    cudaSafeCall(cudaFree((void*)d_addresses));
    cudaSafeCall(cudaFree((void*)d_addresses_aux));
    cudaSafeCall(cudaFree((void*)d_addresses_aux_scanned));

    //========================================================================================
    //=============================== GENERATE TRANSFORMATIONS ===============================
    //========================================================================================

    if (h_transformations_dense_count>1024)
    {
        std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] To meny results! Please make poly threshold heigher (lower value)!" << "ms\x1B[0m" << std::endl;
        return;
    }

    float* d_transformations_dense;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_dense, 16*h_transformations_dense_count*sizeof(float)));
    cudaSafeCall(cudaMemset((void*)d_transformations_dense, 0, 16*h_transformations_dense_count*sizeof(float)));

    int generate_threads = 128;
    int generate_blocks = ceil(float(h_transformations_dense_count)/generate_threads);
    gpuGenerateTransformationsB<<<generate_blocks,generate_threads>>>(d_source_point_cloud, d_source_neighbors, d_source_indices_dense,
                                                                      d_target_point_cloud,
                                                                      d_transformations_dense);
    cudaSafeCall(cudaDeviceSynchronize());

    // End info
    boost::chrono::high_resolution_clock::time_point time_compact = boost::chrono::high_resolution_clock::now();

    //========================================================================================
    //================================== Advance Td,d test ===================================
    //========================================================================================

    if (h_transformations_dense_count>1024)
    {
        cudaSafeCall(cudaFree((void*)d_transformations_dense));

        std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] To meny results! Please make poly threshold heigher (lower value)!" << "ms\x1B[0m" << std::endl;
        return;
    }

    // Rand indices 2
    std::srand(std::time(0));
    int source_indices_2[32*h_transformations_dense_count];
    for (int i=0; i<32*h_transformations_dense_count; ++i) source_indices_2[i] = std::rand() % source_points;

    int *d_source_indices_2;
    cudaSafeCall(cudaMalloc((void**)&d_source_indices_2, 32*h_transformations_dense_count*sizeof(int)));
    cudaSafeCall(cudaMemcpy((void*)d_source_indices_2, (void*)source_indices_2, 32*h_transformations_dense_count*sizeof(int), cudaMemcpyHostToDevice));

    gpu::Feature::PointType* d_transformed_point_cloud;
    cudaSafeCall(cudaMalloc((void**)&d_transformed_point_cloud, 32*h_transformations_dense_count*sizeof(gpu::Feature::PointType)));
    cudaSafeCall(cudaMemset((void*)d_transformed_point_cloud, 0, 32*h_transformations_dense_count*sizeof(gpu::Feature::PointType)));

    int blocks = ceil(float(h_transformations_dense_count)/32);
    gpuGenerateTransformedPoints2<<<blocks,1024>>>(d_source_point_cloud, source_points, d_source_indices_2,
                                                  d_transformations_dense, h_transformations_dense_count, 32,
                                                  d_transformed_point_cloud);

    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaFree((void*)d_source_indices_2));
    gpu::NeighborIndices result_indices(32*h_transformations_dense_count, 1);
    gpu::Octree::Queries queries(d_transformed_point_cloud, 32*h_transformations_dense_count);
    d_target_octree.radiusSearch(queries, ransac_radius, 1, result_indices);

    int* d_transformations_dense_valid;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_dense_valid, h_transformations_dense_count*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformations_dense_valid, 0, h_transformations_dense_count*sizeof(int)));

    int* d_transformations_count;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_count, sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformations_count, 0, sizeof(int)));

    int tdd_threads = 64;
    int tdd_blocks = ceil(float(h_transformations_dense_count)/generate_threads);
    gpuTddTest<<<tdd_blocks,tdd_threads>>>(result_indices.sizes, ransac_threshold, h_transformations_dense_count, d_transformations_dense_valid, d_transformations_count);
    cudaSafeCall(cudaDeviceSynchronize());

    int* d_transformation_matrix_actual_address;
    cudaSafeCall(cudaMalloc((void**)&d_transformation_matrix_actual_address, sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformation_matrix_actual_address, 0, sizeof(int)));

    int h_transformations_count;
    cudaSafeCall(cudaMemcpy((void*)&h_transformations_count, (void*)d_transformations_count, sizeof(int), cudaMemcpyDeviceToHost));

    float* d_transformation_final;
    cudaSafeCall(cudaMalloc((void**)&d_transformation_final, 16*h_transformations_count*sizeof(float)));
    cudaSafeCall(cudaMemset((void*)d_transformation_final, 0, 16*h_transformations_count*sizeof(float)));

    gpuCompactAtomic<<<1,h_transformations_dense_count>>>(d_transformations_dense,h_transformations_dense_count,
                                                          d_transformations_dense_valid, d_transformation_matrix_actual_address,
                                                          d_transformation_final);
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaFree((void*)d_transformations_dense));
    cudaSafeCall(cudaFree((void*)d_transformed_point_cloud));
    cudaSafeCall(cudaFree((void*)d_transformations_dense_valid));
    cudaSafeCall(cudaFree((void*)d_transformations_count));
    cudaSafeCall(cudaFree((void*)d_transformation_matrix_actual_address));

    boost::chrono::high_resolution_clock::time_point time_tdd = boost::chrono::high_resolution_clock::now();

#else

    //========================================================================================
    //================================ Hypothesis generation =================================
    //========================================================================================

    float *d_transformations_sparse;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_sparse, 16*BLOCKS*THREADS*sizeof(float)));
    cudaSafeCall(cudaMemset((void*)d_transformations_sparse, 0, 16*BLOCKS*THREADS*sizeof(float)));

    int *d_transformations_sparse_valid;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_sparse_valid, BLOCKS*THREADS*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformations_sparse_valid, 0, BLOCKS*THREADS*sizeof(int)));

    gpuGenerateTransformations<<<BLOCKS,THREADS>>>(d_source_point_cloud, d_source_neighbors, source_points, time(0),
                                                   d_target_point_cloud,
                                                   d_transformations_sparse, d_transformations_sparse_valid, ransac_poly);
    cudaSafeCall(cudaDeviceSynchronize());
    boost::chrono::high_resolution_clock::time_point time_generation = boost::chrono::high_resolution_clock::now();

    //========================================================================================
    //=============================== Compact transformations ================================
    //========================================================================================

    int *d_addresses;
    cudaSafeCall(cudaMalloc((void**)&d_addresses, BLOCKS*THREADS*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_addresses, 0, BLOCKS*THREADS*sizeof(int)));

    int *d_addresses_aux;
    cudaSafeCall(cudaMalloc((void**)&d_addresses_aux, BLOCKS/2*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_addresses_aux, 0, BLOCKS/2*sizeof(int)));

    int *d_addresses_aux_scanned;
    cudaSafeCall(cudaMalloc((void**)&d_addresses_aux_scanned, BLOCKS/2*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_addresses_aux_scanned, 0, BLOCKS/2*sizeof(int)));

    int numBlocks = ceil((float)BLOCKS*THREADS/(THREADS<<1));
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(THREADS, 1, 1);

    gpuExclusiveSumScan<<<dimGrid,dimBlock,THREADS*2*sizeof(int)>>>(d_transformations_sparse_valid, BLOCKS*THREADS, d_addresses, d_addresses_aux);
    gpuExclusiveSumScan<<<dim3(1,1,1), dimGrid,BLOCKS*sizeof(int)>>>(d_addresses_aux, BLOCKS/2, d_addresses_aux_scanned);
    gpuScanFixup<<<dimGrid,dimBlock>>>(d_addresses, BLOCKS*THREADS, d_addresses_aux_scanned);
    cudaSafeCall(cudaDeviceSynchronize());

    int h_transformations_dense_count;
    cudaSafeCall(cudaMemcpy((void*)&h_transformations_dense_count, (void*)&(d_addresses[BLOCKS*THREADS-1]), sizeof(int), cudaMemcpyDeviceToHost));

    float *d_transformations_dense;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_dense, 16*h_transformations_dense_count*sizeof(float)));
    cudaSafeCall(cudaMemset((void*)d_transformations_dense, 0, 16*h_transformations_dense_count*sizeof(float)));

    gpuCompact<<<BLOCKS,THREADS>>>(d_transformations_sparse, BLOCKS*THREADS,
                                   d_transformations_sparse_valid, d_addresses,
                                   d_transformations_dense);
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaFree((void*)d_transformations_sparse));
    cudaSafeCall(cudaFree((void*)d_transformations_sparse_valid));
    cudaSafeCall(cudaFree((void*)d_addresses));
    cudaSafeCall(cudaFree((void*)d_addresses_aux));
    cudaSafeCall(cudaFree((void*)d_addresses_aux_scanned));

    boost::chrono::high_resolution_clock::time_point time_compact = boost::chrono::high_resolution_clock::now();

    //========================================================================================
    //================================== Advance Td,d test ===================================
    //========================================================================================

    if (h_transformations_dense_count>1024)
    {
        cudaSafeCall(cudaFree((void*)d_transformations_dense));

        std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] To meny results! Please make poly threshold heigher (lower value)!" << "ms\x1B[0m" << std::endl;
        return;
    }

    gpu::Feature::PointType* d_transformed_point_cloud;
    cudaSafeCall(cudaMalloc((void**)&d_transformed_point_cloud, 32*h_transformations_dense_count*sizeof(gpu::Feature::PointType)));
    cudaSafeCall(cudaMemset((void*)d_transformed_point_cloud, 0, 32*h_transformations_dense_count*sizeof(gpu::Feature::PointType)));

    int blocks = ceil(float(h_transformations_dense_count)/32);
    gpuGenerateTransformedPoints<<<blocks,1024>>>(d_source_point_cloud, source_points, time(0),
                                                  d_transformations_dense, h_transformations_dense_count, 32,
                                                  d_transformed_point_cloud);
    cudaSafeCall(cudaDeviceSynchronize());

    gpu::NeighborIndices result_indices(32*h_transformations_dense_count, 1);
    gpu::Octree::Queries queries(d_transformed_point_cloud, 32*h_transformations_dense_count);
    d_target_octree.radiusSearch(queries, ransac_radius, 1, result_indices);

    int* d_transformations_dense_valid;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_dense_valid, h_transformations_dense_count*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformations_dense_valid, 0, h_transformations_dense_count*sizeof(int)));

    int* d_transformations_count;
    cudaSafeCall(cudaMalloc((void**)&d_transformations_count, sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformations_count, 0, sizeof(int)));

    gpuTddTest<<<1,h_transformations_dense_count>>>(result_indices.sizes, ransac_threshold, h_transformations_dense_count, d_transformations_dense_valid, d_transformations_count);
    cudaSafeCall(cudaDeviceSynchronize());

    int* d_transformation_matrix_actual_address;
    cudaSafeCall(cudaMalloc((void**)&d_transformation_matrix_actual_address, sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_transformation_matrix_actual_address, 0, sizeof(int)));

    int h_transformations_count;
    cudaSafeCall(cudaMemcpy((void*)&h_transformations_count, (void*)d_transformations_count, sizeof(int), cudaMemcpyDeviceToHost));

    float* d_transformation_final;
    cudaSafeCall(cudaMalloc((void**)&d_transformation_final, 16*h_transformations_count*sizeof(float)));
    cudaSafeCall(cudaMemset((void*)d_transformation_final, 0, 16*h_transformations_count*sizeof(float)));

    gpuCompactAtomic<<<1,h_transformations_dense_count>>>(d_transformations_dense,h_transformations_dense_count,
                                                          d_transformations_dense_valid, d_transformation_matrix_actual_address,
                                                          d_transformation_final);
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaFree((void*)d_transformations_dense));
    cudaSafeCall(cudaFree((void*)d_transformed_point_cloud));
    cudaSafeCall(cudaFree((void*)d_transformations_dense_valid));
    cudaSafeCall(cudaFree((void*)d_transformations_count));
    cudaSafeCall(cudaFree((void*)d_transformation_matrix_actual_address));

    boost::chrono::high_resolution_clock::time_point time_tdd = boost::chrono::high_resolution_clock::now();

#endif

    //========================================================================================
    //================================== Final inlier ratio ==================================
    //========================================================================================

    gpu::Feature::PointType* d_transformed_point_cloud_final;
    cudaSafeCall(cudaMalloc((void**)&d_transformed_point_cloud_final, source_points*h_transformations_count*sizeof(gpu::Feature::PointType)));
    cudaSafeCall(cudaMemset((void*)d_transformed_point_cloud_final, 0, source_points*h_transformations_count*sizeof(gpu::Feature::PointType)));

    int final_blocks = ceil(float(source_points*h_transformations_count)/1024);
    gpuGenerateTransformedPoints<<<final_blocks,1024>>>(d_source_point_cloud, source_points, time(0),
                                                        d_transformation_final, h_transformations_count, source_points,
                                                        d_transformed_point_cloud_final);
    cudaSafeCall(cudaDeviceSynchronize());

    gpu::NeighborIndices result_indices_final(source_points*h_transformations_count, 1);
    gpu::Octree::Queries queries_final(d_transformed_point_cloud_final, source_points*h_transformations_count);
    d_target_octree.radiusSearch(queries_final, ransac_radius, 1, result_indices_final);

    int* d_final_inliers;
    cudaSafeCall(cudaMalloc((void**)&d_final_inliers, h_transformations_count*sizeof(int)));
    cudaSafeCall(cudaMemset((void*)d_final_inliers, 0, h_transformations_count*sizeof(int)));

    gpuFinalTest<<<h_transformations_count,1024>>>(result_indices_final.sizes, ransac_threshold, source_points,
                                                 d_final_inliers);
    cudaSafeCall(cudaDeviceSynchronize());

    int* h_final_inliers= new int[h_transformations_count];
    cudaSafeCall(cudaMemcpy((void*)h_final_inliers, (void*)d_final_inliers, h_transformations_count*sizeof(int), cudaMemcpyDeviceToHost));

    int index_max=0;
    for (int i=0; i<h_transformations_count; i++)
        if (h_final_inliers[i]>h_final_inliers[index_max]) index_max = i;

    if ((float)h_final_inliers[index_max]/source_points > ransac_threshold)
    {
        ransac_converged = 1;
        cudaSafeCall(cudaMemcpy((void*)transformation_matrix, (void*)&d_transformation_final[16*index_max], 16*sizeof(float), cudaMemcpyDeviceToHost));
        ransac_inliers = h_final_inliers[index_max];
    } else ransac_converged = 0;

    printf("Ilosc transformatcji na poczatku        = %i\n",BLOCKS*THREADS);
    printf("Ilosc transformatcji po rejekcji poly   = %i\n",h_transformations_dense_count);
    printf("Ilosc transformatcji po tescie tdd      = %i\n",h_transformations_count);

    delete h_final_inliers;
    cudaSafeCall(cudaFree((void*)d_transformation_final));
    cudaSafeCall(cudaFree((void*)d_transformed_point_cloud_final));
    cudaSafeCall(cudaFree((void*)d_final_inliers));

    boost::chrono::high_resolution_clock::time_point time_final = boost::chrono::high_resolution_clock::now();

    //========================================================================================
    //========================================= Info =========================================
    //========================================================================================

    float a = ((float)(time_generation - time_start).count())/1000000;
    float b = ((float)(time_compact - time_generation).count())/1000000;
    float c = ((float)(time_tdd - time_compact).count())/1000000;
    float d = ((float)(time_final - time_tdd).count())/1000000;
    std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Time of hypothesis generation   = " << a << "ms\x1B[0m" << std::endl;
    std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Time of compact transformations = " << b << "ms\x1B[0m" << std::endl;
    std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Time of advanced tdd test       = " << c << "ms\x1B[0m" << std::endl;
    std::cout << "\x1B[32m" << "[Heuros::RegistrationUtilsGPU] Time of final test              = " << d << "ms\x1B[0m" << std::endl;

    std::cout << "\033[31m" << "[Heuros::Registration] GPU poly: " << h_transformations_dense_count << " \033[0m" << std::endl;
    std::cout << "\033[31m" << "[Heuros::Registration] GPU tdd: " << h_transformations_count << " \033[0m" << std::endl;
}

}

//========================================================================================
//=================================== Save to file =======================================
//========================================================================================

//printf("!Start copy!\n");

//float* h_transformations_dense = new float[16*h_transformations_dense_count];
//float* h_transformations_sparse = new float[16*BLOCKS*THREADS];
//int*   h_transformations_sparse_valid = new int[BLOCKS*THREADS];
//int*   h_addresses = new int[BLOCKS*THREADS];
//int*   h_addresses_aux = new int[BLOCKS/2];
//int*   h_addresses_aux_scanned = new int[BLOCKS/2];
//cudaSafeCall(cudaMemcpy((void*)h_transformations_dense, (void*)d_transformations_dense, 16*h_transformations_dense_count*sizeof(float), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_transformations_sparse, (void*)d_transformations_sparse, 16*BLOCKS*THREADS*sizeof(float), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_transformations_sparse_valid, (void*)d_transformations_sparse_valid, BLOCKS*THREADS*sizeof(int), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_addresses, (void*)d_addresses, BLOCKS*THREADS*sizeof(int), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_addresses_aux, (void*)d_addresses_aux, BLOCKS/2*sizeof(int), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_addresses_aux_scanned, (void*)d_addresses_aux_scanned, BLOCKS/2*sizeof(int), cudaMemcpyDeviceToHost));

//for (int i=0;i<BLOCKS/2;i++)
//{
//    printf("h_addresses_aux[%i]=%i\t%i\n",i,h_addresses_aux[i],h_addresses_aux_scanned[i]);
//}


//printf("!Start writng a file!\n");

//FILE *file;
//file = fopen("/home/daniel/heuros2_ws/src/heuros2/registration/files/purity/transformations_dense.csv", "w");
//setlocale(LC_ALL, ".OCP");

//for (int i=0; i<h_transformations_dense_count; i++)
//    fprintf(file, "%f\n", h_transformations_dense[16*i+0]);

//fclose(file);

//file = fopen("/home/daniel/heuros2_ws/src/heuros2/registration/files/purity/transformations_sparse.csv", "w");

//for (int i=0; i<BLOCKS*THREADS; i++)
//    fprintf(file, "%f;%i;%i\n", h_transformations_sparse[16*i+0], h_addresses[i], h_transformations_sparse_valid[i]);
//    //fprintf(file, "%i\n", h_addresses[i]);

//fclose(file);

//printf("!Done!\n!");

//========================================================================================
//=================================== Save to file =======================================
//========================================================================================

//========================================================================================
//=================================== Save to file =======================================
//========================================================================================

//printf("!Start copy!\n");

//int h_valid[BLOCKS*THREADS];
//int h_addresses[BLOCKS*THREADS];
//float h_transf[16*BLOCKS*THREADS];
//cudaSafeCall(cudaMemcpy((void*)h_addresses, (void*)d_addresses, BLOCKS*THREADS*sizeof(int), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_valid, (void*)d_valid_transform, BLOCKS*THREADS*sizeof(int), cudaMemcpyDeviceToHost));
//cudaSafeCall(cudaMemcpy((void*)h_transf, (void*)*d_transformation_matrix, 16*transformations_count*sizeof(float), cudaMemcpyDeviceToHost));

//printf("!Start writng a file!\n");

//FILE *file;
//file = fopen("/home/daniel/heuros2_ws/src/heuros2/registration/files/compact_check/compact_check_3.txt", "w");

//for (int i=0; i<transformations_count; i++)
//{
//    fprintf(file, "Nr %i\t",i);
//    for (int j=0; j<16; j++)
//    {
//        fprintf(file, "%f\t",h_transf[16*i+j]);
//    }
//    fprintf(file, "\n");
//}

//fclose(file);

//printf("!Done!\n!");

//========================================================================================
//=================================== Save to file =======================================
//========================================================================================
