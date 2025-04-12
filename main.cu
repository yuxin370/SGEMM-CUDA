#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
#include<stdio.h>
#include<cuda.h>

#define N 1024
#define ITERATIONS 10
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 32
#define TILE_WIDTH 32
using namespace std;

__global__ void sgemm_original(float *A, float *B, float *C, int n, float a, float b) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if( (i<n) && (j<n) ){
    float tmp = b*C[i*n+j];
    for(int k=0; k<n; k++){
      tmp += a*A[i*n+k]*B[k*n+j];
    }
    C[i*n+j]=tmp;
  }
}

__global__ void sgemm(float *A, float *B, float *C, int n, float a, float b) {
  int tx= threadIdx.x, ty = threadIdx.y;
  int bx = blockIdx.x, by = blockIdx.y;
  __shared__ float s_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ float s_b[TILE_WIDTH][TILE_WIDTH];
  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  if((row<n) && (col<n)){
    float result = b*C[row*n+col];
    for(int p = 0; p < n/TILE_WIDTH ; ++p){
      s_a[ty][tx] = A[row*n + (p*TILE_WIDTH + tx)];
      s_b[ty][tx] = B[(p*TILE_WIDTH + ty)*n + col];
      __syncthreads();

      #pragma unroll
      for(int k = 0; k < TILE_WIDTH; ++k){
        result += a*s_a[ty][k] * s_b[k][tx];
      }
      __syncthreads();
    }
    C[row*n + col] = result;
  }
}

void printM(float *res, int n){
  for(int i = 0 ; i < n ; i ++){
    for(int j = 0; j< n ; j ++){
      cout<<res[i*n+j]<<" ";
    }
    cout<<endl;
  }
}

void compare(float* res1, float* res2, int n){
  int fail=0;
  for(int i=0; i<n; i++){
    float a,b;
    if(res1[i]<0)
      a=res1[i]*(-1);
    else 
      a=res1[i];
    if(res2[i]<0)
      b=res2[i]*(-1);
    else 
      b=res2[i];
    if((a<0.01)&&(b<0.01)){
      continue;
    }
    float diff=(a-b)/(a+0.000001);
    if(diff<0)
      diff=diff*(-1);
    if(diff>0.0005)
      fail++;
  }
  printf("Number of errors: %d\n", fail);
}

double timestamp(){
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(){
  // float A[N*N], B[N*N], C_cpu[N*N], C_gpu_final[N*N];
  float *A = new float[N*N];
  float *B = new float[N*N];
  float *C_cpu = new float[N*N];
  float *C_gpu_final = new float[N*N];
  float a=1, b=1;
  for(int i=0; i<N; i++){
    for(int j=0; j<N; j++){
      B[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      A[i*N+j]=(float)rand()/(float)(RAND_MAX/a);
      C_cpu[i*N+j]=0;
      C_gpu_final[i*N+j]=0;
    }
  }
  // cout<<"Matrix A: "<<endl;
  // printM(A,N);
  // cout<<"Matrix B: "<<endl;
  // printM(B,N);

  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      C_cpu[i*N+j]+=b*C_cpu[i*N+j];
      for(int k=0; k<N; k++){
        C_cpu[i*N+j] += a*A[i*N+k]*B[k*N+j];
      }
    }
  }

  float *A_gpu;
  float *B_gpu;
  float *C_gpu;
  cudaMalloc((void **)&A_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&B_gpu, sizeof(float)*N*N);
  cudaMalloc((void **)&C_gpu, sizeof(float)*N*N);
  cudaMemcpy(A_gpu, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, B, sizeof(float)*N*N, cudaMemcpyHostToDevice);
  cudaMemcpy(C_gpu, C_gpu_final, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
  dim3 grid((size_t)ceil( ((float)N) / ((float)block.x) ), (size_t)ceil( ((float)N) / ((float)block.y)) );

  sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);
  cudaThreadSynchronize();
  cudaMemcpy(C_gpu_final, C_gpu, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  // cout<<"right answer : "<<endl;
  // printM(C_cpu,N);
  // cout<<"my answer : "<<endl;
  // printM(C_gpu_final,N);
  compare(C_cpu, C_gpu_final, N*N);

  double time1=timestamp();
  for(int numOfTimes=0; numOfTimes<ITERATIONS; numOfTimes++){

    sgemm<<<grid,block>>>(A_gpu, B_gpu, C_gpu, N, a, b);

  }
  cudaThreadSynchronize();
  double time2=timestamp();

  double time = (time2-time1)/ITERATIONS;
  double flops = 2*(double(N))*N*N;
  double gflopsPerSecond = flops/(1000000000)/time;
  double GB = (double)(N)*N*4/1000000000;
  double GBpS = (double)(N)*N*4/1000000000/time;
  printf("GFLOPS/s=%lf\n",gflopsPerSecond );
  printf("GB/s=%lf\n",GBpS);
  printf("GFLOPS=%lf\n",flops/(1000000000));
  printf("GB=%lf\n",GB);
  printf("time(s)=%lf\n",time);

  
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  return 0;
}