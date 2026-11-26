// Small cyclic convolutions: radix-8 local butterflies, subgroup combination.
// One independent ring occupies N/8 lanes of a warp, N in {8,16,32,64}.
#include "ring_fft.cuh"
#include <cmath>

namespace ringfft {
__device__ __constant__ Twiddle g_tw;

void upload_twiddles() {
    Twiddle h;
    for (int k = 0; k < 64; ++k) {
        double a = -2.0 * 3.14159265358979323846 * k / 64.0;
        h.W[k] = make_float2(static_cast<float>(std::cos(a)), static_cast<float>(std::sin(a)));
    }
    cudaMemcpyToSymbol(g_tw, &h, sizeof(h));
}
__device__ __forceinline__ float2 add(float2 a, float2 b) { return make_float2(a.x+b.x,a.y+b.y); }
__device__ __forceinline__ float2 sub(float2 a, float2 b) { return make_float2(a.x-b.x,a.y-b.y); }
__device__ __forceinline__ float2 mul(float2 a, float2 b) { return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ __forceinline__ float2 twiddle(int k, bool inverse) {
    float2 w = g_tw.W[k & 63];
    return inverse ? make_float2(w.x, -w.y) : w;
}
__device__ __forceinline__ void radix4(const float2 *x, float2 *y, bool inverse) {
    float2 e0=add(x[0],x[2]), e1=sub(x[0],x[2]);
    float2 o0=add(x[1],x[3]), d=sub(x[1],x[3]);
    float2 rot=inverse ? make_float2(-d.y,d.x) : make_float2(d.y,-d.x);
    y[0]=add(e0,o0); y[1]=add(e1,rot); y[2]=sub(e0,o0); y[3]=sub(e1,rot);
}
__device__ __forceinline__ void radix8(const float2 *x, float2 *y, bool inverse) {
    float2 even[4], odd[4], e[4], o[4];
#pragma unroll
    for(int i=0;i<4;i++){even[i]=x[2*i]; odd[i]=x[2*i+1];}
    radix4(even,e,inverse); radix4(odd,o,inverse);
#pragma unroll
    for(int i=0;i<4;i++){float2 t=mul(o[i],twiddle(i*8,inverse)); y[i]=add(e[i],t); y[i+4]=sub(e[i],t);}
}

template<int N>
__device__ __forceinline__ void transform(float2 *v, int sublane, unsigned mask, bool inverse) {
    constexpr int L=N/8;
    float2 local[8], result[8];
    radix8(v,local,inverse);
    // Input lane s has the 8-point DFT of samples x[s + L*t]. For output k,
    // X[k] = sum_s local_s[k mod 8] * W_N^(s*k).
#pragma unroll
    for(int t=0;t<8;t++) {
        int k=sublane+L*t, p=k&7;
        float2 sum=make_float2(0.f,0.f);
#pragma unroll
        for(int s=0;s<L;s++) {
            // p differs between lanes. Shuffle every register with a uniform index,
            // then select locally; shuffling local[p] directly reads the sender's p.
            float2 z=make_float2(0.f,0.f);
#pragma unroll
            for(int r=0;r<8;r++) {
                float2 candidate=make_float2(__shfl_sync(mask,local[r].x,s,L),
                                             __shfl_sync(mask,local[r].y,s,L));
                if(r==p) z=candidate;
            }
            sum=add(sum,mul(z,twiddle(s*k*(64/N),inverse)));
        }
        result[t]=sum;
    }
#pragma unroll
    for(int t=0;t<8;t++) v[t]=result[t];
}

template<int N>
__global__ void ring_conv_kernel(const float2 *a,const float2 *w,float2 *out,int batch) {
    constexpr int L=N/8;
    int global_thread=blockIdx.x*blockDim.x+threadIdx.x;
    int ring=global_thread/L, lane=threadIdx.x % L;
    if(ring>=batch) return;
    unsigned mask=__activemask();
    float2 x[8],h[8];
#pragma unroll
    for(int t=0;t<8;t++){int i=lane+L*t;x[t]=a[ring*N+i];h[t]=w[i];}
    transform<N>(x,lane,mask,false);
    transform<N>(h,lane,mask,false);
#pragma unroll
    for(int t=0;t<8;t++) x[t]=mul(x[t],h[t]);
    transform<N>(x,lane,mask,true);
#pragma unroll
    for(int t=0;t<8;t++){int i=lane+L*t;out[ring*N+i]=make_float2(x[t].x/N,x[t].y/N);}
}

template<int N>
cudaError_t ring_conv_launch(const float2 *a,const float2 *w,float2 *out,int batch,cudaStream_t stream) {
    if(batch<0 || !a || !w || !out) return cudaErrorInvalidValue;
    if(batch==0) return cudaSuccess;
    constexpr int threads=256;
    const int blocks=(batch+(threads/(N/8))-1)/(threads/(N/8));
    ring_conv_kernel<N><<<blocks,threads,0,stream>>>(a,w,out,batch);
    return cudaGetLastError();
}
cudaError_t ring_conv(const float2 *a,const float2 *w,float2 *out,int n,int batch,cudaStream_t stream) {
    switch(n){
        case 8:return ring_conv_launch<8>(a,w,out,batch,stream);
        case 16:return ring_conv_launch<16>(a,w,out,batch,stream);
        case 32:return ring_conv_launch<32>(a,w,out,batch,stream);
        case 64:return ring_conv_launch<64>(a,w,out,batch,stream);
        default:return cudaErrorInvalidValue;
    }
}
}
