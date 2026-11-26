"""Compile the actual device kernel with NVRTC and compare on CUDA to direct convolution.
Standalone diagnostic, no application or checkpoint changes.
"""
import ctypes as C
import ctypes.util
import glob
from pathlib import Path


def with_kernel(callback):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device required; no CPU fallback')
    src=(Path(__file__).resolve().parent/'ring_fft.cu').read_text()
    start=src.index('__device__ __forceinline__ float2 add(')
    end=src.index('\ntemplate<int N>\ncudaError_t ring_conv_launch',start)
    code='struct Twiddle { float2 W[64]; };\n__device__ __constant__ Twiddle g_tw;\n'+src[start:end]
    # unmangled wrapper names for driver API lookup
    for n in (8,16,32,64):
        code+=f'\ntemplate __global__ void ring_conv_kernel<{n}>(const float2*,const float2*,float2*,int);\n'
    paths=glob.glob('/usr/local/lib/python*/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so*')
    lib=C.CDLL(paths[0] if paths else (ctypes.util.find_library('nvrtc') or 'libnvrtc.so.12'))
    def bind(obj,name,args):
        f=getattr(obj,name);f.argtypes=args;f.restype=C.c_int;return f
    create=bind(lib,'nvrtcCreateProgram',[C.POINTER(C.c_void_p),C.c_char_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p])
    compile_=bind(lib,'nvrtcCompileProgram',[C.c_void_p,C.c_int,C.POINTER(C.c_char_p)])
    addname=bind(lib,'nvrtcAddNameExpression',[C.c_void_p,C.c_char_p])
    lowered=bind(lib,'nvrtcGetLoweredName',[C.c_void_p,C.c_char_p,C.POINTER(C.c_char_p)])
    logsize=bind(lib,'nvrtcGetProgramLogSize',[C.c_void_p,C.POINTER(C.c_size_t)])
    logget=bind(lib,'nvrtcGetProgramLog',[C.c_void_p,C.c_char_p])
    ptxsize=bind(lib,'nvrtcGetPTXSize',[C.c_void_p,C.POINTER(C.c_size_t)])
    ptxget=bind(lib,'nvrtcGetPTX',[C.c_void_p,C.c_char_p])
    p=C.c_void_p()
    def ok(code,where):
        if code: raise RuntimeError(f'{where}: error {code}')
    ok(create(C.byref(p),code.encode(),b'ring.cu',0,None,None),'nvrtcCreateProgram')
    names={n:f'ring_conv_kernel<{n}>'.encode() for n in (8,16,32,64)}
    for expr in names.values():ok(addname(p,expr),'nvrtcAddNameExpression')
    options=(C.c_char_p*2)(b'--std=c++17',b'--gpu-architecture=compute_75')
    status=compile_(p,2,options)
    size=C.c_size_t();logsize(p,C.byref(size));log=C.create_string_buffer(max(size.value,1));logget(p,log)
    if status:raise RuntimeError(f'NVRTC compilation failed {status}: {log.value.decode()}')
    lowered_names={}
    for n,expr in names.items():
        name=C.c_char_p();ok(lowered(p,expr,C.byref(name)),'nvrtcGetLoweredName');lowered_names[n]=name.value
    ok(ptxsize(p,C.byref(size)),'nvrtcGetPTXSize');ptx=C.create_string_buffer(size.value);ok(ptxget(p,ptx),'nvrtcGetPTX')
    torch.cuda.init()
    # Force PyTorch to establish its primary CUDA context on this thread.
    context_anchor = torch.empty(1, device='cuda')
    drv=C.CDLL('libcuda.so.1')
    init=bind(drv,'cuInit',[C.c_uint])
    device_get=bind(drv,'cuDeviceGet',[C.POINTER(C.c_int),C.c_int])
    retain=bind(drv,'cuDevicePrimaryCtxRetain',[C.POINTER(C.c_void_p),C.c_int])
    setctx=bind(drv,'cuCtxSetCurrent',[C.c_void_p])
    dev=C.c_int(); ctx=C.c_void_p()
    ok(init(0),'cuInit'); ok(device_get(C.byref(dev),torch.cuda.current_device()),'cuDeviceGet')
    ok(retain(C.byref(ctx),dev.value),'cuDevicePrimaryCtxRetain');ok(setctx(ctx),'cuCtxSetCurrent')
    load=bind(drv,'cuModuleLoadData',[C.POINTER(C.c_void_p),C.c_void_p])
    getfun=bind(drv,'cuModuleGetFunction',[C.POINTER(C.c_void_p),C.c_void_p,C.c_char_p])
    getglobal=bind(drv,'cuModuleGetGlobal_v2',[C.POINTER(C.c_uint64),C.POINTER(C.c_size_t),C.c_void_p,C.c_char_p])
    copy=bind(drv,'cuMemcpyHtoD_v2',[C.c_uint64,C.c_void_p,C.c_size_t])
    launch=bind(drv,'cuLaunchKernel',[C.c_void_p,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_uint,C.c_void_p,C.POINTER(C.c_void_p),C.c_void_p])
    unload=bind(drv,'cuModuleUnload',[C.c_void_p])
    mod=C.c_void_p();ok(load(C.byref(mod),C.cast(ptx,C.c_void_p)),'cuModuleLoadData')
    try:
        ptr=C.c_uint64(); size=C.c_size_t()
        ok(getglobal(C.byref(ptr),C.byref(size),mod,b'g_tw'),'cuModuleGetGlobal')
        import math
        tw=(C.c_float*128)()
        for i in range(64):tw[2*i]=math.cos(-2*math.pi*i/64);tw[2*i+1]=math.sin(-2*math.pi*i/64)
        if size.value!=C.sizeof(tw):raise RuntimeError('twiddle table size mismatch')
        ok(copy(ptr,C.cast(tw,C.c_void_p),size.value),'cuMemcpyHtoD')
        funcs = {}
        for n in (8,16,32,64):
            fn=C.c_void_p();ok(getfun(C.byref(fn),mod,lowered_names[n]),'cuModuleGetFunction')
            funcs[n] = fn

        def convolve(a,w):
            if not (a.is_cuda and w.is_cuda and a.device == w.device):
                raise ValueError('inputs must be CUDA tensors on the same device')
            if a.dtype != torch.complex64 or w.dtype != torch.complex64:
                raise TypeError('inputs must be complex64')
            if a.ndim != 2 or w.ndim != 1 or a.shape[1] != w.shape[0] or w.numel() not in funcs:
                raise ValueError('expected a[batch,n] and w[n], n=8,16,32,64')
            if not a.is_contiguous() or not w.is_contiguous():
                raise ValueError('inputs must be contiguous')
            if a.requires_grad or w.requires_grad:
                raise ValueError('CUDA interface does not implement gradients')
            if a.device.index != torch.cuda.current_device():
                raise ValueError('inputs must be on the active CUDA device')
            n, batch = w.numel(), a.shape[0]
            out=torch.empty_like(a)
            if not batch: return out
            va=C.c_uint64(a.data_ptr());vw=C.c_uint64(w.data_ptr());vo=C.c_uint64(out.data_ptr());vb=C.c_int(batch)
            args=(C.c_void_p*4)(*[C.cast(C.byref(x),C.c_void_p) for x in (va,vw,vo,vb)])
            blocks=(batch+(256//(n//8))-1)//(256//(n//8))
            stream=C.c_void_p(torch.cuda.current_stream().cuda_stream)
            ok(launch(funcs[n],blocks,1,1,256,1,1,0,stream,args,None),'cuLaunchKernel')
            return out
        try:
            return callback(convolve)
        finally:
            # Module unload must not race work queued by the caller.
            torch.cuda.synchronize()
    finally:
        unload(mod)

def verify():
    import torch
    def run(convolve):
        torch.manual_seed(719)
        result=[]
        for n in (8,16,32,64):
            maxerr=0.
            for batch in (1,3,5,31,33,260):
                for case in ('random','delta'):
                    a=torch.randn(batch,n,device='cuda',dtype=torch.complex64)
                    w=torch.randn(n,device='cuda',dtype=torch.complex64)
                    if case=='delta': w.zero_();w[0]=1
                    out=convolve(a,w)
                    expected=sum(torch.roll(a,j,dims=-1)*w[j] for j in range(n))
                    err=(out-expected).abs().max().item();maxerr=max(maxerr,err)
                    if not torch.allclose(out,expected,atol=2e-4,rtol=2e-4):
                        raise AssertionError(f'N={n} batch={batch} {case}: max abs error={err}')
            result.append(f'N={n}: 12 cases PASS, max abs error {maxerr:.3g}')
        return '\n'.join([f'GPU: {torch.cuda.get_device_name()}','Compiled for compute_75; tolerance atol=rtol=2e-4',*result])
    return with_kernel(run)

if __name__=='__main__':print(verify())
