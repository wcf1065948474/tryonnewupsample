#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 256 
#define CUDA_MAX_THREADS 256 

// #define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])
#define EPS 1e-8
#define SAFE_DIV(a, b)  ( (b==0)? ( (a)/(EPS) ): ( (a)/(b) )  )
#define CHECK_LEGALITY(x, min, max) ((x>=min && x<=max)? (true):(false) )

template <typename scalar_t>
__global__ void kernel_block_extractor_update_output(const int n, 
                                                const scalar_t* __restrict__ source_a, 
                                                const scalar_t* __restrict__ source_b, 
                                                const scalar_t* __restrict__ source_c, 
                                                const long4 source_size, 
                                                const long4 source_stride,
                                                const scalar_t* __restrict__ flow_field_a,
                                                const scalar_t* __restrict__ flow_field_b,
                                                const scalar_t* __restrict__ flow_field_c,
                                                const long4 flow_field_size, 
                                                const long4 flow_field_stride,
                                                const scalar_t* __restrict__ masks_a, 
                                                const scalar_t* __restrict__ masks_b, 
                                                const scalar_t* __restrict__ masks_c, 
                                                const long4 masks_size, 
                                                const long4 masks_stride,
                                                scalar_t* __restrict__ output, 
                                                const long4 output_size, 
                                                const long4 output_stride, 
                                                int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const scalar_t* __restrict__ source = NULL;
    const scalar_t* __restrict__ flow_field = NULL;
    const long4 masks_a_stride = masks_stride;
    const long4 masks_b_stride = masks_stride;
    const long4 masks_c_stride = masks_stride;


    if (index >= n) {
        return;
    }
    

    int dim_b   = DIM0(output_size);
    int dim_c   = DIM1(output_size);
    int dim_h   = DIM2(output_size);
    int dim_w   = DIM3(output_size);
    int dim_chw = DIM0(output_stride);
    int dim_hw  = DIM1(output_stride);

    int dim_hs   = DIM2(source_size);
    int dim_ws   = DIM3(source_size);


    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw  ) % dim_c;
    int y = ( index / dim_w   ) % dim_h;
    int x = ( index           ) % dim_w;

    int yf = y/kernel_size;
    int xf = x/kernel_size;
    int yf_offset = y%kernel_size - kernel_size/2;
    int xf_offset = x%kernel_size - kernel_size/2;

    if(DIM3_INDEX(masks_a,b,0,yf,xf)==1){
        source = source_a;
        flow_field = flow_field_a;
    }else if(DIM3_INDEX(masks_b,b,0,yf,xf)==1){
        source = source_b;
        flow_field = flow_field_b;
    }else if(DIM3_INDEX(masks_c,b,0,yf,xf)==1){
        source = source_c;
        flow_field = flow_field_c;
    }else
    return;

    scalar_t flow_y = DIM3_INDEX(flow_field, b, 1, yf, xf) + yf_offset;
    scalar_t flow_x = DIM3_INDEX(flow_field, b, 0, yf, xf) + xf_offset;


    scalar_t dy = flow_y + static_cast<scalar_t>(yf);
    scalar_t dx = flow_x + static_cast<scalar_t>(xf);

    int xL = max(min( int(floor(dx)    ), dim_ws-1), 0);
    int xR = max(min( int(floor(dx) + 1), dim_ws-1), 0);
    int yT = max(min( int(floor(dy)    ), dim_hs-1), 0);
    int yB = max(min( int(floor(dy) + 1), dim_hs-1), 0);
    scalar_t xL_P = 1 - (dx - floor(dx));
    scalar_t xR_P = dx - floor(dx);
    scalar_t yT_P = 1 - (dy - floor(dy));
    scalar_t yB_P = dy - floor(dy);

    scalar_t sample = 0.0f;
    sample += (xL_P*yT_P * DIM3_INDEX(source, b, c, yT, xL));
    sample += (xR_P*yT_P * DIM3_INDEX(source, b, c, yT, xR));
    sample += (xL_P*yB_P * DIM3_INDEX(source, b, c, yB, xL));
    sample += (xR_P*yB_P * DIM3_INDEX(source, b, c, yB, xR));

    output[index] = sample;
}



template <typename scalar_t>
__global__ void kernel_block_extractor_backward(
                                            const int n,  
                                            const scalar_t* __restrict__ source_a, 
                                            const scalar_t* __restrict__ source_b, 
                                            const scalar_t* __restrict__ source_c, 
                                            const long4 source_size, 
                                            const long4 source_stride,
                                            const scalar_t* __restrict__ flow_field_a, 
                                            const scalar_t* __restrict__ flow_field_b, 
                                            const scalar_t* __restrict__ flow_field_c, 
                                            const long4 flow_field_size, 
                                            const long4 flow_field_stride,
                                            const scalar_t* __restrict__ masks_a, 
                                            const scalar_t* __restrict__ masks_b, 
                                            const scalar_t* __restrict__ masks_c, 
                                            const long4 masks_size, 
                                            const long4 masks_stride,
                                            const scalar_t* __restrict__ grad_output, 
                                            const long4 grad_output_size, 
                                            const long4 grad_output_stride,
                                            scalar_t* __restrict__ grad_source_a, 
                                            scalar_t* __restrict__ grad_source_b, 
                                            scalar_t* __restrict__ grad_source_c, 
                                            const long4 grad_source_size, 
                                            const long4 grad_source_stride, 
                                            scalar_t* __restrict__ grad_flow_field_a, 
                                            scalar_t* __restrict__ grad_flow_field_b, 
                                            scalar_t* __restrict__ grad_flow_field_c, 
                                            const long4 grad_flow_field_size, 
                                            const long4 grad_flow_field_stride,         
                                            int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    const scalar_t* __restrict__ source = NULL;
    const scalar_t* __restrict__ flow_field = NULL;
    scalar_t* __restrict__ grad_source = NULL;
    scalar_t* __restrict__ grad_flow_field = NULL;
    const long4 masks_a_stride = masks_stride;
    const long4 masks_b_stride = masks_stride;
    const long4 masks_c_stride = masks_stride;


    if (index >= n) {
        return;
    }


    int dim_b   = DIM0(grad_output_size);
    int dim_c   = DIM1(grad_output_size);
    int dim_h   = DIM2(grad_output_size);
    int dim_w   = DIM3(grad_output_size);
    int dim_chw = DIM0(grad_output_stride);
    int dim_hw  = DIM1(grad_output_stride);

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw  ) % dim_c;
    int y = ( index / dim_w   ) % dim_h;
    int x = ( index           ) % dim_w;

    int dim_hs   = DIM2(source_size);
    int dim_ws   = DIM3(source_size);
    
    int yf = y/kernel_size;
    int xf = x/kernel_size;
    int yf_offset = y%kernel_size - kernel_size/2;
    int xf_offset = x%kernel_size - kernel_size/2;

    if(DIM3_INDEX(masks_a,b,0,yf,xf)==1){
        source = source_a;
        flow_field = flow_field_a;
        grad_source = grad_source_a;
        grad_flow_field = grad_flow_field_a;
    }else if(DIM3_INDEX(masks_b,b,0,yf,xf)==1){
        source = source_b;
        flow_field = flow_field_b;
        grad_source = grad_source_b;
        grad_flow_field = grad_flow_field_b;
    }else if(DIM3_INDEX(masks_c,b,0,yf,xf)==1){
        source = source_c;
        flow_field = flow_field_c;
        grad_source = grad_source_c;
        grad_flow_field = grad_flow_field_c;
    }else
    return;

    scalar_t flow_y = DIM3_INDEX(flow_field, b, 1, yf, xf) + yf_offset;
    scalar_t flow_x = DIM3_INDEX(flow_field, b, 0, yf, xf) + xf_offset;

    scalar_t dy = flow_y + static_cast<scalar_t>(yf);
    scalar_t dx = flow_x + static_cast<scalar_t>(xf);

    int xL = max(min( int(floor(dx)    ), dim_ws-1), 0);
    int xR = max(min( int(floor(dx) + 1), dim_ws-1), 0);
    int yT = max(min( int(floor(dy)    ), dim_hs-1), 0);
    int yB = max(min( int(floor(dy) + 1), dim_hs-1), 0);
    scalar_t xL_P = 1 - (dx - floor(dx));
    scalar_t xR_P = dx - floor(dx);
    scalar_t yT_P = 1 - (dy - floor(dy));
    scalar_t yB_P = dy - floor(dy);

    scalar_t xL_yT = DIM3_INDEX(source, b, c, yT, xL);
    scalar_t xR_yT = DIM3_INDEX(source, b, c, yT, xR);
    scalar_t xL_yB = DIM3_INDEX(source, b, c, yB, xL);
    scalar_t xR_yB = DIM3_INDEX(source, b, c, yB, xR);

    scalar_t grad = DIM3_INDEX(grad_output, b, c, y, x);

    atomicAdd(&DIM3_INDEX(grad_source, b, c, yT, xL), grad*xL_P*yT_P);
    atomicAdd(&DIM3_INDEX(grad_source, b, c, yT, xR), grad*xR_P*yT_P);
    atomicAdd(&DIM3_INDEX(grad_source, b, c, yB, xL), grad*xL_P*yB_P);
    atomicAdd(&DIM3_INDEX(grad_source, b, c, yB, xR), grad*xR_P*yB_P);

    scalar_t grady = grad*(-xL_P*xL_yT - xR_P*xR_yT + xL_P*xL_yB + xR_P*xR_yB);
    scalar_t gradx = grad*(-yT_P*xL_yT - yB_P*xL_yB + yT_P*xR_yT + yB_P*xR_yB);


    atomicAdd(&DIM3_INDEX(grad_flow_field, b, 1, yf, xf), grady);
    atomicAdd(&DIM3_INDEX(grad_flow_field, b, 0, yf, xf), gradx);

}

void block_extractor_kernel_forward(
    at::Tensor& source_a,
    at::Tensor& source_b,
    at::Tensor& source_c,
    at::Tensor& flow_field_a, 
    at::Tensor& flow_field_b, 
    at::Tensor& flow_field_c, 
    at::Tensor& masks_a,
    at::Tensor& masks_b,
    at::Tensor& masks_c,
    at::Tensor& output,
    int kernel_size) {
    // clock_t start, end;
    // start = clock();

    int n = output.numel();

    const long4 source_size = make_long4(source_a.size(0), source_a.size(1), source_a.size(2), source_a.size(3));
    const long4 source_stride = make_long4(source_a.stride(0), source_a.stride(1), source_a.stride(2), source_a.stride(3));

    const long4 flow_field_size = make_long4(flow_field_a.size(0), flow_field_a.size(1), flow_field_a.size(2), flow_field_a.size(3));
    const long4 flow_field_stride = make_long4(flow_field_a.stride(0), flow_field_a.stride(1), flow_field_a.stride(2), flow_field_a.stride(3));

    const long4 masks_size = make_long4(masks_a.size(0), masks_a.size(1), masks_a.size(2), masks_a.size(3));
    const long4 masks_stride = make_long4(masks_a.stride(0), masks_a.stride(1), masks_a.stride(2), masks_a.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    int Threads = CUDA_NUM_THREADS;

    const dim3 threads(Threads);
    const dim3 blocks((n + Threads - 1) / Threads);

    AT_DISPATCH_FLOATING_TYPES(source_a.type(), "block_extractor_forward_kernel", ([&] {
        kernel_block_extractor_update_output<scalar_t><<< blocks, threads, 0, at::cuda::getCurrentCUDAStream() >>>(
            n,
            source_a.data<scalar_t>(),
            source_b.data<scalar_t>(),
            source_c.data<scalar_t>(),
            source_size,
            source_stride, 
            flow_field_a.data<scalar_t>(),
            flow_field_b.data<scalar_t>(),
            flow_field_c.data<scalar_t>(),
            flow_field_size,
            flow_field_stride,
            masks_a.data<scalar_t>(),
            masks_b.data<scalar_t>(),
            masks_c.data<scalar_t>(),
            masks_size,
            masks_stride,
            output.data<scalar_t>(),
            output_size,
            output_stride,
            kernel_size);
    }));
    
    // end = clock();
    // printf("%d\n", end-start);
        // TODO: ATen-equivalent check

       //    THCudaCheck(cudaGetLastError());

}




void block_extractor_kernel_backward(
    at::Tensor& source_a,
    at::Tensor& source_b, 
    at::Tensor& source_c, 
    at::Tensor& flow_field_a,
    at::Tensor& flow_field_b,
    at::Tensor& flow_field_c,
    at::Tensor& masks_a,
    at::Tensor& masks_b,
    at::Tensor& masks_c,
    at::Tensor& grad_output,
    at::Tensor& grad_source_a, 
    at::Tensor& grad_source_b, 
    at::Tensor& grad_source_c, 
    at::Tensor& grad_flow_field_a, 
    at::Tensor& grad_flow_field_b,
    at::Tensor& grad_flow_field_c,
    int kernel_size) {  

    int n = grad_output.numel();

    const long4 source_size = make_long4(source_a.size(0), source_a.size(1), source_a.size(2), source_a.size(3));
    const long4 source_stride = make_long4(source_a.stride(0), source_a.stride(1), source_a.stride(2), source_a.stride(3));

    const long4 flow_field_size = make_long4(flow_field_a.size(0), flow_field_a.size(1), flow_field_a.size(2), flow_field_a.size(3));
    const long4 flow_field_stride = make_long4(flow_field_a.stride(0), flow_field_a.stride(1), flow_field_a.stride(2), flow_field_a.stride(3));

    const long4 masks_size = make_long4(masks_a.size(0), masks_a.size(1), masks_a.size(2), masks_a.size(3));
    const long4 masks_stride = make_long4(masks_a.stride(0), masks_a.stride(1), masks_a.stride(2), masks_a.stride(3));


    const long4 grad_output_size = make_long4(grad_output.size(0), grad_output.size(1), grad_output.size(2), grad_output.size(3));
    const long4 grad_output_stride = make_long4(grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3));

    const long4 grad_source_size = make_long4(grad_source_a.size(0), grad_source_a.size(1), grad_source_a.size(2), grad_source_a.size(3));
    const long4 grad_source_stride = make_long4(grad_source_a.stride(0), grad_source_a.stride(1), grad_source_a.stride(2), grad_source_a.stride(3));

    const long4 grad_flow_field_size = make_long4(grad_flow_field_a.size(0), grad_flow_field_a.size(1), grad_flow_field_a.size(2), grad_flow_field_a.size(3));
    const long4 grad_flow_field_stride = make_long4(grad_flow_field_a.stride(0), grad_flow_field_a.stride(1), grad_flow_field_a.stride(2), grad_flow_field_a.stride(3));


    int Threads = CUDA_NUM_THREADS;

    const dim3 threads(Threads);
    const dim3 blocks((n + Threads - 1) / Threads);

    AT_DISPATCH_FLOATING_TYPES(source_a.type(), "block_extractor_backward", ([&] {
        kernel_block_extractor_backward<scalar_t><<< blocks, threads, 0, at::cuda::getCurrentCUDAStream() >>>(
            n, 
            source_a.data<scalar_t>(), 
            source_b.data<scalar_t>(), 
            source_c.data<scalar_t>(), 
            source_size,
            source_stride,
            flow_field_a.data<scalar_t>(),
            flow_field_b.data<scalar_t>(),
            flow_field_c.data<scalar_t>(),
            flow_field_size, 
            flow_field_stride,
            masks_a.data<scalar_t>(),
            masks_b.data<scalar_t>(),
            masks_c.data<scalar_t>(),
            masks_size,
            masks_stride,
            grad_output.data<scalar_t>(),
            grad_output_size,
            grad_output_stride,
            grad_source_a.data<scalar_t>(),
            grad_source_b.data<scalar_t>(),
            grad_source_c.data<scalar_t>(),
            grad_source_size,
            grad_source_stride, 
            grad_flow_field_a.data<scalar_t>(),
            grad_flow_field_b.data<scalar_t>(),
            grad_flow_field_c.data<scalar_t>(),
            grad_flow_field_size,
            grad_flow_field_stride,    
            kernel_size);

    }));
    // TODO: Use the ATen equivalent to get last error

    //    THCudaCheck(cudaGetLastError());

}
