// preprocess.cu
#include <cuda_runtime.h>
#include <cstdint>

__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src, int srcW, int srcH, int srcStep,
    float* __restrict__ dst, int dstW, int dstH, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    int scaledW = (int)(srcW * scale);
    int scaledH = (int)(srcH * scale);

    int planeSize = dstW * dstH;
    int dstIdx = y * dstW + x;

    // Zero-pad areas outside the aspect-ratio scaled image
    if (x >= scaledW || y >= scaledH) {
        dst[dstIdx] = 0.0f;
        dst[dstIdx + planeSize] = 0.0f;
        dst[dstIdx + 2 * planeSize] = 0.0f;
        return;
    }

    // Nearest-neighbor coordinate mapping (or bilinear)
    int srcX = (int)(x / scale);
    int srcY = (int)(y / scale);
    srcX = srcX < srcW ? srcX : srcW - 1;
    srcY = srcY < srcH ? srcY : srcH - 1;

    const uint8_t* pixel = src + srcY * srcStep + srcX * 3;
    float b = (float)pixel[0];
    float g = (float)pixel[1];
    float r = (float)pixel[2];

    // ImageNet Normalization: (px / 255.0 - mean) / std
    // NCHW Layout: Plane 0 = R, Plane 1 = G, Plane 2 = B
    dst[dstIdx]                  = ((r / 255.0f) - 0.485f) / 0.229f; // R
    dst[dstIdx + planeSize]      = ((g / 255.0f) - 0.456f) / 0.224f; // G
    dst[dstIdx + 2 * planeSize]  = ((b / 255.0f) - 0.406f) / 0.225f; // B
}

void launchPreprocess(const uint8_t* d_src, int srcW, int srcH, int srcStep,
                      float* d_dst, int dstW, int dstH, cudaStream_t stream)
{
    float scale = (float)dstW / (srcW > srcH ? srcW : srcH);

    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        d_src, srcW, srcH, srcStep, d_dst, dstW, dstH, scale);
}

__global__ void upscale_mask_kernel(
    const float* __restrict__ src, int srcW, int srcH,
    float* __restrict__ dst, int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstW || y >= dstH) return;

    // Calculate source coordinates
    float srcX = x * ((float)srcW / dstW);
    float srcY = y * ((float)srcH / dstH);

    // Get integer coordinates for the 4 neighbors
    int x1 = (int)srcX;
    int y1 = (int)srcY;
    int x2 = min(x1 + 1, srcW - 1);
    int y2 = min(y1 + 1, srcH - 1);

    // Calculate interpolation weights
    float wx = srcX - x1;
    float wy = srcY - y1;

    // Read the 4 neighboring values from the source mask
    float v00 = src[y1 * srcW + x1];
    float v10 = src[y1 * srcW + x2];
    float v01 = src[y2 * srcW + x1];
    float v11 = src[y2 * srcW + x2];

    // Bilinear interpolation formula
    float val = (1.0f - wx) * (1.0f - wy) * v00 +
                wx * (1.0f - wy) * v10 +
                (1.0f - wx) * wy * v01 +
                wx * wy * v11;

    dst[y * dstW + x] = val;
}

void launchUpscaleMask(const float* d_src, int srcW, int srcH,
                       float* d_dst, int dstW, int dstH, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

    upscale_mask_kernel<<<grid, block, 0, stream>>>(
        d_src, srcW, srcH, d_dst, dstW, dstH);
}