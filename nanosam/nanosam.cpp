#include "nanosam.h"
#include "config.h"

using namespace std;

// Constructor
NanoSam::NanoSam(string encoderPath, string decoderPath)
{
    mImageEncoder = new TRTModule(encoderPath,
        { "image" },
        { "image_embeddings" }, true, false);

    mMaskDecoder = new TRTModule(decoderPath,
        { "image_embeddings", "point_coords", "point_labels", "mask_input", "has_mask_input" },
        { "iou_predictions", "low_res_masks" }, true, false);

    mFeatures = new float[HIDDEN_DIM * FEATURE_WIDTH * FEATURE_HEIGHT];
    mMaskInput = new float[HIDDEN_DIM * HIDDEN_DIM];
    mHasMaskInput = new float;
    mIouPrediction = new float[NUM_LABELS];
    mLowResMasks = new float[NUM_LABELS * HIDDEN_DIM * HIDDEN_DIM];

    // Pre-allocate max prompt memory on GPU (e.g. 10 points)
    void* d_coords = mMaskDecoder->getGpuBuffer("point_coords");
    void* d_labels = mMaskDecoder->getGpuBuffer("point_labels");
    
    // Add stream creation
    cudaStreamCreate(&mStream);
}

// Deconstructor
NanoSam::~NanoSam()
{
    if (mFeatures)      delete[] mFeatures;
    if (mMaskInput)     delete[] mMaskInput;
    if (mIouPrediction) delete[] mIouPrediction;
    if (mLowResMasks)   delete[] mLowResMasks;

    if (mImageEncoder)  delete mImageEncoder;
    if (mMaskDecoder)   delete mMaskDecoder;
    
    if (d_rawImage) cudaFree(d_rawImage);
    if (mStream) cudaStreamDestroy(mStream);
}

// Perform inference using NanoSam models
Mat NanoSam::predict(Mat& image, vector<Point> points, vector<float> labels)
{
    if (points.size() == 0) return cv::Mat(image.rows, image.cols, CV_32FC1);

    // Preprocess encoder input
    auto resizedImage = resizeImage(image, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT);

    // Encoder Inference
    mImageEncoder->setInput(resizedImage);
    mImageEncoder->infer();
    mImageEncoder->getOutput(mFeatures);
    // Preprocess decoder input
    auto pointData = new float[2 * points.size()];
    prepareDecoderInput(points, pointData, points.size(), image.cols, image.rows);

    // Decoder Inference
    mMaskDecoder->setInput(mFeatures, pointData, labels.data(), mMaskInput, mHasMaskInput, points.size());
    mMaskDecoder->infer();
    mMaskDecoder->getOutput(mIouPrediction, mLowResMasks);

    // Postprocessing: Extract the best mask using the calculated pointer offset
    Mat imgMask(HIDDEN_DIM, HIDDEN_DIM, CV_32FC1, mLowResMasks);
    upscaleMask(imgMask, image.cols, image.rows);

    delete[] pointData;
    return imgMask;
}

void NanoSam::prepareDecoderInput(vector<Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight)
{
    float scale = MODEL_INPUT_WIDTH / max(imageWidth, imageHeight);

    for (int i = 0; i < numPoints; i++)
    {
        pointData[i * 2] = (float)points[i].x * scale;
        pointData[i * 2 + 1] = (float)points[i].y * scale;
    }

    for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++)
    {
        mMaskInput[i] = 0;
    }
    *mHasMaskInput = 0;
}

Mat NanoSam::resizeImage(Mat& img, int inputWidth, int inputHeight)
{
    int w, h;
    float aspectRatio = (float)img.cols / (float)img.rows;

    if (aspectRatio >= 1)
    {
        w = inputWidth;
        h = int(inputHeight / aspectRatio);
    }
    else
    {
        w = int(inputWidth * aspectRatio);
        h = inputHeight;
    }

    Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, INTER_LINEAR);
    Mat out(inputHeight, inputWidth, CV_8UC3, 0.0);
    re.copyTo(out(Rect(0, 0, re.cols, re.rows)));

    return out;
}

void NanoSam::upscaleMask(Mat& mask, int targetWidth, int targetHeight, int size)
{
    int limX, limY;
    if (targetWidth > targetHeight)
    {
        limX = size;
        limY = size * targetHeight / targetWidth;
    }
    else
    {
        limX = size * targetWidth / targetHeight;
        limY = size;
    }

    // FIX: Use a temporary Mat because cv::resize cannot execute in-place
    Mat tempMask;
    cv::resize(mask(Rect(0, 0, limX, limY)), tempMask, Size(targetWidth, targetHeight));
    mask = tempMask;
}

void NanoSam::encodeImageGpu(const cv::cuda::GpuMat& gpuImage)
{
    // // 1. Upload raw BGR image to GPU
    // size_t imgBytes = cpuImage.rows * cpuImage.step;
    // if (!d_rawImage) {
    //     cudaMalloc((void**)&d_rawImage, imgBytes);
    // }
    // cudaMemcpyAsync(d_rawImage, cpuImage.data, imgBytes, cudaMemcpyHostToDevice, mStream);
    // 
    // // 2. Preprocess directly on GPU into the Image Encoder's input buffer
    // float* d_encoderInput = (float*)mImageEncoder->getGpuBuffer("image");
    // launchPreprocess(d_rawImage, cpuImage.cols, cpuImage.rows, cpuImage.step,
    //                  d_encoderInput, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, mStream);
    // 
    // // 3. Run Encoder without downloading features to CPU
    // mImageEncoder->inferGpuOnly();

    // Preprocess directly on GPU into the Image Encoder's input buffer
    float* d_encoderInput = (float*)mImageEncoder->getGpuBuffer("image");
    
    // Use gpuImage.data to get the raw device pointer
    launchPreprocess((const uint8_t*)gpuImage.data, 
                     gpuImage.cols, gpuImage.rows, gpuImage.step,
                     d_encoderInput, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, mStream);

    // Run Encoder without downloading features to CPU
    mImageEncoder->inferGpuOnly();
}

cv::Mat NanoSam::predictGpu(const cv::cuda::GpuMat& image, const std::vector<cv::Point>& points, const std::vector<float>& labels)
{
    // FIX: Safe Device-to-Device copy avoids double-free memory crashes
    void* d_enc_emb = mImageEncoder->getGpuBuffer("image_embeddings");
    void* d_dec_emb = mMaskDecoder->getGpuBuffer("image_embeddings");
    constexpr int mem_dim = 1048576;// 256*64*64
    cudaMemcpyAsync(d_dec_emb, d_enc_emb, mem_dim * sizeof(float), cudaMemcpyDeviceToDevice, mStream);
    // mMaskDecoder->setTensorAddress("image_embeddings", d_enc_emb);

    // Scale points and copy coordinates to GPU
    float scale = (float)MODEL_INPUT_WIDTH / std::max(image.cols, image.rows);
    std::vector<float> coords(points.size() * 2);
    for (size_t i = 0; i < points.size(); ++i) {
        coords[i * 2]     = (float)points[i].x * scale;
        coords[i * 2 + 1] = (float)points[i].y * scale;
    }

    void* d_coords = mMaskDecoder->getGpuBuffer("point_coords");
    void* d_labels = mMaskDecoder->getGpuBuffer("point_labels");

    cudaMemcpyAsync(d_coords, coords.data(), sizeof(float) * coords.size(), cudaMemcpyHostToDevice, mStream);
    cudaMemcpyAsync(d_labels, labels.data(), sizeof(float) * labels.size(), cudaMemcpyHostToDevice, mStream);

    // FIX: Use explicit nvinfer1 namespace for dynamic shapes
    mMaskDecoder->setInputShape("point_coords", nvinfer1::Dims3{ 1, (int)points.size(), 2 });
    mMaskDecoder->setInputShape("point_labels", nvinfer1::Dims2{ 1, (int)points.size() });

    // Run Decoder
    mMaskDecoder->inferGpuOnly();

    // Copy only the final low-res mask back to CPU (or upscale on GPU)
    void* d_masks = mMaskDecoder->getGpuBuffer("low_res_masks");
    cv::Mat lowResMask(HIDDEN_DIM, HIDDEN_DIM, CV_32FC1);
    cudaMemcpyAsync(lowResMask.data, d_masks, sizeof(float) * HIDDEN_DIM * HIDDEN_DIM, cudaMemcpyDeviceToHost, mStream);
    cudaStreamSynchronize(mStream);

    // Upscale to original size
    upscaleMask(lowResMask, image.cols, image.rows);
    return lowResMask;
}