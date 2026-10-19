#pragma once

#include <string>
#include <opencv2/cudawarping.hpp>
#include "trt_module.h"
#include "preprocess.cuh"

class NanoSam
{

public:

    NanoSam(string encoderPath, string decoderPath);

    ~NanoSam();

    Mat predict(Mat& image, vector<Point> points, vector<float> labels);
    Mat predictGpu(const cv::cuda::GpuMat& image, const std::vector<cv::Point>& points, const std::vector<float>& labels);
    void encodeImageGpu(const cv::cuda::GpuMat& gpuImage);

private:

    // Variables
    float* mFeatures;
    float* mMaskInput;
    float* mHasMaskInput;
    float* mIouPrediction;
    float* mLowResMasks;

    cv::Mat mLowResMask;

    TRTModule* mImageEncoder;
    TRTModule* mMaskDecoder;

    void upscaleMask(Mat& mask, int targetWidth, int targetHeight, int size = 256);
    void upscaleMaskGpu(cv::cuda::GpuMat& mask, int targetWidth, int targetHeight, int size = 256);
    Mat resizeImage(Mat& img, int modelWidth, int modelHeight);
    void prepareDecoderInput(vector<Point>& points, float* pointData, int numPoints, int imageWidth, int imageHeight);
    
    uint8_t* d_rawImage = nullptr;
    cudaStream_t mStream = nullptr;

};

