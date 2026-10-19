#include "trt_module.h"
#include "logging.h"
#include "cuda_utils.h"
#include "config.h"
#include "macros.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

using namespace nvinfer1;
using namespace std;
static Logger gLogger;


std::string getFileExtension(const std::string& filePath) {
    size_t dotPos = filePath.find_last_of(".");
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos + 1);
    }
    return ""; // No extension found
}

TRTModule::TRTModule(string modelPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    if (getFileExtension(modelPath) == "onnx")
    {
        cout << "Building Engine from " << modelPath << endl;
        build(modelPath, inputNames, outputNames, isDynamicShape, isFP16);
    }
    else
    {
        cout << "Deserializing Engine." << endl;
        deserializeEngine(modelPath, inputNames, outputNames);
    }
}

TRTModule::~TRTModule()
{
    // Release stream and buffers
    cudaStreamDestroy(mCudaStream);
    for (int i = 0; i < mGpuBuffers.size(); i++)
        CUDA_CHECK(cudaFree(mGpuBuffers[i]));
    for (int i = 0; i < mCpuBuffers.size(); i++)    
        delete[] mCpuBuffers[i];
    
    // Destroy the engine
    delete mContext;
    delete mEngine;
    delete mRuntime;
}

void TRTModule::build(string onnxPath, vector<string> inputNames, vector<string> outputNames, bool isDynamicShape, bool isFP16)
{
    auto builder = createInferBuilder(gLogger);
    assert(builder != nullptr);

    INetworkDefinition* network = builder->createNetworkV2(0);
    assert(network != nullptr);

    IBuilderConfig* config = builder->createBuilderConfig();
    assert(config != nullptr);

    if (isDynamicShape) // Only designed for NanoSAM mask decoder
    {
        auto profile = builder->createOptimizationProfile();

        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMIN, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kOPT, Dims3{ 1, 1, 2 });
        profile->setDimensions(inputNames[1].c_str(), OptProfileSelector::kMAX, Dims3{ 1, 10, 2 });

        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMIN, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kOPT, Dims2{ 1, 1 });
        profile->setDimensions(inputNames[2].c_str(), OptProfileSelector::kMAX, Dims2{ 1, 10 });

        config->addOptimizationProfile(profile);
    }

    if (isFP16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
    assert(parser != nullptr);

    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    assert(parsed);


    // CUDA stream used for profiling by the builder.
    assert(mCudaStream != nullptr);

    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };
    assert(plan != nullptr);

    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime != nullptr);

    mEngine = mRuntime->deserializeCudaEngine(plan->data(), plan->size());
    assert(mEngine != nullptr);

    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);

    delete network;
    delete config;
    delete parser;
    delete plan;

    initialize(inputNames, outputNames);
}

void TRTModule::deserializeEngine(string engine_name, vector<string> inputNames, vector<string> outputNames)
{
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serializedEngine = new char[size];
    assert(serializedEngine);
    file.read(serializedEngine, size);
    file.close();

    mRuntime = createInferRuntime(gLogger);
    assert(mRuntime);
    mEngine = mRuntime->deserializeCudaEngine(serializedEngine, size);
    assert(mEngine != nullptr);
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);
    delete[] serializedEngine;

    // TensorRT 10 replaces bindings with IO Tensors
    assert(mEngine->getNbIOTensors() == inputNames.size() + outputNames.size()); 

    initialize(inputNames, outputNames);
}

void TRTModule::initialize(vector<string> inputNames, vector<string> outputNames)
{
    int numTensors = mEngine->getNbIOTensors();
    
    mGpuBuffers.resize(numTensors);
    mCpuBuffers.resize(numTensors);

    for (int i = 0; i < numTensors; ++i)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        size_t binding_size = getSizeByDim(mEngine->getTensorShape(tensorName));
        
        mBufferBindingSizes.push_back(binding_size);
        mBufferBindingBytes.push_back(binding_size * sizeof(float));

        mCpuBuffers[i] = new float[binding_size];
        cudaMalloc(&mGpuBuffers[i], mBufferBindingBytes[i]);

        if (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT)
        {
            mInputDims.push_back(mEngine->getTensorShape(tensorName));
        }
        else
        {
            mOutputDims.push_back(mEngine->getTensorShape(tensorName));
        }
    }

    CUDA_CHECK(cudaStreamCreate(&mCudaStream));
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
bool TRTModule::infer()
{
    // Memcpy from host input buffers to device input buffers
    copyInputToDeviceAsync(mCudaStream);

    // TensorRT 10 Execution Setup
    for (int i = 0; i < mEngine->getNbIOTensors(); i++)
    {
        mContext->setTensorAddress(mEngine->getIOTensorName(i), mGpuBuffers[i]);
    }

    bool status = mContext->enqueueV3(mCudaStream);

    if (!status)
    {
        cout << "inference error!" << endl;
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    copyOutputToHostAsync(mCudaStream);

    return true;
}

void TRTModule::copyInputToDeviceAsync(const cudaStream_t& stream)
{
    memcpyBuffers(true, false, true, stream);
}

void TRTModule::copyOutputToHostAsync(const cudaStream_t& stream)
{
    memcpyBuffers(false, true, true, stream);
}

void TRTModule::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream)
{
    for (int i = 0; i < mEngine->getNbIOTensors(); i++)
    {
        const char* tensorName = mEngine->getIOTensorName(i);
        void* dstPtr = deviceToHost ? mCpuBuffers[i] : mGpuBuffers[i];
        const void* srcPtr = deviceToHost ? mGpuBuffers[i] : mCpuBuffers[i];
        const size_t byteSize = mBufferBindingBytes[i];
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;

        bool isInput = (mEngine->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT);

        if ((copyInput && isInput) || (!copyInput && !isInput))
        {
            if (async)
            {
                CUDA_CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
            }
            else
            {
                CUDA_CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
            }
        }
    }
}

size_t TRTModule::getSizeByDim(const Dims& dims)
{
    size_t size = 1;
    
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] == -1)
            size *= MAX_NUM_PROMPTS;
        else
            size *= dims.d[i];
    }

    return size;
}

void TRTModule::setInput(Mat& image)
{
    int idx = 0;
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
        if (string(mEngine->getIOTensorName(i)) == "image") { 
            idx = i; 
            break; 
        }
    }

    int H = image.rows;
    int W = image.cols;
    int channelSize = H * W;

    // 1. Split BGR channels (bgr[0]=B, bgr[1]=G, bgr[2]=R)
    std::vector<cv::Mat> bgr;
    cv::split(image, bgr);

    // 2. Wrap destination buffers directly to avoid copying memory
    // Layout expected by ResNet encoder: NCHW (R, G, B)
    cv::Mat outR(H, W, CV_32FC1, mCpuBuffers[idx]);
    cv::Mat outG(H, W, CV_32FC1, mCpuBuffers[idx] + channelSize);
    cv::Mat outB(H, W, CV_32FC1, mCpuBuffers[idx] + 2 * channelSize);

    // 3. Exact ImageNet formula: (pixel / 255.0 - mean) / std
    // Rewritten as: pixel * (1.0 / (255.0 * std)) - (mean / std)
    // convertTo utilizes ARM NEON vectorized instructions on Jetson Orin (~1-2 ms)
    bgr[2].convertTo(outR, CV_32FC1, 1.0f / (255.0f * 0.229f), -0.485f / 0.229f); // R
    bgr[1].convertTo(outG, CV_32FC1, 1.0f / (255.0f * 0.224f), -0.456f / 0.224f); // G
    bgr[0].convertTo(outB, CV_32FC1, 1.0f / (255.0f * 0.225f), -0.406f / 0.225f); // B
}

// Set dynamic input
void TRTModule::setInput(float* features, float* imagePointCoords, float* imagePointLabels, float* maskInput, float* hasMaskInput, int numPoints)
{
    int id_feat = -1, id_coords = -1, id_labels = -1, id_mask = -1, id_has_mask = -1;
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
        string name = mEngine->getIOTensorName(i);
        if (name == "image_embeddings") id_feat = i;
        else if (name == "point_coords") id_coords = i;
        else if (name == "point_labels") id_labels = i;
        else if (name == "mask_input") id_mask = i;
        else if (name == "has_mask_input") id_has_mask = i;
    }

    delete[] mCpuBuffers[id_coords];
    delete[] mCpuBuffers[id_labels];
    mCpuBuffers[id_coords] = new float[numPoints * 2];
    mCpuBuffers[id_labels] = new float[numPoints];

    mBufferBindingBytes[id_coords] = sizeof(float) * numPoints * 2;
    mBufferBindingBytes[id_labels] = sizeof(float) * numPoints;

    // Copy only the active coordinates into pre-allocated memory
    memcpy(mCpuBuffers[id_feat], features, mBufferBindingBytes[id_feat]);
    memcpy(mCpuBuffers[id_coords], imagePointCoords, sizeof(float) * numPoints * 2);
    memcpy(mCpuBuffers[id_labels], imagePointLabels, sizeof(float) * numPoints);
    memcpy(mCpuBuffers[id_mask], maskInput, mBufferBindingBytes[id_mask]);
    memcpy(mCpuBuffers[id_has_mask], hasMaskInput, mBufferBindingBytes[id_has_mask]);

    // Update TensorRT dynamic input shape
    mContext->setInputShape("point_coords", Dims3{ 1, numPoints, 2 });
    mContext->setInputShape("point_labels", Dims2{ 1, numPoints });
}

void TRTModule::getOutput(float* features)
{
    int idx = 1; // default fallback
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
        if (string(mEngine->getIOTensorName(i)) == "image_embeddings") { idx = i; break; }
    }
    memcpy(features, mCpuBuffers[idx], mBufferBindingBytes[idx]);
}

void TRTModule::getOutput(float* iouPrediction, float* lowResolutionMasks)
{    
    int id_iou = -1, id_masks = -1;
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
        string name = mEngine->getIOTensorName(i);
        if (name == "iou_predictions") id_iou = i;
        else if (name == "low_res_masks") id_masks = i;
    }
    
    memcpy(lowResolutionMasks, mCpuBuffers[id_masks], mBufferBindingBytes[id_masks]);
    memcpy(iouPrediction, mCpuBuffers[id_iou], mBufferBindingBytes[id_iou]);
}

void* TRTModule::getGpuBuffer(const std::string& tensorName)
{
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
        if (mEngine->getIOTensorName(i) == tensorName) {
            return mGpuBuffers[i];
        }
    }
    return nullptr;
}

// Executes inference directly on GPU without host <-> device memcpys
bool TRTModule::inferGpuOnly()
{
    for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
        mContext->setTensorAddress(mEngine->getIOTensorName(i), mGpuBuffers[i]);
    }
    return mContext->enqueueV3(mCudaStream);
}

void TRTModule::setInputShape(const std::string& tensorName, nvinfer1::Dims dims)
{
    mContext->setInputShape(tensorName.c_str(), dims);
}

