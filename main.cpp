#include "nanosam/nanosam.h"
#include "utils.h"
#include <chrono>
#include <vector>
#include <numeric>

void segmentClickedPoint(NanoSam& nanosam, string imagePath) {

    auto image = imread(imagePath);

    // Create a window to display the image
    cv::namedWindow("Image");

    // Data structure to hold clicked point
    PointData pointData;
    pointData.clicked = false;
    cv::Mat clonedImage = image.clone();
    int clickCount = 0;

    // Loop until Esc key is pressed
    while (true)
    {
        // Display the original image
        cv::imshow("Image", clonedImage);

        if (pointData.clicked)
        {
            pointData.clicked = false; // Reset clicked flag            

            auto mask = nanosam.predict(image, { pointData.point }, { 1.0f });

            cv::circle(clonedImage, pointData.point, 5, cv::Scalar(0, 0, 255), -1);

            if (clickCount >= CITYSCAPES_COLORS.size()) clickCount = 0;
            overlay(image, mask, CITYSCAPES_COLORS[clickCount * 9]);
            clickCount++;
        }

        // Set the callback function for mouse events on the displayed cloned image
        cv::setMouseCallback("Image", onMouse, &pointData);

        // Check for Esc key press
        char key = cv::waitKey(1);
        if (key == 27) // ASCII code for Esc key
        {
            clonedImage = image.clone();
        }
    }
    cv::destroyAllWindows();
}

void segmentBbox(NanoSam& nanosam, string imagePath, string outputPath, vector<Point> bbox)
{
    auto image = imread(imagePath);

    // 2 : Bounding box top-left, 3 : Bounding box bottom-right
    vector<float> labels = { 2, 3 }; 

    auto mask = nanosam.predict(image, bbox, labels);

    overlay(image, mask);

    rectangle(image, bbox[0], bbox[1], cv::Scalar(255, 255, 0), 3);

    imwrite(outputPath, image);
}

void segmentWithPoint(NanoSam& nanosam, string imagePath, string outputPath, Point promptPoint, 
                      double* predictDurationMs = nullptr)
{
    auto image = imread(imagePath);
    
    // 1 : Foreground
    vector<float> labels = { 1.0f }; 

    // Mask returns as a CV_32FC1 matrix of raw float logits
    auto start = std::chrono::high_resolution_clock::now();
    //auto mask = nanosam.predict(image, { promptPoint }, labels);
    nanosam.encodeImageGpu(image);
    auto mask = nanosam.predictGpu(image, { promptPoint }, labels);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (predictDurationMs != nullptr)
    {
        *predictDurationMs = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Pass the proper 8-bit mask to overlay
    overlay(image, mask);
    
    imwrite(outputPath, image);
}

int main()
{
    /* 1. Load engine examples */

    // Option 1: Load the engines
    // NanoSam nanosam("/home/admin/Downloads/nanosam/resnet18_image_encoder_fp16_opt_5_avg_100.engine",  "/home/admin/Downloads/nanosam/mobile_sam_mask_decoder.engine");
    NanoSam nanosam("/home/admin/Downloads/nanosam/resnet18_image_encoder_int8.engine",  "/home/admin/Downloads/nanosam/mobile_sam_mask_decoder.engine");

    // Option 2: Build the engines from onnx files
    // NanoSam nanosam("/home/admin/Downloads/nanosam/resnet18_image_encoder.onnx", "/home/admin/Downloads/nanosam/mobile_sam_mask_decoder.onnx");
    
    const int num_iterations = 2;
    std::vector<double> predict_latencies;
    predict_latencies.reserve(num_iterations);

    double current_predict_ms = 0.0;

    // 1. Warm-up run (ignores disk I/O and GPU compilation overhead)
    std::cout << "Warming up..." << std::endl;
    segmentWithPoint(nanosam, "assets/dog.jpg", "assets/dog_mask.jpg", Point(1300, 900), &current_predict_ms);
    std::cout << "Warm-up predict time: " << current_predict_ms << " ms\n" << std::endl;

    // 2. Benchmark loop
    std::cout << "Benchmarking predict over " << num_iterations << " iterations..." << std::endl;
    for (int i = 0; i < num_iterations; ++i)
    {
        segmentWithPoint(nanosam, "assets/dog.jpg", "assets/dog_mask.jpg", Point(1300, 900), &current_predict_ms);
        predict_latencies.push_back(current_predict_ms);
        std::cout << current_predict_ms << " ";
    }
    std::cout << std::endl;

    // 3. Compute statistics
    double total_time = std::accumulate(predict_latencies.begin(), predict_latencies.end(), 0.0);
    double avg_time = total_time / num_iterations;

    std::cout << "\n--- NanoSam Predict Results ---" << std::endl;
    std::cout << "Average predict time: " << avg_time << " ms" << std::endl;
    // Demo 2: Segment using a bounding box
    //segmentBbox(nanosam, "assets/dogs.jpg", "assets/dogs_mask.jpg", { Point(100, 100), Point(750, 759) });

    // Demo 3: Segment the clicked object
    //segmentClickedPoint(nanosam, "assets/dogs.jpg");

    return 0;
}
