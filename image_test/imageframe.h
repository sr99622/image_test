#ifndef IMAGEFRAME_H
#define IMAGEFRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>

#define num_channels 3
#define feature_size 128
#define crop_width 64
#define crop_height 128

class Bbox
{

public:
    void setX2Y2();
    void setWH();

    float confidence;

    float x1;
    float y1;
    float x2;
    float y2;
    float w;
    float h;
};

class ImageFrame
{

public:
    ImageFrame();
    ~ImageFrame();
    void clear();
    cv::Mat getCrop(const cv::Mat& image, Bbox* box) const;

    std::vector<Bbox> detections;
    std::vector<cv::Mat> crops;
    std::vector<std::vector<float>> features;

};

#endif // IMAGEFRAME_H
