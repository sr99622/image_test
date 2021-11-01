#include "imageframe.h"

ImageFrame::ImageFrame()
{

}

cv::Mat ImageFrame::getCrop(const cv::Mat& image, Bbox* box) const
{
    cv::Mat crop;
    float target_aspect = crop_width / (float)crop_height;
    float new_width = target_aspect * box->h;
    box->x1 -= (new_width - box->w) / 2;
    box->w = new_width;
    box->setX2Y2();

    box->x1 = std::max(box->x1, 0.0f);
    box->y1 = std::max(box->y1, 0.0f);
    box->x2 = std::min(box->x2, (float)image.cols - 1);
    box->y2 = std::min(box->y2, (float)image.rows - 1);
    box->setWH();

    crop = image(cv::Range((int)box->y1, (int)box->y2), cv::Range((int)box->x1, (int)box->x2));
    cv::resize(crop, crop, cv::Size(crop_width, crop_height));

    return crop;
}

void ImageFrame::clear()
{
    for (int i = 0; i < features.size(); i++)
        features[i].clear();
    features.clear();
    crops.clear();
    detections.clear();
}

ImageFrame::~ImageFrame()
{
    clear();
}

void Bbox::setX2Y2()
{
    x2 = x1 + w;
    y2 = y1 + h;
}

void Bbox::setWH()
{
    w = x2 - x1;
    h = y2 - y1;
}
