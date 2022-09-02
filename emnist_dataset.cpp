#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/types_c.h"
#include "torch/torch.h"
#include "torch/data/datasets/base.h"
//#include "torchvision/vision.h"

using namespace cv;
using namespace std;
using namespace torch;
//using namespace vision;
#include "emnist_dataset.h"

/*
###emnist-balanced-train-labels-idx1-ubyte###
[offset]    [type]      [value]            [description]
0000        32 bits     0x00000801(2049)   magic number         - big endian
0004        32 bits     0x0001b8a0(112800) number of items      - big endian
0008        unsigned    byte   ??          label
0009        unsigned    byte   ??          label
....
xxxx        unsigned    byte   ??          label
labels  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
classes = 47

###emnist-balanced-train-images-idx3-ubyte###
[offset] [type]          [value]            [description]
0000     32 bit integer  0x00000803(2051)   magic number        - big endian
0004     32 bit integer  0x0001b8a0(112800) number of images    - big endian
0008     32 bit integer  0x0000001c(28)     number of rows      - big endian
0012     32 bit integer  0x0000001c(28)     number of columns   - big endian
0016     unsigned byte   ??                 pixel
0017     unsigned byte   ??                 pixel
....
xxxx     unsigned byte   ??                 pixel
Pixels are arranged in row-wise.
Pixel value between 0 to 255.
0   = background white.
255 = foreground black.
The original EMNIST images provided are inverted horizontally and rotated 90 anti-clockwise
*/

EminstDataset::EminstDataset()
{
    char                    rawbyte;
    uint32_t                size_label;
    uint32_t                size_image;
    ifstream                if_label;
    ifstream                if_image;
    //char                  letters[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";

    if_label.open("./data/emnist-balanced-train-labels-idx1-ubyte", ios::in | ios::binary); // Binary label file
    if_image.open("./data/emnist-balanced-train-images-idx3-ubyte", ios::in | ios::binary); // Binary image file

    //skip magic
    for (int i = 0; i < 4; i++)
        if_label.read(&rawbyte, 1);

    //read number of items
    if_label.read((char*)&size_label, 4);
    size_label = __builtin_bswap32(size_label);

    for (uint32_t i = 0; i < size_label; i++)
    {
        if_label.read(&rawbyte, 1);
        //torch::Tensor tensor_label = torch::tensor({rawbyte}, torch::kInt32);
        //cout << tensor_label << endl;
        m_class_labels.push_back(rawbyte);
    }

    //skip magic
    for (uint32_t i = 0; i < 4; i++)
        if_image.read(&rawbyte, 1);

    //read number of images
    if_image.read((char*)&size_image, 4);
    size_image = __builtin_bswap32(size_image);

    //skip row 0x0000001c and col 0x0000001c
    for (uint32_t i = 0; i < 8; i++)
        if_image.read(&rawbyte, 1);

    uint8_t* imgbuf = new uint8_t[28*28];
    //for (uint32_t i = 0; i < 100; i++)
    for (uint32_t i = 0; i < size_image; i++)
    {
        if_image.read((char*)imgbuf, 28*28);

#pragma omp parallel for //speed up this for
        for (int k = 0; k < 28*28; k++)
            imgbuf[k] = 255 - imgbuf[k];

        cv::Mat src = cv::Mat(28, 28, CV_8UC1, imgbuf);
        cv::Mat img_flip;
        cv::Mat img_rotate;

        cv::flip(src, img_flip, 0); //0=x-axis
        cv::rotate(img_flip, img_rotate, cv::ROTATE_90_CLOCKWISE);

        /*char    buf[64]     = {0};
        char    letters[]   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt";
        cv::Mat img_resize;
        cv::resize(img_rotate, img_resize, cv::Size(640, 480));
        sprintf(buf,"letter: [%c]", letters[m_class_labels[i]]);
        cv::imshow(buf, img_resize);
        cv::waitKey(0);*/

        //Transform(img_rotate); //test

        m_class_images.push_back(std::move(img_rotate));
    }
    delete imgbuf;
}

EminstDataset::~EminstDataset()
{
    m_class_labels.clear();
    m_class_images.clear();
}

//single channel
torch::Tensor EminstDataset::Transform(cv::Mat cv_image)
{
    torch::Tensor tensor_image;

    cv_image.convertTo(cv_image, CV_32F, 1.0f / 255.0f);
    //cout << cv_image << endl;

    tensor_image     = torch::from_blob(cv_image.data, { cv_image.rows, cv_image.cols, cv_image.channels() }, torch::kFloat32);
    tensor_image     = tensor_image.permute({2, 0, 1});
    /*for(uint32_t k=0; k<28; k++)
        cout << tensor_image[0][k] << endl;*/

    tensor_image    -= 0.5f;
    /*for(uint32_t k=0; k<28; k++)
        cout << tensor_image[0][k] << endl;*/

    return tensor_image;
}

/*
 ~/pytorch/torch/csrc/api/include/torch/data/example.h
 template <typename Data = Tensor, typename Target = Tensor>
*/
torch::data::Example<>EminstDataset::get(size_t index)
{
    char            cv_label;
    cv::Mat         cv_image;
    torch::Tensor   tensor_label,tensor_image;

    cv_label = m_class_labels[index];
    cv_image = m_class_images[index];

    tensor_label = torch::tensor({cv_label}, at::kChar);
    tensor_image = Transform(cv_image);

    /*cout << tensor_image.sizes() << endl;
    cout << tensor_label.sizes() << endl;*/

    return {tensor_image, tensor_label};
}

optional<size_t> EminstDataset::size() const
{
    return m_class_images.size();
}

int main(int argc, char *argv[])
{
    size_t  batch_size      = 1;
    auto train_dataset      = EminstDataset().map(torch::data::transforms::Stack<>());
    auto train_dataloader   = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_dataset), batch_size);

    for (auto& batch : *train_dataloader)
    {
        torch::Tensor image = batch.data;
        torch::Tensor label = batch.target;

        /*cout << image.sizes() << endl;
        cout << label.sizes() << endl;*/
    }

    torch::NoGradGuard guard_release; //release torch memory

    return EXIT_SUCCESS;
}
