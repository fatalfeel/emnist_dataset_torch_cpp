#ifndef IC_RECOGNIZE_H
#define IC_RECOGNIZE_H

class EminstDataset : public torch::data::Dataset<EminstDataset>
{
public:
    EminstDataset();
    ~EminstDataset();

    torch::data::Example<> get(size_t index) override;
    optional<size_t> size() const override;
    torch::Tensor Transform(cv::Mat cv_image);

private:
    std::vector<char>       m_class_labels;
    std::vector<cv::Mat>    m_class_images;
};

#endif //SWIFTPR_FASTDESKEW_H
