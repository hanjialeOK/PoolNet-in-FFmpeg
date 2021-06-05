#ifndef DEEPLAB_RESNET_H_
#define DEEPLAB_RESNET_H_

#include <torch/script.h>
#include <torch/torch.h>

/* BottleNeck */
class BottleNeckImpl : public torch::nn::Module {
public:
    BottleNeckImpl(int64_t inplanes_,   int64_t planes_, 
                   int64_t stride_ = 1, int64_t dilation_ = 1, 
                   torch::nn::Sequential downsample_ = torch::nn::Sequential());
    torch::Tensor forward(torch::Tensor x);
private:
    int64_t inplanes, planes, stride, dilation;
    torch::nn::Sequential downsample;
    torch::nn::Conv2d conv1,conv2, conv3;
    torch::nn::BatchNorm2d bn1, bn2, bn3;
};
TORCH_MODULE(BottleNeck);

const int64_t expansion = 4;

/* ResNet */
class ResNetImpl : public torch::nn::Module {
public:
    ResNetImpl(std::vector<int> layers);
    std::vector<torch::Tensor> forward(torch::Tensor x);
    torch::nn::Sequential _make_layer(int64_t planes,     int64_t blocks, 
                                      int64_t stride = 1, int64_t dilation = 1);
private:
	int64_t inplanes = 64;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
};
TORCH_MODULE(ResNet);

/* ResNet_locate */
class ResNet_locateImpl : public torch::nn::Module {
public:
    ResNet_locateImpl(std::vector<int> layers);
    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
        forward(torch::Tensor x);
    torch::nn::ModuleList _make_ppms_layer();
    torch::nn::ModuleList _make_infos_layer();
private:
    int64_t inplanes = 512;
    int64_t planes_list[4] = { 512, 256, 256, 128 };
    int64_t pool_sizes[3] = { 1, 3, 5 };
    ResNet resnet;
    torch::nn::Conv2d ppms_pre;
    torch::nn::ModuleList ppms, infos;
    torch::nn::Sequential ppm_cat;
};
TORCH_MODULE(ResNet_locate);

ResNet_locate resnet50();

#endif // DEEPLAB_RESNET_H_