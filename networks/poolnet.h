#ifndef POOLNET_H_
#define POOLNET_H_

#include "deeplab_resnet.h"

/* ConvertLayer */
class ConvertLayerImpl : public torch::nn::Module {
public:
    ConvertLayerImpl();
    std::vector<torch::Tensor> forward(std::vector<torch::Tensor> x);
    torch::nn::ModuleList _make_convertlayer();
private:
    int64_t inplanes[5] = { 64, 256, 512, 1024, 2048 };
    int64_t planes[5] = { 128, 256, 256, 512, 512 };
    torch::nn::ModuleList convert0;
};
TORCH_MODULE(ConvertLayer);

/* DeepPoolLayer */
class DeepPoolLayerImpl : public torch::nn::Module {
public:
    DeepPoolLayerImpl(int64_t inplanes_, int64_t planes_, bool need_x2_, bool need_fuse_);
    torch::Tensor forward(torch::Tensor x, 
                          torch::Tensor x2 = torch::zeros({0}), 
                          torch::Tensor x3 = torch::zeros({0}));
    torch::nn::ModuleList _make_pools_layer();
    torch::nn::ModuleList _make_convs_layer();
private:
    int64_t inplanes, planes;
    bool need_x2, need_fuse;
    int64_t pool_sizes[3] =  { 2, 4, 8 };
    torch::nn::ModuleList pools, convs;
    torch::nn::Conv2d conv_sum, conv_sum_c;
};
TORCH_MODULE(DeepPoolLayer);

/* ScoreLayer */
class ScoreLayerImpl : public torch::nn::Module {
public:
    ScoreLayerImpl(int64_t inplanes=128);
    torch::Tensor forward(torch::Tensor x, c10::IntArrayRef x_size);
private:
    torch::nn::Conv2d score;
};
TORCH_MODULE(ScoreLayer);

/* PoolNet */
class PoolNetImpl : public torch::nn::Module {
public:
    PoolNetImpl();
    torch::Tensor forward(torch::Tensor x);
    torch::nn::ModuleList _make_deeppool_layers();
private:
    ResNet_locate base;
    torch::nn::ModuleList deep_pool;
    ScoreLayer score;
    ConvertLayer convert;
};
TORCH_MODULE(PoolNet);

#endif // POOLNET_H_