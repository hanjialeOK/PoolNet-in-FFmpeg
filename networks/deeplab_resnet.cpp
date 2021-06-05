#include "deeplab_resnet.h"

#include <iostream>

/* BottleNeck */
BottleNeckImpl::BottleNeckImpl(int64_t inplanes_, int64_t planes_, int64_t stride_, 
                               int64_t dilation_, torch::nn::Sequential downsample_)
    : inplanes(inplanes_),
      planes(planes_),
      stride(stride_),
      dilation(dilation_),
      downsample(downsample_),
      conv1(torch::nn::Conv2dOptions(/*in_channels=*/ inplanes, 
                                     /*out_channels=*/planes, 
                                     /*kernal_size=*/ 1)
                                     .stride(stride).padding(0).bias(false)),
      bn1(torch::nn::BatchNorm2dOptions(planes).affine(true)),
      conv2(torch::nn::Conv2dOptions(/*in_channels=*/ planes, 
                                     /*out_channels=*/planes,
                                     /*kernal_size=*/ 3)
                                     .stride(1).padding(dilation).bias(false).dilation(dilation)),
      bn2(torch::nn::BatchNorm2dOptions(planes).affine(true)),
      conv3(torch::nn::Conv2dOptions(/*in_channels=*/ planes, 
                                     /*out_channels=*/planes * expansion, 
                                     /*kernal_size=*/ 1)
                                     .stride(1).padding(0).bias(false)),
      bn3(torch::nn::BatchNorm2dOptions(planes * expansion).affine(true)) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    for (const auto& p : bn1->parameters()) {
        p.requires_grad_(false);
    }
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    for (const auto& p : bn2->parameters()) {
        p.requires_grad_(false);
    }
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    for (const auto& p : bn3->parameters()) {
        p.requires_grad_(false);
    }

    if (!downsample->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor BottleNeckImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);
    x = torch::relu(x);

    x = conv3->forward(x);
    x = bn3->forward(x);

    if (!downsample->is_empty()){
        residual = downsample->forward(residual);
    }
    x += residual;
    x = torch::relu(x);

    return x;
}

/* ResNet */
ResNetImpl::ResNetImpl(std::vector<int> layers) 
    : conv1(torch::nn::Conv2dOptions(/*in_channels=*/ 3, 
                                     /*out_channels=*/64, 
                                     /*kernal_size=*/ 7)
                                     .stride(2).padding(3).bias(false)),
      bn1(torch::nn::BatchNorm2dOptions(64).affine(true)),
      layer1(_make_layer(/*planes=*/64,  /*blocks=*/layers[0])),
      layer2(_make_layer(/*planes=*/128, /*blocks=*/layers[1], /*stride=*/2)),
      layer3(_make_layer(/*planes=*/256, /*blocks=*/layers[2], /*stride=*/2)),
      layer4(_make_layer(/*planes=*/512, /*blocks=*/layers[3], /*stride=*/1, /*dilation=*/2)) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    for (const auto& p : bn1->parameters()) {
        p.requires_grad_(false);
    }
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);

    /* Initilize weights */
    for (const auto& m : this->modules(/*include_self=*/false)) {
        if (auto* conv = m->as<torch::nn::Conv2d>()) {
            conv->weight.data().normal_(0, 0.01);
        }
        else if (auto* bn = m->as<torch::nn::BatchNorm2d>()) {
            bn->weight.data().fill_(1);
            bn->bias.data().zero_();
        }
    }
}

std::vector<torch::Tensor> ResNetImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> tmp_x;
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    tmp_x.push_back(x);
    x = torch::max_pool2d(/*tensor=*/x, /*kernel_size=*/3, /*stride=*/2, 
                          /*padding=*/1, /*dilation=*/1, /*ceil_mode=*/true);

    x = layer1->forward(x);
    tmp_x.push_back(x);
    x = layer2->forward(x);
    tmp_x.push_back(x);
    x = layer3->forward(x);
    tmp_x.push_back(x);
    x = layer4->forward(x);
    tmp_x.push_back(x);

    return tmp_x;
}

torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, 
                                              int64_t stride, int64_t dilation) {
    torch::nn::Sequential downsample;
    if (stride != 1 || inplanes != planes * expansion || dilation == 2 || dilation == 4) {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(
                /*in_channels=*/ inplanes, 
                /*out_channels=*/planes * expansion, 
                /*kernel_size=*/ 1)
                .stride(stride).padding(0).bias(false)),
            torch::nn::BatchNorm2d(
                torch::nn::BatchNorm2dOptions(planes * expansion).affine(true)));
    }
    for (const auto& p : downsample[1]->parameters()) {
        p.requires_grad_(false);
    }
    torch::nn::Sequential layers;
    layers->push_back(BottleNeck(inplanes, planes, stride, dilation, downsample));
    inplanes = planes * expansion;
    for (int64_t i = 1; i < blocks; i++) {
        layers->push_back(BottleNeck(inplanes, planes, /*stride=*/1, dilation));
    }

    return layers;
}

/* ResNet_locate */
ResNet_locateImpl::ResNet_locateImpl(std::vector<int> layers) 
    : resnet(layers),
      ppms_pre(torch::nn::Conv2dOptions(/*in_channels=*/ 2048, 
                                        /*out_channels=*/inplanes, 
                                        /*kernal_size=*/ 1)
                                        .stride(1).padding(0).bias(false)),
      ppms(_make_ppms_layer()),
      ppm_cat(
          torch::nn::Sequential(
              torch::nn::Conv2d(torch::nn::Conv2dOptions(
                  /*in_channels=*/ 2048, 
                  /*out_channels=*/inplanes, 
                  /*kernel_size=*/ 3)
                  .stride(1).padding(1).bias(false)),
              torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true)))),
      infos(_make_infos_layer()) {
    register_module("resnet", resnet);
    register_module("ppms_pre", ppms_pre);
    register_module("ppms", ppms);
    register_module("ppm_cat", ppm_cat);
    register_module("infos", infos);

    /* Initilize weights */
    for (const auto& m : this->modules(/*include_self=*/false)) {
        if (auto* conv = m->as<torch::nn::Conv2d>()) {
            conv->weight.data().normal_(0, 0.01);
        }
        else if (auto* bn = m->as<torch::nn::BatchNorm2d>()) {
            bn->weight.data().fill_(1);
            bn->bias.data().zero_();
        }
    }
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
ResNet_locateImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> tmp_x = resnet->forward(x);
    /* y.sizes() : { 1, 512, 24, 32 } */
    torch::Tensor y = ppms_pre->forward(tmp_x.back());

    std::vector<torch::Tensor> xls{ y };
    for (int i = 0; i < ppms->size(); i++) {
        xls.push_back(
            torch::nn::functional::interpolate(
                /*input=*/  ppms[i]->as<torch::nn::Sequential>()->forward(y), 
                /*options=*/torch::nn::functional::InterpolateFuncOptions()
                                .size(std::vector<int64_t>({y.size(2), y.size(3)}))
                                .mode(torch::kBilinear)
                                .align_corners(true)));
    }
    /* z.sizes() : { 1, 2048, 24, 32 } */
    torch::Tensor z = ppm_cat->forward(torch::cat(/*TensorList=*/xls, /*dim=*/1));

    std::vector<torch::Tensor> infos_out;
    for (int i = 0; i < infos->size(); i++) {
        c10::IntArrayRef size = tmp_x[infos->size() - 1 - i].sizes();
        infos_out.push_back(
            infos[i]->as<torch::nn::Sequential>()->forward(
                torch::nn::functional::interpolate(
                    /*input=*/  z, 
                    /*options=*/torch::nn::functional::InterpolateFuncOptions()
                                    .size(std::vector<int64_t>({size[2], size[3]}))
                                    .mode(torch::kBilinear)
                                    .align_corners(true))));
    }
    return std::make_pair(tmp_x, infos_out);
}

torch::nn::ModuleList ResNet_locateImpl::_make_ppms_layer() {
    torch::nn::ModuleList list;
    for (const auto i : pool_sizes) {
        list->push_back(
            torch::nn::Sequential(
                torch::nn::AdaptiveAvgPool2d(
                    torch::nn::AdaptiveAvgPool2dOptions(/*output_size=*/i)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(
                    /*in_channels=*/ inplanes, 
                    /*out_channels=*/inplanes, 
                    /*kernel_size=*/ 1)
                    .stride(1).padding(0).bias(false)),
                torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true))));
    }
    return list;
}

torch::nn::ModuleList ResNet_locateImpl::_make_infos_layer() {
    torch::nn::ModuleList list;
    for (const auto planes : planes_list) {
        list->push_back(
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(
                    /*in_channels=*/ inplanes, 
                    /*out_channels=*/planes, 
                    /*kernel_size=*/ 3)
                    .stride(1).padding(1).bias(false)),
                torch::nn::ReLU(torch::nn::ReLUOptions(/*inplace=*/true))));
    }
    return list;
}

ResNet_locate resnet50() {
    ResNet_locate net(std::vector<int>{ 3, 4, 6, 3 });
    return net;
}