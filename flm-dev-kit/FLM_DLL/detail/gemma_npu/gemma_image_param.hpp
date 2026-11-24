// Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.

#pragma once

constexpr unsigned int  IMAGE_DH =  72;
constexpr unsigned int  IMAGE_NUM_HEADS =  16;
constexpr unsigned int  IMAGE_NUM_KEY_VALUE_HEADS = 16;
constexpr unsigned int  IMAGE_DQ= IMAGE_DH * IMAGE_NUM_HEADS;
constexpr unsigned int  IMAGE_DK= IMAGE_DH * IMAGE_NUM_KEY_VALUE_HEADS;
constexpr unsigned int  IMAGE_DV= IMAGE_DH * IMAGE_NUM_KEY_VALUE_HEADS;

constexpr unsigned int  CONV2D_CIN= 3;
constexpr unsigned int  CONV2D_COUT= 1152;
constexpr unsigned int  CONV2D_KERNEL_SIZE= 14;
constexpr unsigned int  CONV2D_STRIDE= 14;
constexpr unsigned int  CONV2D_PADDING= 0;

constexpr unsigned int  CONV2D_IMAGE_IN_HEIGHT= 896;
constexpr unsigned int  CONV2D_IMAGE_IN_WIDTH= 896;
constexpr unsigned int  CONV2D_IMAGE_OUT_HEIGHT = (CONV2D_IMAGE_IN_HEIGHT + 2 * CONV2D_PADDING - CONV2D_KERNEL_SIZE) / CONV2D_STRIDE + 1;
constexpr unsigned int  CONV2D_IMAGE_OUT_WIDTH = (CONV2D_IMAGE_IN_WIDTH + 2 * CONV2D_PADDING - CONV2D_KERNEL_SIZE) / CONV2D_STRIDE + 1;
constexpr unsigned int  IMAGE_HIDDEN_SIZE= 1152;
constexpr unsigned int  IMAGE_INTERMEDIATE_SIZE= 4304;

constexpr unsigned int AVERAGE_POOLING_KERNEL_SIZE = 4;
constexpr unsigned int    AVERAGE_POOLING_PADDING = 0;
constexpr unsigned int    AVERAGE_POOLING_STRIDE = 4;

constexpr unsigned int  AVERAGE_POOLING_IMAGE_IN_HEIGHT = CONV2D_IMAGE_OUT_HEIGHT;
constexpr unsigned int  AVERAGE_POOLING_IMAGE_IN_WIDTH = CONV2D_IMAGE_OUT_WIDTH;
constexpr unsigned int  AVERAGE_POOLING_IMAGE_OUT_HEIGHT = (AVERAGE_POOLING_IMAGE_IN_HEIGHT + 2 * AVERAGE_POOLING_PADDING - AVERAGE_POOLING_KERNEL_SIZE) / AVERAGE_POOLING_STRIDE + 1;
constexpr unsigned int  AVERAGE_POOLING_IMAGE_OUT_WIDTH = (AVERAGE_POOLING_IMAGE_IN_WIDTH + 2 * AVERAGE_POOLING_PADDING - AVERAGE_POOLING_KERNEL_SIZE) / AVERAGE_POOLING_STRIDE + 1;

constexpr unsigned int VISION_LAYER = 27;
constexpr float VISION_LAYER_NORM_EPS =  1e-06;
constexpr float VISION_RMS_NORM_EPS =  1e-06;
constexpr unsigned int MULTI_MODAL_HIDDEN_SIZE = 2560;

constexpr unsigned int  IMAGE_SEQ_LEN_AVERAGE_POOLING = 256;

constexpr unsigned int TOTAL_OPENMP_THREAD = 4;
