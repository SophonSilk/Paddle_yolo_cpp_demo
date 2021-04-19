//
//  yolo_detection.cpp
//  Paddle_yolo_cpp_demo
//
//  Created by Bitmain on 2021/4/16.
//  Copyright © 2021年 AnBaolei. All rights reserved.
//

#include <iostream>
#include <assert.h>
#include "yolo_detection.hpp"

using namespace std;

YoloDetection::YoloDetection(const std::string bmodel, int device_id) {
  bmodel_path_ = bmodel;
  device_id_ = device_id;
  load_model();

  // input 0 : image
  float input_scale = 1.0 / 255;
  input_scale *= net_info_->input_scales[0];
  convert_attr_.alpha_0 = input_scale / 0.229;
  convert_attr_.beta_0 = (-0.485 / 0.229) * input_scale;
  convert_attr_.alpha_1 = input_scale / 0.224;
  convert_attr_.beta_1 = (-0.456 / 0.224) * input_scale;
  convert_attr_.alpha_2 = input_scale / 0.225;
  convert_attr_.beta_2 = (-0.406 / 0.225) * input_scale;
  bm_status_t ret = bm_image_create_batch(bm_handle_, net_h_, net_w_,
                        FORMAT_RGB_PLANAR,
                        data_type_,
                        scaled_inputs_, batch_size_);
  if (BM_SUCCESS != ret) {
    std::cerr << "ERROR: bm_image_create_batch failed" << std::endl;
    exit(-1);
  }

  bm_tensor_t image_tensor;
  bm_device_mem_t image_dev_mem;
  bm_image_get_device_mem(*scaled_inputs_, &image_dev_mem);
  bmrt_tensor_with_device(&image_tensor,
                          image_dev_mem,
                          net_info_->input_dtypes[0],
                          input_shape_);
  device_inputs_.push_back(image_tensor);

  // input 1 : image shape
  int alloc_size = 2 * batch_size_ * sizeof(int);
  bmrt_must_alloc_device_mem(p_bmrt_,
                            &image_shape_dev_mem_, alloc_size);
  bm_tensor_t image_shape_tensor;
  bmrt_tensor_with_device(&image_shape_tensor,
                          image_shape_dev_mem_,
                          net_info_->input_dtypes[1],
                          net_info_->stages[0].input_shapes[1]);
  device_inputs_.push_back(image_shape_tensor);

  // output
  for (int i = 0; i < net_info_->output_num; i++) {
    bm_device_mem_t mem;
    bmrt_must_alloc_device_mem(
                 p_bmrt_, &mem, net_info_->max_output_bytes[i]);
    output_mem_.push_back(mem);
    bm_tensor_t output_tensor;
    bmrt_tensor_with_device(&output_tensor,
                            mem,
                            net_info_->output_dtypes[i],
                            net_info_->stages[0].output_shapes[i]);
    device_outputs_.push_back(output_tensor);
  }

}

YoloDetection::~YoloDetection() {
  bmrt_must_free_device_mem(p_bmrt_, image_shape_dev_mem_);
  for (int i = 0; i < net_info_->output_num; i++) {
    bmrt_must_free_device_mem(p_bmrt_, output_mem_[i]);
  }
}

bool YoloDetection::run(std::vector<cv::Mat>& imgs,
                        std::vector<std::vector<st_DetectionInfo> >&results) {
  std::vector<bm_image> input_imgs;
  for (size_t i = 0; i < imgs.size(); i++) {
    bm_image bmimg;
    bm_image_from_mat(bm_handle_, imgs[i], bmimg);
    input_imgs.push_back(bmimg);
  }
  std::vector<bm_image> processed_imgs;
  preprocess(input_imgs, processed_imgs);

  bmcv_image_convert_to(bm_handle_, batch_size_,
             convert_attr_, &processed_imgs[0], scaled_inputs_);

  std::shared_ptr<int> sh_image_shape_ptr(new int(2 * batch_size_));
  int* image_shape_ptr = sh_image_shape_ptr.get();
  for (size_t i = 0; i < input_imgs.size(); i++) {
    image_shape_ptr[i * 2] = input_imgs[i].height;
    image_shape_ptr[i * 2 + 1] = input_imgs[i].width;
  }
  bm_memcpy_s2d(bm_handle_,
                  device_inputs_[1].device_mem,
                  reinterpret_cast<void*>(image_shape_ptr));
  bmrt_launch_tensor_ex(p_bmrt_,
                        net_names_[0],
                        static_cast<const bm_tensor_t*>(&device_inputs_[0]),
                        net_info_->input_num,
                        static_cast<bm_tensor_t*>(&device_outputs_[0]),
                        net_info_->output_num,
                        true,
                        false);
  bm_thread_sync(bm_handle_);
  for (int i = 0; i < net_info_->output_num; i++) {
    bm_memcpy_d2s(bm_handle_,
                    const_cast<void*>(outputs_[i]),
                    device_outputs_[i].device_mem);
  }
  postprocess(results);
  for (size_t i = 0; i < processed_imgs.size(); i++) {
    bm_image_destroy(processed_imgs[i]);
  }
  return true;
}

void YoloDetection::preprocess(const std::vector<bm_image>& input_imgs,
                                   std::vector<bm_image>& processed_imgs) {
  for (size_t i = 0; i < input_imgs.size(); i++) {
    bm_image processed_img;
    bm_image_create(bm_handle_, net_h_, net_w_,
             FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &processed_img, NULL);
    bmcv_rect_t crop_rect = {0, 0, input_imgs[i].width, input_imgs[i].height};
    bmcv_image_vpp_convert(bm_handle_, 1, input_imgs[i], &processed_img, &crop_rect);
    processed_imgs.push_back(processed_img);
  }
  return;
}

void YoloDetection::postprocess(std::vector<std::vector<st_DetectionInfo> >&results) {
  for (int i = 0; i < batch_size_; i++) {
    vector<st_DetectionInfo> image_result;
    for (int j = 0; j < output_num_; j++) {
      auto pred = reinterpret_cast<float*>(outputs_[j]) + output_sizes_[j] * i;
      for (int k = 0; k < output_sizes_[j]; k += 6) {
        if (pred[k + 1] >= score_threshold_) {
          st_DetectionInfo res;
          res.type = static_cast<int>(pred[k * 6]);
          res.score = pred[k  + 1];
          res.left = static_cast<int>(pred[k  + 2]);
          res.top = static_cast<int>(pred[k + 3]);
          res.right = static_cast<int>(pred[k  + 4]);
          res.bottom = static_cast<int>(pred[k  + 5]);
          image_result.push_back(res);
        } else {
         // break;
        }
      } 
    }
    results.push_back(image_result);
  }
  return;
}
