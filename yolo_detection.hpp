//
//  yolo_detection.hpp
//  Paddle_yolo_cpp_demo
//
//  Created by Bitmain on 2021/4/16.
//  Copyright © 2021年 AnBaolei. All rights reserved.
//

#ifndef yolo_detection_hpp
#define yolo_detection_hpp

#include <cstring>
#include <memory>
#include "bmodel_base.hpp"
#include "bm_wrapper.hpp"

typedef struct __tag_st_DetectionInfo {
  int type;
  float score;
  int left;
  int top;
  int right;
  int bottom;
}st_DetectionInfo;


class YoloDetection : public BmodelBase{

public:
  YoloDetection(const std::string bmodel, int device_id);
  ~YoloDetection();

  bool run(std::vector<cv::Mat>& input_imgs,
            std::vector<std::vector<st_DetectionInfo> >&results);

private:
  void preprocess(const std::vector<bm_image>& input_imgs,
                                   std::vector<bm_image>& processed_imgs);
  void postprocess(std::vector<std::vector<st_DetectionInfo> >&results);

private:
  float score_threshold_ = 0.5f;
  bm_device_mem_t image_shape_dev_mem_;
  std::vector<bm_tensor_t> device_inputs_;
  std::vector<bm_tensor_t> device_outputs_;
  std::vector<bm_device_mem_t> output_mem_;
};

#endif /* face_detection_hpp */
