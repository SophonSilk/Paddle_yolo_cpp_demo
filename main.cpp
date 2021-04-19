//
//  main.cpp
//  Paddle_yolo_cpp_demo
//
//  Created by Bitmain on 2021/4/17.
//  Copyright © 2021年 AnBaolei. All rights reserved.
//

#include <iostream>

#include <boost/filesystem.hpp>
#include <condition_variable>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include "yolo_detection.hpp"
#include "utils.hpp"

#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

namespace fs = boost::filesystem;
using namespace std;
using time_stamp_t = time_point<steady_clock, microseconds>;

static void detect(YoloDetection &net, vector<cv::Mat>& images,
                                      vector<string> names, TimeStamp *ts) {
  ts->save("detection overall");
  std::vector<std::vector<st_DetectionInfo> >results;
  net.run(images, results);
  ts->save("detection overall");

  string save_folder = "result_imgs";
  if (!fs::exists(save_folder)) {
    fs::create_directory(save_folder);
  }

  for (size_t i = 0; i < images.size(); i++) {
    for (size_t j = 0; j < results[i].size(); j++) {
      int x_min = results[i][j].left;
      int x_max = results[i][j].right;
      int y_min = results[i][j].top;
      int y_max = results[i][j].bottom;

      std::cout << "Category: " << results[i][j].type
        << " Score: " << results[i][j].score << " : " << x_min <<
        "," << y_min << "," << x_max << "," << y_max << std::endl;

      cv::Rect rc;
      rc.x = x_min;
      rc.y = y_min;;
      rc.width = x_max - x_min;
      rc.height = y_max - y_min;
      cv::rectangle(images[i], rc, cv::Scalar(0, 255, 0), 2, 1, 0);
    }
    cv::imwrite(save_folder + "/" + names[i], images[i]);
  }
}

int main(int argc, char **argv) {
  cout.setf(ios::fixed);

  if (argc < 4) {
    cout << "USAGE:" << endl;
    cout << "  " << argv[0] << " image <image list> <bmodel file> " << endl;
    cout << "  " << argv[0] << " video <video url>  <bmodel file> " << endl;
    exit(1);
  }

  bool is_video = false;
  if (strcmp(argv[1], "video") == 0) {
    is_video = true;
  } else if (strcmp(argv[1], "image") == 0) {
    is_video = false;
  } else {
    cout << "Wrong input type, neither image nor video." << endl;
    exit(1);
  }

  string image_list = argv[2];
  if (!is_video && !fs::exists(image_list)) {
    cout << "Cannot find input image file." << endl;
    exit(1);
  }

  string bmodel_file = argv[3];
  if (!fs::exists(bmodel_file)) {
    cout << "Cannot find valid model file." << endl;
    exit(1);
  }

  YoloDetection net(bmodel_file, 0);
  int batch_size = net.get_batch_size();
  TimeStamp ts;
  char image_path[1024] = {0};
  ifstream fp_img_list(image_list);
  if (!is_video) {
    vector<cv::Mat> batch_imgs;
    vector<string> batch_names;
    while(fp_img_list.getline(image_path, 1024)) {
      ts.save("decode overall");
      ts.save("stage 0: decode");
      cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR, 0);
      ts.save("stage 0: decode");
      if (img.empty()) {
         cout << "read image error!" << endl;
         exit(1);
      }
      ts.save("decode overall");
      fs::path fs_path(image_path);
      string img_name = fs_path.filename().string();
      batch_imgs.push_back(img);
      batch_names.push_back(img_name);
      if (static_cast<int>(batch_imgs.size()) == batch_size) {
        detect(net, batch_imgs, batch_names, &ts);
        batch_imgs.clear();
        batch_names.clear();
      }
    }
  } else {
    vector <cv::VideoCapture> caps;
    vector <string> cap_srcs;
    while(fp_img_list.getline(image_path, 1024)) {
      cv::VideoCapture cap(image_path);
      cap.set(cv::CAP_PROP_OUTPUT_YUV, 1);
      caps.push_back(cap);
      cap_srcs.push_back(image_path);
    }

    if ((int)caps.size() != batch_size) {
      cout << "video num should equal model's batch size" << endl;
      exit(1);
    }

    uint32_t batch_id = 0;
    const uint32_t run_frame_no = 200;
    uint32_t frame_id = 0;
    while(1) {
      if (frame_id == run_frame_no) {
        break;
      }
      vector<cv::Mat> batch_imgs;
      vector<string> batch_names;
      ts.save("decode overall");
      ts.save("stage 0: decode");
      for (size_t i = 0; i < caps.size(); i++) {
         if (caps[i].isOpened()) {
           int w = int(caps[i].get(cv::CAP_PROP_FRAME_WIDTH));
           int h = int(caps[i].get(cv::CAP_PROP_FRAME_HEIGHT));
           cv::Mat img;
           caps[i] >> img;
           if (img.rows != h || img.cols != w) {
             break;
           }
           batch_imgs.push_back(img);
           batch_names.push_back(to_string(batch_id) + "_" +
                            to_string(i) + "_video.jpg");
           batch_id++;
         } else {
           cout << "VideoCapture " << i << " "
                   << cap_srcs[i] << " open failed!" << endl;
         }
      }
      if ((int)batch_imgs.size() < batch_size) {
        break;
      }
      ts.save("stage 0: decode");
      ts.save("decode overall");
      detect(net, batch_imgs, batch_names, &ts);
      batch_imgs.clear();
      batch_names.clear();
      frame_id += 1;
    }
  }

  time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("yolo detect");
  ts.show_summary("detect ");
  ts.clear();

  std::cout << std::endl;

  return 0;
}
