# Paddle_yolo_cpp_demo
在比特大陆AI硬件上运行paddle yolo模型的demo

将工程clone到比特大陆bmnnsdk2的bmnnsdk2-bm1684_vbm1684_v2.x.x/examples/YOLOv3_object目录

1. 编译

    a) X86 + pcie模式
  
      make -f Makefile.pcie -j
    
    b) SoC 模式

      make -f Makefile.arm -j

2. 运行

    a) 图片模式

      ./yolo_test image imagelist.txt compilation.bmodel
      其中，imagelist.txt的每一行是图片的路径。

    b) 视频模式

      ./yolo_test video videolst.txt compilation.bmodel 
      其中，videolst.txt的每一行是视频文件流的url地址或者本地视频文件的路径。 


