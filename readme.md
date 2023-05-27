本项目基于 [YOLOv5 v6.0](https://github.com/ultralytics/yolov5/tree/v6.0)
图形化界面项目地址：[界面](https://github.com/Javacr/PyQt5-YOLOv5)
基于bilibliUP主:[景唯acr](https://space.bilibili.com/34704910)

Download the models of  YOLOv5 v6.1 from [here](https://github.com/ultralytics/yolov5/releases/tag/v6.0)，and put the them to the pt folder. When the GUI runs, the existing models will be automatically detected.

其他版本: [v5.0](https://github.com/Javacr/PyQt5-YOLOv5/tree/yolov5_v5.0), ...


### 演示视频：
[https://www.bilibili.com/video/BV1jM4y1i794/?spm_id_from=333.788&vd_source=f2fccfb804f78aa96a9f694581bafb10](https://www.bilibili.com/video/BV1jM4y1i794/?spm_id_from=333.788&vd_source=f2fccfb804f78aa96a9f694581bafb10)

### 快速搭建

```bash
conda create -n yolov5_pyqt5 python=3.8
conda activate yolov5_pyqt5
pip install -r requirements.txt
python main.py
```
### 其他包

- install pyinstaller

```
pip install pyinstaller==5.7.0
```

- 打包图形化界面

```
pyinstaller -D -w --add-data="./utils/*;./utils" --add-data="./tools/*;./tools" --add-data="./config/*;./config" --add-data="./main_win/icon/*;./main_win/icon" --add-data="./pt/*;./pt" main.py
```

- 如果没出现错误 数据应在 dist/main中

### 功能

1. 支持视频、图片、摄像头
2. 改变模型
3. 改变 IoU
4. 改变置信度
5. 设置延时
6. 播放、暂停、停止
7. 返回统计结果
8. 保存结果

您可以在[main_win]（./main_win）和 [dialog](dialog)中找到ui文件



