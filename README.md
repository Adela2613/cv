# 视觉
# 小车识别与颜色检测实践

## 一、系统整体架构
### 1.1 硬件系统
- 摄像头模块
- 处理器平台
- 通信模块
- 显示/输出设备

### 1.2 软件系统  
- 图像采集
- 图像预处理
- 目标检测
- 颜色识别
- 结果输出

## 二、硬件选型建议

### 2.1 摄像头选择
推荐配置：
- 工业相机：HIKROBOT 
  - 1280×1024分辨率
  - 高达60fps
  - USB3.0接口
  - 支持外触发
- 普通USB摄像头：普通模组相机
  - 1080p分辨率
  - 30fps
  - 价格实惠
  - 适合入门

### 2.2 处理平台
入门级：
- 树莓派4B（或者k210但不推荐）
  - 4GB/8GB RAM
  - 支持OpenCV
  - 价格适中
  - 适合原型验证
![](@attachment/Clipboard_2025-03-08-16-08-08.png)（#图可以扣一下）
![](@attachment/Clipboard_2025-03-08-16-08-16.png)
### 3. 优化建议

#### 方案一：轻量级模型
```python
# 使用YOLOv5n（nano版本）
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.conf = 0.25  # 降低置信度阈值
model.iou = 0.45   # 调整IOU阈值
```

#### 方案二：模型量化
```python
# 使用ONNX运行时
import onnxruntime as ort
model = ort.InferenceSession("yolov5n.onnx")
```

#### 方案三：降低输入分辨率
```python
# 处理较小尺寸的图像
frame = cv2.resize(frame, (320, 240))
results = model(frame)
```

进阶级：
- NVIDIA Jetson Nano
  - 支持深度学习加速
  - 128个CUDA核心
  - 功耗低
  - 适合实际部署
![](@attachment/Clipboard_2025-03-08-16-08-25.png)
![](@attachment/Clipboard_2025-03-08-16-08-34.png)
专业级：
- 工控机+独立GPU
  - 强大算力
  - 稳定可靠
  - 适合复杂场景

## 三、软件实现流程

学习网站：csdn ，GitHub ，gitcode，直接谷歌搜索（各种博客）

### 3.1 图像采集
```python
# OpenCV读取视频流示例
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 处理frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
```

### 3.2 图像预处理
关键步骤：
1. 图像去噪
2. 光照均衡
3. 几何校正
4. 尺寸调整

```python
def preprocess_image(frame):
    # 高斯滤波去噪
    denoised = cv2.GaussianBlur(frame, (5,5), 0)
    
    # 直方图均衡化
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # 调整大小
    resized = cv2.resize(equalized, (640,480))
    return resized
```

### 3.3 车辆检测方案


#### 方案一：传统计算机视觉
```python
# 基于HOG特征的车辆检测
def detect_vehicle_hog(frame):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(frame)
    return boxes
```

#### 方案二：深度学习方法
推荐使用YOLOv5：
```python
import torch

# 加载预训练模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = [2,5,7]  # 只检测车辆类别

# 检测
results = model(frame)
vehicles = results.pandas().xyxy[0]
```

### 3.4 颜色识别
![](@attachment/Clipboard_2025-03-08-16-11-15.png)

做一个阈值选择器，便于实时调节阈值
![](@attachment/Clipboard_2025-03-08-16-13-13.png)

#### 3.4.1 HSV颜色空间转换
```python
def detect_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 定义颜色范围
    red_lower = np.array([0,50,50])
    red_upper = np.array([10,255,255])
    
    # 创建掩码
    mask = cv2.inRange(hsv, red_lower, red_upper)
    
    # 计算颜色占比
    ratio = cv2.countNonZero(mask)/(roi.shape[0]*roi.shape[1])
    return 'red' if ratio > 0.5 else 'unknown'
```

#### 3.4.2 主要颜色阈值参考：
- 红色：H(0-10/156-180)
- 蓝色：H(100-124)
- 绿色：H(35-77)
- 黄色：H(26-34)
- 白色：S(0-30),V(221-255)
- 黑色：V(0-46)

## 四、实际应用注意事项

### 4.1 环境因素
- 光照变化
- 阴影干扰
- 天气影响
- 背景复杂度

### 4.2 优化建议
1. 算法优化：
   - 帧间差分
   - 运动预测
   - 目标跟踪
   - 结果平滑

2. 性能提升：
   - GPU加速
   - 多线程处理
   - 降采样
   - ROI提取

## 六、项目实战建议

1. 先搭建简单原型
2. 逐步优化改进
3. 注重实时性
4. 做好异常处理
5. 保存调试日志

# 车辆识别模型训练专题

## 一、数据集准备

### 1.1 常用数据集检索网站
kaggle，GitHub，ultralytics（需要具备服务器，一个集成ai训练平台，自己的电脑3060以上的可以试试）,robotflow（这个有小车数据集，也可以自己上传照片标注数据）

### 1.2 数据标注
推荐工具：
```plaintext
- LabelImg：目标检测标注
- CVAT：在线协作标注
- Labelme：多边形标注
```

标注格式：
```python
# YOLO格式示例
class_id x_center y_center width height

# VOC格式示例
<annotation>
    <object>
        <name>car</name>
        <bndbox>
            <xmin>64</xmin>
            <ymin>32</ymin>
            <xmax>128</xmax>
            <ymax>96</ymax>
        </bndbox>
    </object>
</annotation>
```

## 二、模型选择

### 2.1 主流模型对比
1. **YOLOv5**
   - 速度快
   - 部署方便
   - 适合实时检测

2. **Faster R-CNN**
   - 精度高
   - 两阶段检测
   - 适合精确场景

3. **SSD**
   - 速度和精度平衡
   - 单阶段检测
   - 适合一般应用

### 2.2 模型配置示例
```python
# YOLOv5配置文件示例
model:
  backbone:
    type: CSPDarknet
    depth_multiple: 0.33  # 模型深度
    width_multiple: 0.50  # 模型宽度
  head:
    anchors: 
      - [10,13, 16,30, 33,23]  # P3/8
      - [30,61, 62,45, 59,119]  # P4/16
      - [116,90, 156,198, 373,326]  # P5/32
```

## 三、训练流程

### 3.1 环境配置
```bash
# 创建虚拟环境
conda create -n vehicle_detection python=3.8
conda activate vehicle_detection

# 安装依赖
pip install torch torchvision
pip install opencv-python
pip install albumentations
```

### 3.2 数据增强
```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Blur(blur_limit=3, p=0.3),
    A.CLAHE(p=0.3),
    A.RandomScale(scale_limit=0.3, p=0.5)
])
```

### 3.3 训练代码示例
```python
# YOLOv5训练示例
from utils.torch_utils import select_device
from models.yolo import Model
from utils.datasets import LoadImages

# 加载配置
device = select_device('0')  # GPU
model = Model(cfg='models/yolov5s.yaml')

# 训练循环
for epoch in range(300):  # 300轮
    for batch_i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        pred = model(imgs)
        
        # 计算损失
        loss = compute_loss(pred, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 3.4 验证与调优

#### 3.4.1 评估指标
```python
def calculate_metrics(pred_boxes, true_boxes, iou_threshold=0.5):
    # 计算mAP
    AP = []
    for c in range(num_classes):
        pred_c = pred_boxes[pred_boxes[:, -1] == c]
        true_c = true_boxes[true_boxes[:, -1] == c]
        AP.append(calculate_AP(pred_c, true_c, iou_threshold))
    
    mAP = np.mean(AP)
    return mAP
```

#### 3.4.2 超参数调优
```python
# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=300,
    eta_min=1e-6
)

# 早停策略
early_stopping = EarlyStopping(
    patience=30,
    min_delta=1e-4
)
```

## 四、实用技巧

### 4.1 训练加速
1. **混合精度训练**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    pred = model(imgs)
    loss = compute_loss(pred, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

2. **多GPU训练**
```python
model = torch.nn.DataParallel(model)
model = model.to(device)
```

### 4.2 常见问题解决

1. **过拟合处理**
   - 增加数据增强
   - 添加正则化
   - 使用Dropout
   - 减小模型容量

2. **类别不平衡**
```python
# 使用Focal Loss
class FocalLoss(nn.Module):
    def forward(self, pred, target):
        alpha = 0.25
        gamma = 2.0
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
```

## 五、部署优化

### 5.1 模型压缩
1. **剪枝**
```python
# 结构剪枝示例
pruned_model = prune_model(
    model,
    pruning_ratio=0.3,
    method='l1_norm'
)
```

2. **量化**
```python
# INT8量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 5.2 推理优化
```python
# TensorRT加速
import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

# 构建引擎
with open("model.onnx", 'rb') as model:
    parser.parse(model.read())
engine = builder.build_cuda_engine(network)
```

## 六、实验记录

推荐使用实验管理工具：
```python
# Weights & Biases使用示例
import wandb

wandb.init(project="vehicle-detection")
wandb.config.update({
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 300
})

# 训练时记录指标
wandb.log({
    "train_loss": loss.item(),
    "mAP": mAP
})
```

## 结语
模型训练是一个需要不断调试和优化的过程，希望这些内容能帮助大家更好地开展实验！




