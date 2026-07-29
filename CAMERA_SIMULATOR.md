# OpenMV 拼图自动归位模拟器

`camera_puzzle_simulator.py` 直接读取 `polygon_detection.py` 输出的 JSON，
实时显示摄像头识别到的碎片。点击 `SET` 后，模拟器冻结当时的顶点坐标，
计算拼接关系、目标矩阵和移动顺序，然后播放逐片归位动画。模拟器不会生成
随机碎片。

模拟器提供两个题型：

- `第一题：固定4片（10cm×6cm）`：根据PDF图2的固定尺寸识别4块身份，
  推算 `px/cm` 比例并按唯一固定拼法归位；
- `第二题：通用1～4片`：根据近似等长切割边搜索未知拼法。

固定模板识别全部位于桌面模拟软件的 `puzzle_restoration.py` 中，没有写入或
修改 OpenMV 摄像头识别算法。摄像头仍然只负责输出通用多边形顶点。

## 安装与启动

Tkinter 和求解器只使用 Python 标准库。连接真实 OpenMV 串口需要安装
PySerial：

```bash
python3 -m pip install -r requirements-simulator.txt
python3 camera_puzzle_simulator.py
```

如果只想先检查界面和动画，不需要安装 PySerial：

```bash
python3 camera_puzzle_simulator.py
```

启动后点击“载入 JSON”，选择 `sample_camera_payload.json`，然后点击
`SET · 冻结当前值并计算`。

也可以从 OpenMV IDE 复制控制台输出，点击“粘贴输出 JSON”。粘贴窗口支持：

- 单条紧凑 JSON；
- 多行格式化 JSON；
- 混有启动信息、FPS 日志和多条 JSON 的完整终端输出。

如果存在多条完整 JSON，模拟器自动采用最后一条有效数据。

## 连接 OpenMV

1. 在 OpenMV 上运行当前项目的 `main.py`。
2. 在模拟器中点击“刷新”，选择 OpenMV USB 虚拟串口。
3. 波特率默认使用 115200，然后点击“连接”。
4. 移动碎片时画面会持续采用新的有效 JSON。
5. 确认识别稳定后点击 `SET`。此后串口仍继续接收，但用于求解的初始值不再
   变化，直到点击“返回 LIVE”。

如果串口显示被占用，请先停止 OpenMV IDE 的串口终端或断开 IDE 对虚拟串口
的占用。

在 macOS 上，OpenMV 一般显示为 `/dev/cu.usbmodem…` 或
`/dev/cu.usbserial…`。`/dev/cu.debug-console` 是系统调试控制台，不是
OpenMV，模拟器会阻止连接该设备。如果没有发现 USB 串口，可以直接采用上述
粘贴方式，不影响 SET、求解和动画。

## 数据格式

模拟器兼容 `polygon_detection.make_serial_payload()` 当前输出：

```json
{
  "status": "ok",
  "count": 2,
  "polygons": [
    {
      "id": 1,
      "side_count": 4,
      "vertices_px": [[100, 50], [180, 60], [170, 140], [90, 130]]
    }
  ]
}
```

非 JSON 的启动日志会被自动忽略。只有 `status` 为 `ok`、包含 1～4 块且每块
有 3～5 个顶点的数据会更新实时画面。

## 界面参数

- 默认画面尺寸为当前 OpenMV 配置的 VGA：640×480。
- 默认上下区域分界线为 `Y=240`。
- 目标矩形根据拼接后的实际大小自动居中放入下半区。
- 表格中的抓取中心、目标中心和旋转角仍是图像像素坐标。接入机械臂前需要
  使用相机标定矩阵转换为机械坐标。

### 第一题固定模板

PDF图2给出的外框为 `10cm×6cm`，上边为 `2cm+8cm`，左边为
`2cm+1cm+3cm`。主斜边连接 `(2,0)` 与 `(10,6)`，长度为10cm；其上端
第一段为2cm、下端最后一段为3cm，因此两个汇合点为：

```text
H = (3.6, 1.2) cm
I = (7.6, 4.2) cm
```

由此得到一个三角形和三个四边形的精确模板。SET 后，模拟器会：

1. 对检测轮廓枚举顶点起点并进行旋转、平移、统一缩放无关的形状拟合；
2. 在4!种身份分配中选择总拟合误差最小的唯一对应；
3. 汇总4块的拟合比例，取中位数作为全局 `px/cm`；
4. 用题目尺寸作为身份和相对方位参考，用摄像头实际顶点作为碎片的最终
   轮廓；
5. 只通过平移和旋转计算归位矩阵，不缩放、不拉伸，也不在动画途中修正
   形状；
6. 对实际目标轮廓进行碰撞分离，默认在碎片之间预留 `0.5cm`，可在界面的
   “第一题间隙 cm”中设为 `0～2cm`。

默认不再因轮廓比例差异直接拒绝求解。结果详情区会显示 `px/cm`、`cm/px`、
P编号到F1～F4模板身份、拟合误差、顶点残差以及每块的3×3矩阵。最大顶点
残差在 `2cm` 内视为可用；超过 `2cm` 只给出提示，仍按固定模板的最佳姿态
继续归位。目标范围、抓取中心、目标中心和动画都以 SET 时冻结的实际像素
轮廓为准。若实际轮廓加留隙后无法完全放入下半区，程序保持实际尺寸并尽量
在画面内居中显示，同时给出提示。

## 文件

- `camera_puzzle_simulator.py`：串口、SET 快照和动画界面。
- `puzzle_restoration.py`：无 NumPy/OpenCV 依赖的拼图求解器。
- `test_puzzle_restoration.py`：确定性求解和动画回归测试。
- `sample_camera_payload.json`：离线界面测试数据。

运行测试：

```bash
python3 -m unittest -v test_puzzle_restoration.py
```
