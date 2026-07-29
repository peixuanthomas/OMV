# 导入必要的模块：时间、操作系统、系统、垃圾回收
# (Import required modules: time, operating system, system, garbage collection)
import time, os, sys, gc
# 导入媒体相关模块：传感器、显示、媒体管理
# (Import media-related modules: sensor, display, media management)
from media.sensor import *
from media.display import *
from media.media import *
# 导入PipeLine库，用于图像处理Pipeline和性能计时
# (Import PipeLine library for image processing pipeline and performance timing)
from libs.PipeLine import PipeLine, ScopedTiming

# 设置图像处理分辨率常量
# (Set image processing resolution constants)
PICTURE_WIDTH = 160
PICTURE_HEIGHT = 120

# 初始化摄像头变量为空
# (Initialize camera variable as None)
sensor = None

# 设置显示分辨率常量
# (Set display resolution constants)
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


def scale_coordinates(data_tuple, target_resolution="640x480"):
    # 声明全局变量
    # (Declare global variables)
    global PICTURE_WIDTH, PICTURE_HEIGHT
    """
    将160x120分辨率下的坐标元组等比例缩放到目标分辨率
    (Scale coordinate tuple from 160x120 resolution proportionally to target resolution)
    
    参数 (Parameters):
        data_tuple: 包含坐标信息的元组 (x1, y1, x2, y2)
                   (Tuple containing coordinate information (x1, y1, x2, y2))
        target_resolution: 目标分辨率，可选 "640x480" 或 "640x480"
                          (Target resolution, optional "640x480" or "640x480")
    
    返回 (Returns):
        包含缩放后坐标的新元组 (x1, y1, x2, y2, length)
        (New tuple containing scaled coordinates (x1, y1, x2, y2, length))
    """
    # 检查输入类型，确保是至少包含4个元素的元组
    # (Check input type, ensure it's a tuple with at least 4 elements)
    if not isinstance(data_tuple, tuple) or len(data_tuple) < 4:
        raise TypeError(f"期望输入至少包含4个元素的元组，但收到了 {type(data_tuple).__name__}")
        # (Expected a tuple with at least 4 elements, but received {type})
    
    # 从元组中解析坐标点
    # (Extract coordinates from the tuple)
    x1, y1, x2, y2 = data_tuple[:4]
    
    # 设置原始分辨率
    # (Set source resolution)
    src_width, src_height = PICTURE_WIDTH, PICTURE_HEIGHT
    
    # 根据目标分辨率参数设置目标宽高
    # (Set target width and height based on target resolution parameter)
    if target_resolution == "640x480":
        dst_width, dst_height = 640, 480
    elif target_resolution == "640x480":  # 注意：这里条件与上面相同，可能是代码错误
        # (Note: this condition is the same as above, might be a code error)
        dst_width, dst_height = 640, 480
    else:
        raise ValueError("不支持的分辨率，请使用 '640x480' 或 '640x480'")
        # (Unsupported resolution, please use '640x480' or '640x480')
    
    # 计算横向和纵向的缩放比例
    # (Calculate horizontal and vertical scaling ratios)
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    
    # 对坐标进行缩放，并四舍五入保证是整数
    # (Scale coordinates and round to ensure integers)
    scaled_x1 = round(x1 * scale_x)
    scaled_y1 = round(y1 * scale_y)
    scaled_x2 = round(x2 * scale_x)
    scaled_y2 = round(y2 * scale_y)
    
    # 计算缩放后线段的长度（欧几里得距离）
    # (Calculate length of scaled line segment (Euclidean distance))
    dx = scaled_x2 - scaled_x1
    dy = scaled_y2 - scaled_y1
    length = round((dx**2 + dy**2)**0.5)
    
    # 返回缩放后的坐标元组
    # (Return tuple of scaled coordinates)
    return (scaled_x1, scaled_y1, scaled_x2, scaled_y2)

# 设置显示模式为LCD
# (Set display mode to LCD)

display_mode = "LCD"

# 创建图像处理Pipeline，设置RGB888格式尺寸和显示尺寸
# (Create image processing pipeline with RGB888 format size and display size)
pl = PipeLine(rgb888p_size=[640,360], display_size=[640,480], display_mode=display_mode)
# 创建Pipeline实例，设置通道1的帧大小
# (Create pipeline instance, set frame size for channel 1)
pl.create(ch1_frame_size=[PICTURE_WIDTH,PICTURE_HEIGHT])

# 主循环
# (Main loop)
while True:
    # 从通道1捕获图像
    # (Capture image from channel 1)
    img = pl.sensor.snapshot(chn=CAM_CHN_ID_1)
    
    # 在图像中查找线段，合并距离为20，最大theta差异为5度
    # (Find line segments in the image, merge distance 20, max theta difference 5 degrees)
    lines = img.find_line_segments(merge_distance=15, max_theta_diff=10)
    
    # 创建一个新的ARGB8888格式的图像用于显示
    # (Create a new ARGB8888 format image for display)
    img = image.Image(640, 480, image.ARGB8888)
    
    # 清空图像
    # (Clear the image)
    img.clear()
    
    # 遍历找到的所有线段
    # (Iterate through all found line segments)
    for i, line in enumerate(lines):
        # 获取线段坐标并缩放到显示分辨率
        # (Get line segment coordinates and scale to display resolution)
        line = scale_coordinates(line.line())
        
        # 在图像上绘制红色线段，线宽为6
        # (Draw red line on the image with thickness 6)
        img.draw_line(line, color=(255,0,0), thickness=6)
    
    # 在OSD3层显示图像
    # (Display the image on OSD3 layer)
    Display.show_image(img, 0, 0, Display.LAYER_OSD3)
    
    # 短暂休眠微秒级延时，避免CPU过度占用
    # (Brief microsecond sleep to avoid excessive CPU usage)
    time.sleep_us(1)