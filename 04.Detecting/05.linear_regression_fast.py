import time, os, sys
from media.sensor import *
from media.display import *
from media.media import *

# 图像处理参数 / Image processing parameters
THRESHOLD = (0, 100)  # 灰度阈值范围 / Grayscale threshold range
BINARY_VISIBLE = True  # 是否显示二值化图像 / Whether to show binary image
                      # 注意：启用二值化可能降低FPS / Note: enabling binary mode may reduce FPS

# 显示参数 / Display parameters
DISPLAY_WIDTH = 640   # LCD显示宽度 / LCD display width
DISPLAY_HEIGHT = 480  # LCD显示高度 / LCD display height
SENSOR_WIDTH = 640    # 图像宽度 / Image width
SENSOR_HEIGHT = 480   # 图像高度 / Image height

def init_sensor():
    """初始化摄像头 / Initialize camera sensor"""
    # sensor = Sensor(width=1280, height=960)  # 4:3比例 / 4:3 aspect ratio
    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(width=SENSOR_WIDTH, height=SENSOR_HEIGHT)
    sensor.set_pixformat(Sensor.GRAYSCALE)
    return sensor

def init_display():
    """初始化显示 / Initialize display"""
    Display.init(Display.ST7701, to_ide=True)
    MediaManager.init()

def process_image(img):
    """处理图像并进行线性回归 / Process image and perform linear regression"""
    # 二值化处理 / Binary thresholding
    if BINARY_VISIBLE:
        img = img.binary([THRESHOLD])

    # 线性回归检测 / Linear regression detection
    # magnitude(): 回归拟合度指标(0,INF], 0表示圆形,值越大越线性
    # magnitude(): regression fitness index (0,INF], 0=circle, larger=more linear
    line = img.get_regression([(255,255) if BINARY_VISIBLE else THRESHOLD])

    if line:
        # 在图像上绘制检测线 / Draw detected line on image
        img.draw_line(line.line(), color=127, thickness=4)
        print(line)

    return line

def main():
    try:
        # 初始化设备 / Initialize devices
        sensor = init_sensor()
        init_display()
        sensor.run()

        clock = time.clock()

        # 计算显示偏移量以居中显示 / Calculate display offsets for center alignment
        x_offset = round((DISPLAY_WIDTH - SENSOR_WIDTH) / 2)
        y_offset = round((DISPLAY_HEIGHT - SENSOR_HEIGHT) / 2)

        while True:
            clock.tick()

            # 捕获图像 / Capture image
            img = sensor.snapshot()

            # 处理图像 / Process image
            line = process_image(img)

            # 显示图像 / Display image
            Display.show_image(img, x=x_offset, y=y_offset)

            # 打印FPS和拟合度 / Print FPS and magnitude
            magnitude = str(line.magnitude()) if line else "N/A"
            print(f"FPS {clock.fps()}, mag = {magnitude}")

    except KeyboardInterrupt as e:
        print("用户中断 / User interrupted: ", e)
    except Exception as e:
        print(f"发生错误 / Error occurred: {e}")
    finally:
        # 清理资源 / Cleanup resources
        if 'sensor' in locals() and isinstance(sensor, Sensor):
            sensor.stop()
        Display.deinit()
        MediaManager.deinit()

if __name__ == "__main__":
    main()
