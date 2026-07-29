import time, os, sys, gc
from media.sensor import *
from media.display import *
from media.media import *

# 设置图像捕获分辨率 / Set image capture resolution
SENSOR_WIDTH = 1280
SENSOR_HEIGHT = 960
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# 显示屏分辨率 / Display resolution
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

def init_sensor():
    """初始化传感器 / Initialize sensor"""
    # sensor = Sensor(width=SENSOR_WIDTH, height=SENSOR_HEIGHT)  # 4:3分辨率 / 4:3 resolution
    sensor = Sensor()
    sensor.reset()
    sensor.set_framesize(width=FRAME_WIDTH, height=FRAME_HEIGHT)  # 设置帧大小 / Set frame size
    sensor.set_pixformat(Sensor.GRAYSCALE)  # 灰度图像格式 / Grayscale format
    return sensor

def init_display():
    """初始化显示器 / Initialize display"""
    # 初始化屏幕 / Initialize screen
    Display.init(Display.ST7701, to_ide=True)
    # 虚拟显示配置（已注释） / Virtual display config (commented)
    # Display.init(Display.VIRT, sensor.width(), sensor.height())

def process_image(img):
    """处理图像 / Process image"""
    # Canny边缘检测 / Canny edge detection
    img.find_edges(image.EDGE_CANNY, threshold=(50, 80))

    # 简单边缘检测配置 / Simple edge detection config
    # img.find_edges(image.EDGE_SIMPLE, threshold=(100, 255))

def main():
    try:
        # 初始化设备 / Initialize devices
        sensor = init_sensor()
        init_display()
        MediaManager.init()
        sensor.run()

        # 初始化时钟 / Initialize clock
        clock = time.clock()

        # 计算显示偏移量以居中显示 / Calculate display offsets for center alignment
        x_offset = round((DISPLAY_WIDTH - FRAME_WIDTH) / 2)
        y_offset = round((DISPLAY_HEIGHT - FRAME_HEIGHT) / 2)

        while True:
            clock.tick()  # 更新时钟 / Update clock

            # 捕获和处理图像 / Capture and process image
            img = sensor.snapshot()
            process_image(img)

            # 居中显示图像 / Display image in center
            Display.show_image(img, x=x_offset, y=y_offset)

            # 显示FPS / Display FPS
            print(clock.fps())

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
        gc.collect()  # 垃圾回收 / Garbage collection

if __name__ == "__main__":
    main()
