# 导入必要的模块 Import required modules
import time, os, sys
from media.sensor import *
from media.display import *
from media.media import *

# 图像分辨率设置 Image resolution settings
PICTURE_WIDTH = 400
PICTURE_HEIGHT = 240

# 摄像头配置 Camera configuration
sensor = None

# 显示模式选择 Display mode selection
# 可选: "VIRT", "LCD"
# Options: "VIRT"(Virtual Display), "LCD"
DISPLAY_MODE = "LCD"

# 根据显示模式设置分辨率 Set resolution based on display mode
if DISPLAY_MODE == "VIRT":
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080
elif DISPLAY_MODE == "LCD":
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
else:
    raise ValueError("Unknown DISPLAY_MODE, please select 'VIRT', 'LCD'")

# 创建时钟对象用于FPS计算 Create clock object for FPS calculation
clock = time.clock()

try:
    # 初始化摄像头 Initialize camera
    sensor = Sensor()
    sensor.reset()

    # 设置图像分辨率和格式 Set image resolution and format
    sensor.set_framesize(width=PICTURE_WIDTH, height=PICTURE_HEIGHT, chn=CAM_CHN_ID_0)
    sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_0)

    # 初始化显示器 Initialize display
    if DISPLAY_MODE == "VIRT":
        Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, fps=60)
    elif DISPLAY_MODE == "LCD":
        Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)

    # 初始化媒体管理器 Initialize media manager
    MediaManager.init()
    sensor.run()

    while True:
        os.exitpoint()
        clock.tick() # 开始计时 Start timing

        # 捕获图像 Capture image
        img = sensor.snapshot(chn=CAM_CHN_ID_0)

        print("【矩形信息 Line Statistics Start】")

        
        for r in img.find_rects(threshold = 8000):
            img.draw_rectangle(r.rect(), color = (40, 167, 225),thickness=2) 
            for p in r.corners(): img.draw_circle(p[0], p[1], 8, color = (78, 90, 34))
            print(r)
        print("【==============================】")

        # 显示FPS Display FPS
        print(f"FPS: {clock.fps()}")

        # 居中显示图像 Display image centered
        x = int((DISPLAY_WIDTH - PICTURE_WIDTH) / 2)
        y = int((DISPLAY_HEIGHT - PICTURE_HEIGHT) / 2)
        Display.show_image(img, x=x, y=y)

except KeyboardInterrupt as e:
    print("User Stop: ", e)
except BaseException as e:
    print(f"Exception: {e}")
finally:
    # 清理资源 Cleanup resources
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
