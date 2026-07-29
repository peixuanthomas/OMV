# 优化的三角形检测算法 / Optimized Triangle Detection Algorithm
# 专门针对黑色实心三角形在白色背景上的检测 / Specifically for black solid triangles on white background

import time, os, sys, math
from media.sensor import *
from media.display import *
from media.media import *

# 图像分辨率设置 / Image resolution settings
PICTURE_WIDTH = 640  # 降低分辨率提高帧率 / Lower resolution for better FPS
PICTURE_HEIGHT = 480

# 摄像头配置 / Camera configuration
sensor = None

# 显示模式选择 / Display mode selection
DISPLAY_MODE = "LCD"

# 绘制控制参数 / Drawing control parameters
DRAW_GENERAL_TRIANGLES = False  # 是否绘制普通三角形（绿色）/ Whether to draw general triangles (green)

# 根据显示模式设置分辨率 / Set resolution based on display mode
if DISPLAY_MODE == "VIRT":
    DISPLAY_WIDTH = ALIGN_UP(1920, 16)
    DISPLAY_HEIGHT = 1080
elif DISPLAY_MODE == "LCD":
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
else:
    raise ValueError("Unknown DISPLAY_MODE, please select 'VIRT', 'LCD'")

# 创建时钟对象用于FPS计算 / Create clock object for FPS calculation
clock = time.clock()

def distance(p1, p2):
    """计算两点间距离 / Calculate distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(p1, p2, p3):
    """计算三点构成的角度 / Calculate angle formed by three points"""
    try:
        # 向量 p2->p1 和 p2->p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # 计算向量长度
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if len1 == 0 or len2 == 0:
            return 0

        # 计算夹角余弦值
        cos_angle = (v1[0]*v2[0] + v1[1]*v2[1]) / (len1 * len2)
        cos_angle = max(-1, min(1, cos_angle))  # 限制在[-1,1]范围内

        # 转换为角度
        angle = math.acos(cos_angle) * 180 / math.pi
        return angle
    except:
        return 0

def is_equilateral_triangle(corners, side_tolerance=0.2, angle_tolerance=20):
    """
    判断三个角点是否构成等边三角形 / Check if three corners form an equilateral triangle

    参数 / Parameters:
        corners: 三个角点的坐标列表 / List of three corner coordinates
        side_tolerance: 边长相对误差容忍度 / Side length relative tolerance (0.15 = 15%)
        angle_tolerance: 角度误差容忍度 / Angle tolerance in degrees

    返回 / Returns:
        bool: 是否为等边三角形 / Whether it's an equilateral triangle
    """
    if len(corners) != 3:
        return False

    # 计算三边长度 / Calculate three side lengths
    side1 = distance(corners[0], corners[1])
    side2 = distance(corners[1], corners[2])
    side3 = distance(corners[2], corners[0])

    # 检查边长是否过小 / Check if sides are too small
    if side1 < 10 or side2 < 10 or side3 < 10:
        return False

    # 方法1: 检查三边长度是否相等（允许相对误差）/ Method 1: Check if three sides are equal (with relative tolerance)
    avg_side = (side1 + side2 + side3) / 3
    side1_diff = abs(side1 - avg_side) / avg_side
    side2_diff = abs(side2 - avg_side) / avg_side
    side3_diff = abs(side3 - avg_side) / avg_side

    sides_equal = (side1_diff <= side_tolerance and
                   side2_diff <= side_tolerance and
                   side3_diff <= side_tolerance)

    # 方法2: 检查三个内角是否都接近60度 / Method 2: Check if three angles are close to 60 degrees
    angle1 = calculate_angle(corners[0], corners[1], corners[2])
    angle2 = calculate_angle(corners[1], corners[2], corners[0])
    angle3 = calculate_angle(corners[2], corners[0], corners[1])

    angles_equal = (abs(angle1 - 60) <= angle_tolerance and
                    abs(angle2 - 60) <= angle_tolerance and
                    abs(angle3 - 60) <= angle_tolerance)

    # 两种方法都满足才认为是等边三角形 / Both methods must be satisfied
    return sides_equal and angles_equal

def is_valid_triangle(corners, min_area=100, max_area=10000, angle_tolerance=20):
    """
    判断三个角点是否构成有效三角形 / Check if three corners form a valid triangle

    参数 / Parameters:
        corners: 三个角点的坐标列表 / List of three corner coordinates
        min_area: 最小面积阈值 / Minimum area threshold
        max_area: 最大面积阈值 / Maximum area threshold
        angle_tolerance: 角度容忍度 / Angle tolerance

    返回 / Returns:
        bool: 是否为有效三角形 / Whether it's a valid triangle
    """
    if len(corners) != 3:
        return False

    # 计算三边长度 / Calculate three side lengths
    side1 = distance(corners[0], corners[1])
    side2 = distance(corners[1], corners[2])
    side3 = distance(corners[2], corners[0])

    # 检查边长是否合理 / Check if side lengths are reasonable
    # if side1 < 10 or side2 < 10 or side3 < 10:  # 最小边长限制
    #     return False

    # if side1 > 200 or side2 > 200 or side3 > 200:  # 最大边长限制
    #     return False

    # 计算面积 / Calculate area using cross product
    area = abs((corners[1][0] - corners[0][0]) * (corners[2][1] - corners[0][1]) -
               (corners[2][0] - corners[0][0]) * (corners[1][1] - corners[0][1])) / 2

    # if area < min_area or area > max_area:
    #     return False

    # 计算三个内角 / Calculate three interior angles
    angle1 = calculate_angle(corners[0], corners[1], corners[2])
    angle2 = calculate_angle(corners[1], corners[2], corners[0])
    angle3 = calculate_angle(corners[2], corners[0], corners[1])

    # 检查角度是否合理（三角形内角和应该接近180度）/ Check if angles are reasonable
    angle_sum = angle1 + angle2 + angle3
    if abs(angle_sum - 180) > 30:  # 允许一定误差
        return False

    # 检查是否有过小的角度 / Check for too small angles
    if angle1 < 20 or angle2 < 20 or angle3 < 20:
        return False

    return True

def find_triangles_from_contours(img):
    """
    使用轮廓检测寻找三角形 / Find triangles using contour detection

    参数 / Parameters:
        img: 输入图像 / Input image

    返回 / Returns:
        list: 检测到的三角形列表 / List of detected triangles
    """
    triangles = []

    try:
        # 转换为灰度图像进行边缘检测 / Convert to grayscale for edge detection
        img_gray = img.to_grayscale()

        # 使用自适应阈值进行二值化 / Use adaptive threshold for binarization
        # 针对黑色三角形，使用反向阈值 / For black triangles, use inverted threshold
        img_binary = img_gray.binary([(0, 100)], invert=True)  # 检测黑色区域

        # 寻找矩形作为候选区域 / Find rectangles as candidate regions
        rects = img_binary.find_rects(threshold=1000, roi=None)

        for rect in rects:
            corners = rect.corners()
            if corners is not None and len(corners) >= 3:
                # 尝试所有可能的三点组合 / Try all possible three-point combinations
                for i in range(len(corners)):
                    for j in range(i+1, len(corners)):
                        for k in range(j+1, len(corners)):
                            triangle_corners = [corners[i], corners[j], corners[k]]
                            if is_valid_triangle(triangle_corners):
                                center = ((triangle_corners[0][0] + triangle_corners[1][0] + triangle_corners[2][0]) // 3,
                                         (triangle_corners[0][1] + triangle_corners[1][1] + triangle_corners[2][1]) // 3)
                                area = abs((triangle_corners[1][0] - triangle_corners[0][0]) *
                                          (triangle_corners[2][1] - triangle_corners[0][1]) -
                                          (triangle_corners[2][0] - triangle_corners[0][0]) *
                                          (triangle_corners[1][1] - triangle_corners[0][1])) / 2

                                # 检查是否为等边三角形 / Check if it's an equilateral triangle
                                is_equilateral = is_equilateral_triangle(triangle_corners)

                                triangles.append({
                                    'corners': triangle_corners,
                                    'center': center,
                                    'area': area,
                                    'is_equilateral': is_equilateral,
                                    'type': 'equilateral' if is_equilateral else 'general'
                                })
                                break
    except Exception as e:
        pass
#        print(f"Error in contour detection: {e}")

    return triangles

def find_triangles_from_blobs(img):
    """
    使用blob检测寻找三角形 / Find triangles using blob detection

    参数 / Parameters:
        img: 输入图像 / Input image

    返回 / Returns:
        list: 检测到的三角形列表 / List of detected triangles
    """
    triangles = []

    try:
        # 检测黑色blob / Detect black blobs
        blobs = img.find_blobs([(0, 50, -128, 127, -128, 127)],
                              pixels_threshold=200,
                              area_threshold=500,
                              merge=True)

        for blob in blobs:
            # 获取blob的边界框 / Get blob bounding box
            x, y, w, h = blob.rect()

            # 检查长宽比是否合理 / Check if aspect ratio is reasonable
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 3:  # 过于细长的形状可能不是三角形
                continue

            # 使用blob的角点信息 / Use blob corner information
            if hasattr(blob, 'corners') and callable(blob.corners):
                corners = blob.corners()
                if corners is not None and len(corners) >= 3:
                    # 选择最合适的三个角点 / Select the most suitable three corners
                    best_triangle = None
                    best_score = 0

                    for i in range(len(corners)):
                        for j in range(i+1, len(corners)):
                            for k in range(j+1, len(corners)):
                                triangle_corners = [corners[i], corners[j], corners[k]]
                                if is_valid_triangle(triangle_corners):
                                    # 计算三角形质量分数 / Calculate triangle quality score
                                    area = abs((triangle_corners[1][0] - triangle_corners[0][0]) *
                                             (triangle_corners[2][1] - triangle_corners[0][1]) -
                                             (triangle_corners[2][0] - triangle_corners[0][0]) *
                                             (triangle_corners[1][1] - triangle_corners[0][1])) / 2

                                    # 分数基于面积和形状规整度 / Score based on area and shape regularity
                                    score = area
                                    if score > best_score:
                                        best_score = score
                                        best_triangle = triangle_corners

                    if best_triangle:
                        center = ((best_triangle[0][0] + best_triangle[1][0] + best_triangle[2][0]) // 3,
                                 (best_triangle[0][1] + best_triangle[1][1] + best_triangle[2][1]) // 3)

                        # 检查是否为等边三角形 / Check if it's an equilateral triangle
                        is_equilateral = is_equilateral_triangle(best_triangle)

                        triangles.append({
                            'corners': best_triangle,
                            'center': center,
                            'area': best_score,
                            'is_equilateral': is_equilateral,
                            'type': 'equilateral' if is_equilateral else 'general'
                        })
    except Exception as e:
        print(f"Error in blob detection: {e}")

    return triangles

def find_triangles_optimized(img):
    """
    优化的三角形检测主函数 / Optimized main triangle detection function

    参数 / Parameters:
        img: 输入图像 / Input image

    返回 / Returns:
        list: 检测到的三角形列表 / List of detected triangles
    """
    all_triangles = []

    # 方法1: 轮廓检测 / Method 1: Contour detection
    triangles1 = find_triangles_from_contours(img)
    all_triangles.extend(triangles1)

    # 方法2: Blob检测 / Method 2: Blob detection
    triangles2 = find_triangles_from_blobs(img)
    all_triangles.extend(triangles2)

    # 去重和筛选 / Remove duplicates and filter
    unique_triangles = []
    for triangle in all_triangles:
        is_duplicate = False
        for existing in unique_triangles:
            # 检查中心点距离 / Check center point distance
            center_dist = distance(triangle['center'], existing['center'])
            if center_dist < 20:  # 如果中心点很近，认为是重复的
                is_duplicate = True
                break

        if not is_duplicate:
            unique_triangles.append(triangle)

    # 按面积排序，优先返回较大的三角形 / Sort by area, prioritize larger triangles
    unique_triangles.sort(key=lambda x: x['area'], reverse=True)

    return unique_triangles[:5]  # 最多返回5个三角形

def process_triangles(img, triangles):
    """处理检测到的三角形 / Process detected triangles"""
    print("【三角形检测结果 / Triangle Detection Results】")
    equilateral_count = 0
    general_count = 0

    for i, triangle in enumerate(triangles):
        corners = triangle['corners']
        center = triangle['center']
        area = triangle['area']
        is_equilateral = triangle.get('is_equilateral', False)
        triangle_type = triangle.get('type', 'general')

        # 统计三角形类型 / Count triangle types
        if is_equilateral:
            equilateral_count += 1
        else:
            general_count += 1

        # 根据三角形类型和控制参数决定是否绘制 / Draw based on triangle type and control parameters
        should_draw = False
        if is_equilateral:
            should_draw = True
            line_color = (255, 255, 0)  # 黄色表示等边三角形 / Yellow for equilateral triangles
            corner_color = (255, 165, 0)  # 橙色角点 / Orange corner points
            center_color = (255, 0, 255)  # 紫色中心点 / Purple center point
            label = "EQUI"
        elif DRAW_GENERAL_TRIANGLES:
            should_draw = True
            line_color = (0, 255, 0)  # 绿色表示普通三角形 / Green for general triangles
            corner_color = (255, 0, 0)  # 红色角点 / Red corner points
            center_color = (0, 0, 255)  # 蓝色中心点 / Blue center point
            label = "GEN"

        if should_draw:
            # 绘制三角形边框 / Draw triangle outline
            for j in range(3):
                start = corners[j]
                end = corners[(j + 1) % 3]
                img.draw_line(start[0], start[1], end[0], end[1], color=line_color, thickness=2)

            # 绘制角点 / Draw corner points
            for corner in corners:
                img.draw_circle(corner[0], corner[1], 4, color=corner_color, thickness=2)

            # 绘制中心点 / Draw center point
            img.draw_circle(center[0], center[1], 3, color=center_color, thickness=2)

            # 在三角形旁边显示类型标签 / Display type label next to triangle
            img.draw_string_advanced(center[0] + 10, center[1] - 10, 12, label, color=line_color, scale=1)

        # 计算边长 / Calculate side lengths
        side1 = distance(corners[0], corners[1])
        side2 = distance(corners[1], corners[2])
        side3 = distance(corners[2], corners[0])

        # 计算三个内角用于验证 / Calculate three angles for verification
        angle1 = calculate_angle(corners[0], corners[1], corners[2])
        angle2 = calculate_angle(corners[1], corners[2], corners[0])
        angle3 = calculate_angle(corners[2], corners[0], corners[1])

        print(f"Triangle {i+1} ({triangle_type.upper()}): Center({center[0]}, {center[1]}), Area: {area:.1f}")
        print(f"  Corners: {corners}")
        print(f"  Side lengths: {side1:.1f}, {side2:.1f}, {side3:.1f}")
        if is_equilateral:
            print(f"  Angles: {angle1:.1f}°, {angle2:.1f}°, {angle3:.1f}°")
            print(f"  ★ EQUILATERAL TRIANGLE DETECTED! ★")

    print(f"Total triangles found: {len(triangles)} (Equilateral: {equilateral_count}, General: {general_count})")
    print("【==============================】")

try:
    # 初始化摄像头 / Initialize camera
    sensor = Sensor()
#    sensor = Sensor(width=640, height=480)
    sensor.reset()

    # 设置图像分辨率和格式 / Set image resolution and format
    sensor.set_framesize(width=PICTURE_WIDTH, height=PICTURE_HEIGHT, chn=CAM_CHN_ID_0)
    sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_0)

    # 初始化显示器 / Initialize display
    if DISPLAY_MODE == "VIRT":
        Display.init(Display.VIRT, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, fps=60)
    elif DISPLAY_MODE == "LCD":
        Display.init(Display.ST7701, width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, to_ide=True)

    # 初始化媒体管理器 / Initialize media manager
    MediaManager.init()
    sensor.run()

    # 计算显示偏移量以居中显示 / Calculate display offsets for center alignment
    x_offset = (DISPLAY_WIDTH - PICTURE_WIDTH) // 2
    y_offset = (DISPLAY_HEIGHT - PICTURE_HEIGHT) // 2

    print("三角形检测已启动，请将黑色三角形放在白色背景前...")
    print("Triangle detection started, please place black triangles on white background...")

    while True:
        os.exitpoint()
        clock.tick()  # 开始计时 / Start timing

        # 捕获图像 / Capture image
        img = sensor.snapshot(chn=CAM_CHN_ID_0)

        # 寻找三角形 / Find triangles
        triangles = find_triangles_optimized(img)

        # 处理检测到的三角形 / Process detected triangles
        equilateral_count = 0
        general_count = 0
        if len(triangles) > 0:
            process_triangles(img, triangles)
            # 统计等边三角形数量 / Count equilateral triangles
            for triangle in triangles:
                if triangle.get('is_equilateral', False):
                    equilateral_count += 1
                else:
                    general_count += 1

        # 显示FPS / Display FPS
        fps = clock.fps()
        print(f"FPS: {fps:.1f}")

        # 在图像上显示信息 / Display information on image
        img.draw_string_advanced(10, 10, 15, f"FPS: {fps:.1f}", color=(255, 255, 255), scale=2)
        img.draw_string_advanced(10, 30, 15, f"Total: {len(triangles)}", color=(255, 255, 255), scale=2)
        if equilateral_count > 0:
            img.draw_string_advanced(10, 50, 15, f"Equilateral: {equilateral_count}", color=(255, 255, 0), scale=2)
        if general_count > 0 and DRAW_GENERAL_TRIANGLES:
            img.draw_string_advanced(10, 70, 15, f"General: {general_count}", color=(0, 255, 0), scale=2)

        # 居中显示图像 / Display image centered
        Display.show_image(img, x=x_offset, y=y_offset)

except KeyboardInterrupt as e:
    print("User Stop: ", e)
except BaseException as e:
    print(f"Exception: {e}")
finally:
    # 清理资源 / Cleanup resources
    if isinstance(sensor, Sensor):
        sensor.stop()
    Display.deinit()
    os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
    time.sleep_ms(100)
    MediaManager.deinit()
