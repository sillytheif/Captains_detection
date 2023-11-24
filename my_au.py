from PIL import Image
import numpy as np
import os
import random
import math
import cv2
from visualize_bbox import clear_images_in_folder
import string
from PIL import Image, ImageDraw, ImageFont


def load_yolo_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    annotations = [list(map(float, line.strip().split())) for line in lines]
    return annotations

def save_yolo_annotations(annotations, output_path):
    with open(output_path, 'w') as file:
        for ann in annotations:
            file.write(' '.join(map(str, ann)) + '\n')

def save_data(aug_image, aug_bboxes,img_path, save_root_path, pre_fix=""):
    img_name = os.path.split(img_path)[1]
    save_jpg_name = save_root_path + pre_fix + img_name
    save_txt_name = save_jpg_name[:-3] + "txt" 
    aug_image.save(save_jpg_name)
    save_yolo_annotations(aug_bboxes, save_txt_name)
    return


def translate_annotations(annotations, dx, dy, img_width, img_height):
    translated_annotations = []
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        x_center += dx / img_width
        y_center += dy / img_height
        translated_annotations.append([class_id, x_center, y_center, width, height])
    return translated_annotations

def flip_annotations_horizontally(annotations, img_width):
    flipped_annotations = []
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        x_center = 1 - x_center
        flipped_annotations.append([class_id, x_center, y_center, width, height])
    return flipped_annotations

def flip_annotations_vertically(annotations, img_height):
    flipped_annotations = []
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        y_center = 1 - y_center
        flipped_annotations.append([class_id, x_center, y_center, width, height])
    return flipped_annotations

def scale_annotations(annotations, scale_factor):
    scaled_annotations = []
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann
        width *= scale_factor
        height *= scale_factor
        scaled_annotations.append([class_id, x_center, y_center, width, height])
    return scaled_annotations

def create_image_with_random_background(image_size):
    # 定义背景颜色选项
    background_colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "dark_green": (0, 100, 0),
        "orange": (255, 165, 0),
        "pink": (255, 192, 203),
        "light_blue": (173, 216, 230),
        "dark_red": (139, 0, 0),
        "turquoise": (64, 224, 208),
        "magenta": (255, 0, 255),
        "gold": (255, 215, 0),
        "lime": (0, 255, 0),
        "teal": (0, 128, 128),
        "olive": (128, 128, 0),       # 橄榄色
        "maroon": (128, 0, 0),        # 栗色
        "navy": (0, 0, 128),          # 海军蓝
        "coral": (255, 127, 80),      # 珊瑚色
        "mint": (189, 252, 201),      # 薄荷色
        "orchid": (218, 112, 214),    # 兰花色
        "beige": (245, 245, 220),     # 米色
        "mustard": (255, 219, 88),    # 芥末色
        "lavender": (230, 230, 250),  # 薰衣草色
        "salmon": (250, 128, 114),    # 鲑鱼色
        "charcoal": (54, 69, 79),     # 木炭色
        "emerald": (80, 200, 120)     # 翡翠色
    }

    # 随机选择一个背景颜色
    random_color = random.choice(list(background_colors.values()))

    # 创建带有随机背景颜色的新图像
    new_image = Image.new("RGB", image_size, random_color)
    return new_image

def is_overlap_allowed(bbox1, bbox2):
    """
    判断两个边界框是否重叠，但允许小于20%的重叠。
    每个边界框格式：[x_center, y_center, width, height]
    """
    def bbox_area(bbox):
        return bbox[2] * bbox[3]

    def intersect_area(b1, b2):
        x_overlap = max(0, min(b1[0] + b1[2] / 2, b2[0] + b2[2] / 2) - max(b1[0] - b1[2] / 2, b2[0] - b2[2] / 2))
        y_overlap = max(0, min(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2) - max(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2))
        return x_overlap * y_overlap
    overlap_area = intersect_area(bbox1, bbox2)
    return overlap_area < 0.2 * bbox_area(bbox2)  # 允许小于20%的重叠


def rotate_bbox(bbox, angle, img_width, img_height):
    # 旋转 90，180，270度
    class_id, x_center, y_center, width, height = bbox

    # 对于90度和270度旋转，交换宽度和高度
    if angle == 90 or angle == 270:
        width, height = height, width

    # 对于180度旋转，翻转中心点
    if angle == 180:
        x_center, y_center = 1 - x_center, 1 - y_center

    # 确保边界框仍在图像内
    x_center = min(max(x_center, width / 2), 1 - width / 2)
    y_center = min(max(y_center, height / 2), 1 - height / 2)

    return [class_id, x_center, y_center, width, height]


def rotate_point_cv2(x, y, angle, cx, cy):
    angle = -np.radians(angle)
    x, y = x - cx, y - cy
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return new_x + cx, new_y + cy

def rotate_point(x, y, angle, cx, cy):
    """
    旋转点 (x, y) 围绕中心点 (cx, cy) 指定角度。
    """
    angle = -math.radians(angle)
    x, y = x - cx, y - cy
    new_x = x * math.cos(angle) - y * math.sin(angle)
    new_y = x * math.sin(angle) + y * math.cos(angle)
    return new_x + cx, new_y + cy

def rotate_bbox_fined(bbox, angle, original_width, original_height, rotated_width, rotated_height):
    class_id, x_center, y_center, width, height = bbox

    # 将YOLO格式转换为像素值
    x_center *= original_width
    y_center *= original_height
    width *= original_width
    height *= original_height

    # 计算旋转前后中心点的偏移
    shift_x = (rotated_width - original_width) / 2
    shift_y = (rotated_height - original_height) / 2

    # 更新边界框坐标以考虑中心点偏移
    x_center += shift_x
    y_center += shift_y

    # 计算边界框的角点
    xmin = x_center - width / 2
    xmax = x_center + width / 2
    ymin = y_center - height / 2
    ymax = y_center + height / 2

    # 旋转四个角点
    rotated_corners = [
        rotate_point(xmin, ymin, angle, rotated_width / 2, rotated_height / 2),
        rotate_point(xmax, ymin, angle, rotated_width / 2, rotated_height / 2),
        rotate_point(xmax, ymax, angle, rotated_width / 2, rotated_height / 2),
        rotate_point(xmin, ymax, angle, rotated_width / 2, rotated_height / 2)
    ]

    # 找到旋转后的边界框的角点
    rxmin, rymin = map(min, zip(*rotated_corners))
    rxmax, rymax = map(max, zip(*rotated_corners))

    # 计算旋转后的中心点和尺寸，并转换回比例值
    new_x_center = ((rxmin + rxmax) / 2) / rotated_width
    new_y_center = ((rymin + rymax) / 2) / rotated_height
    new_width = (rxmax - rxmin) / rotated_width
    new_height = (rymax - rymin) / rotated_height

    return [class_id, new_x_center, new_y_center, new_width, new_height]

def rotate_bbox_cv2(bbox, angle, img_width, img_height):
    class_id, x_center, y_center, width, height = bbox

    # 将比例坐标转换为像素坐标
    x_center = x_center * img_width
    y_center = y_center * img_height
    width = width * img_width
    height = height * img_height

    # 计算边界框的角点
    xmin = x_center - width / 2
    xmax = x_center + width / 2
    ymin = y_center - height / 2
    ymax = y_center + height / 2

    # 旋转四个角点
    rotated_corners = [
        rotate_point_cv2(xmin, ymin, angle, img_width / 2, img_height / 2),
        rotate_point_cv2(xmax, ymin, angle, img_width / 2, img_height / 2),
        rotate_point_cv2(xmax, ymax, angle, img_width / 2, img_height / 2),
        rotate_point_cv2(xmin, ymax, angle, img_width / 2, img_height / 2)
    ]

    # 找到旋转后的边界框的角点
    rxmin, rymin = map(min, zip(*rotated_corners))
    rxmax, rymax = map(max, zip(*rotated_corners))

    # 计算旋转后的中心点和尺寸，并转换回比例值
    new_x_center = ((rxmin + rxmax) / 2) / img_width
    new_y_center = ((rymin + rymax) / 2) / img_height
    new_width = (rxmax - rxmin) / img_width
    new_height = (rymax - rymin) / img_height

    return [class_id, new_x_center, new_y_center, new_width, new_height]

def rotate_image_cv2(image, angle):
    # 读取图像
    height, width = image.shape[:2]

    # 计算旋转的中心点
    center = (width / 2, height / 2)

    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

    # 计算新的边界尺寸
    cos_val = np.abs(rotation_matrix[0, 0])
    sin_val = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))

    # 调整旋转矩阵以考虑平移
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 执行仿射变换（旋转）
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    # 裁剪图像以使尺寸不变
    start_x = max(0, int((new_width - width) / 2))
    start_y = max(0, int((new_height - height) / 2))
    end_x = start_x + width
    end_y = start_y + height
    cropped_rotated_image = rotated_image[start_y:end_y, start_x:end_x]

    return cropped_rotated_image

def roate_image_bbox_v2(img_path, txt_path, angle=0, save_root_path = "aug_data/",pre_fix = "roate"):
    image = cv2.imread(image_path)
    roate_image = rotate_image_cv2(image, angle)    
    height, width = image.shape[:2]
    annotations = load_yolo_annotations(txt_path)
    new_bboxes = [rotate_bbox_cv2(bbox, angle, width, height) for bbox in annotations]
    img_name = os.path.split(img_path)[1]
    save_jpg_name = save_root_path + pre_fix + str(angle)+img_name
    save_txt_name = save_jpg_name[:-3] + "txt" 
    cv2.imwrite(save_jpg_name, roate_image)
    save_yolo_annotations(new_bboxes, save_txt_name)
    return

# 加载图片和标注
def hori_flip(img_path, txt_path, save_root_path = "aug_data/",pre_fix = "hori-"):
    # 进行数据增强操作
    # 例如：水平翻转
    img_name = os.path.split(img_path)[1]
    image = Image.open(img_path)
    annotations = load_yolo_annotations(txt_path)
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_annotations = flip_annotations_horizontally(annotations, image.width)
    # 保存增强后的图片和标注
    save_jpg_name = save_root_path + pre_fix + img_name
    save_txt_name = save_jpg_name[:-3] + "txt"
    flipped_image.save(save_jpg_name)
    save_yolo_annotations(flipped_annotations, save_txt_name)

def veri_flip(img_path, txt_path, save_root_path = "aug_data/",pre_fix = "veri-"):
    # 进行数据增强操作
    # 例如：竖直翻转
    img_name = os.path.split(img_path)[1]
    image = Image.open(img_path)
    annotations = load_yolo_annotations(txt_path)
    flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
    flipped_annotations = flip_annotations_vertically(annotations, image.height)
    # 保存增强后的图片和标注
    save_jpg_name = save_root_path + pre_fix + img_name
    save_txt_name = save_jpg_name[:-3] + "txt" 
    flipped_image.save(save_jpg_name)
    save_yolo_annotations(flipped_annotations, save_txt_name)

def smart_random_translate_image_and_annotations(img_path, txt_path, max_dx_ratio=0.2, max_dy_ratio=0.2, 
                                                    save_root_path = "aug_data/",pre_fix = "translate"):
    """
    按比例随机平移图像及其边界框。
    如果边界框中心点在图像内，则裁剪边界框以适应图像边缘。
    如果中心点不在图像内，返回增强失败的标志。
    """
    image = Image.open(img_path)
    annotations = load_yolo_annotations(txt_path)
    img_width, img_height = image.size

    # 计算随机平移量
    dx = random.uniform(-max_dx_ratio, max_dx_ratio) * img_width
    dy = random.uniform(-max_dy_ratio, max_dy_ratio) * img_height

    # 平移图像
    translated_image = create_image_with_random_background(image.size)
    translated_image.paste(image, (int(dx), int(dy)))

    translated_annotations = []
    augmentation_failed = False
    for ann in annotations:
        class_id, x_center, y_center, width, height = ann

        # 更新边界框位置
        new_x_center = (x_center * img_width + dx) / img_width
        new_y_center = (y_center * img_height + dy) / img_height

        # 确保新的边界框中心点在图像内
        if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1:
            # 调整边界框大小以确保其不超出图像边缘
            xmin = max(new_x_center - width / 2, 0)
            xmax = min(new_x_center + width / 2, 1)
            ymin = max(new_y_center - height / 2, 0)
            ymax = min(new_y_center + height / 2, 1)

            # 更新边界框大小
            new_width = xmax - xmin
            new_height = ymax - ymin

            translated_annotations.append([class_id, (xmin + xmax) / 2, (ymin + ymax) / 2, new_width, new_height])
        else:
            # 中心点不在图像内，增强失败
            augmentation_failed = True
            print("out of bbox: {}".format(img_path))
            return  augmentation_failed
    strdxdy = str(round(dx,2)+round(dy,2))
    pre_fix = "translate"+strdxdy+"-"
    img_name = os.path.split(img_path)[1]
    save_jpg_name = save_root_path + pre_fix + img_name
    save_txt_name = save_jpg_name[:-3] + "txt" 
    translated_image.save(save_jpg_name)
    save_yolo_annotations(translated_annotations, save_txt_name)
    return  augmentation_failed

def add_scaled_bbox_to_image(source_image_path, source_bbox_path, target_image_path, target_bboxes_path, scale_factor,
                            save_root_path = "aug_data/"):
    """
    从源图像中扣出一个边界框区域，调整大小后加到目标图像上。
    确保新加的区域不与目标图像上的已有边界框重叠。
    """
    # 计算源边界框的绝对坐标
    source_image = Image.open(source_image_path)
    source_bboxes = load_yolo_annotations(source_bbox_path)

    if not source_bboxes:
        return None, None
    source_bbox = random.choice(source_bboxes)

    class_num = source_bbox[0]
    source_bbox = source_bbox[1:]
    target_image = Image.open(target_image_path)
    target_bboxes = load_yolo_annotations(target_bboxes_path)
    src_xmin = int((source_bbox[0] - source_bbox[2] / 2) * source_image.width)
    src_xmax = int((source_bbox[0] + source_bbox[2] / 2) * source_image.width)
    src_ymin = int((source_bbox[1] - source_bbox[3] / 2) * source_image.height)
    src_ymax = int((source_bbox[1] + source_bbox[3] / 2) * source_image.height)
    # 提取边界框区域
    cropped_region = source_image.crop((src_xmin, src_ymin, src_xmax, src_ymax))

    # 调整提取区域大小
    new_width = int(cropped_region.width * scale_factor)
    new_height = int(cropped_region.height * scale_factor)
    cropped_region = cropped_region.resize((new_width, new_height), Image.ANTIALIAS)

    angle = random.choice([90, 180, 270])
    rotated_cropped_region = cropped_region.rotate(angle, expand=True)
    #rotated_bbox = rotate_bbox(source_bbox, angle, rotated_cropped_region.width, rotated_cropped_region.height)
    # 随机尝试放置区域
    attempts = 300  # 尝试次数
    for _ in range(attempts):
        try:
            x = random.randint(0, target_image.width - rotated_cropped_region.width)
            y = random.randint(0, target_image.height - rotated_cropped_region.height)
        except:
            break
        new_bbox = [
            class_num,  # 类别
            (x + rotated_cropped_region.width / 2) / target_image.width,
            (y + rotated_cropped_region.height / 2) / target_image.height,
            rotated_cropped_region.width / target_image.width,
            rotated_cropped_region.height / target_image.height
        ]

        # 检查是否允许与现有边界框重叠
        if all(is_overlap_allowed(new_bbox[1:], tb[1:]) for tb in target_bboxes):
            # 将区域放置在目标图像上
            target_bboxes.append(new_bbox)
            target_image.paste(rotated_cropped_region, (x, y))
            img_name = os.path.split(target_image_path)[1]
            pre_fix = "add"+str(scale_factor)+"cls"+str(new_bbox[0])+"pos"+str(x)+str(y)+str(random.randint(0,100))+'-'
            save_jpg_name = save_root_path + pre_fix + img_name
            save_txt_name = save_jpg_name[:-3] + "txt" 
            target_image.save(save_jpg_name)
            save_yolo_annotations(target_bboxes, save_txt_name)
            return target_image, new_bbox
    print("fail to add {} to: {}, with scale:{}".format(source_image_path, target_image_path, scale_factor))
    # 如果找不到合适的位置，返回原始图像和None
    return None


def rotate_image_and_bboxes(img_path, txt_path, angle=0, save_root_path = "aug_data/",pre_fix = "roate"):
    """
    旋转整个图像及其边界框。
    """
    # 旋转图像
    background_colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "dark_green": (0, 100, 0),
        "orange": (255, 165, 0),
        "pink": (255, 192, 203),
        "light_blue": (173, 216, 230),
        "dark_red": (139, 0, 0),
        "turquoise": (64, 224, 208),
        "magenta": (255, 0, 255),
        "gold": (255, 215, 0),
        "lime": (0, 255, 0),
        "teal": (0, 128, 128),
        "olive": (128, 128, 0),       # 橄榄色
        "maroon": (128, 0, 0),        # 栗色
        "navy": (0, 0, 128),          # 海军蓝
        "coral": (255, 127, 80),      # 珊瑚色
        "mint": (189, 252, 201),      # 薄荷色
        "orchid": (218, 112, 214),    # 兰花色
        "beige": (245, 245, 220),     # 米色
        "mustard": (255, 219, 88),    # 芥末色
        "lavender": (230, 230, 250),  # 薰衣草色
        "salmon": (250, 128, 114),    # 鲑鱼色
        "charcoal": (54, 69, 79),     # 木炭色
        "emerald": (80, 200, 120)     # 翡翠色
    }

    # 随机选择一个背景颜色
    color_name = random.choice(list(background_colors.keys()))
    bg_color = background_colors[color_name]
    image = Image.open(img_path)
    annotations = load_yolo_annotations(txt_path)


    image_rgba = image.convert('RGBA')

    # 旋转图像，使用透明背景
    rotated_image = image_rgba.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))

    # 创建一个新的RGB图像并填充背景颜色
    new_image = Image.new("RGB", rotated_image.size, bg_color)

    # 将旋转后的图像粘贴到新图像上
    new_image.paste(rotated_image, (0, 0), rotated_image)
    # rotated_image = image.rotate(angle, expand=True)
    # new_image = Image.new("RGB", rotated_image.size, bg_color)
    # new_image.paste(rotated_image, (0, 0))
    # 更新边界框
    rotated_bboxes = []
    rotated_bboxes = [rotate_bbox_fined(bbox, angle, image.width, image.height, rotated_image.width, rotated_image.height) for bbox in annotations]
    # for bbox in annotations:
    #     rotated_bbox = rotate_bbox_fined(bbox, angle, image.width, image.height)
    #     rotated_bboxes.append(rotated_bbox)
    pre_fix = "roate"+str(angle)+color_name+"-"
    save_data(new_image, rotated_bboxes, img_path, save_root_path, pre_fix)
    return

def convert_to_black_white_rgb(img_path, txt_path, save_root_path = "aug_data/",pre_fix = "black-"):
    # 使用PIL读取图像
    image = Image.open(img_path)
    annotations = load_yolo_annotations(txt_path)
    # 转换为灰度图像
    bw_image = image.convert('L')
    # 转换回RGB模式（仍为黑白风格）
    bw_rgb_image = bw_image.convert('RGB')
    
    # 保存黑白风格图像（RGB模式）
    save_data(bw_rgb_image, annotations, img_path, save_root_path, pre_fix)
    
    return bw_rgb_image

def add_random_text_to_image_cv2(image_path, txt_path, save_root_path = "aug_data/",pre_fix = "addtxtnum-"):
    # 使用OpenCV读取图像
    image = cv2.imread(image_path)
    annotations = load_yolo_annotations(txt_path)
    height, width, _ = image.shape
    
    # 生成随机文本
    text = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(6, 12)))
    
    # 随机选择文本大小
    font_scale = random.uniform(0.5, 2.0)  # OpenCV字体缩放比例
    font_thickness = random.randint(1, 3)  # 文本线条的粗细
    
    # 获取文本框大小
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # 确定文本位置
    if width - text_width >=2 and height - baseline-text_height>=2:
        x = random.randint(0, width - text_width)
        y = random.randint(text_height, height - baseline)
    else:
        x = random.randint(0, width)
        y = random.randint(0, height)
    
    # 随机选择文本颜色（确保颜色的对比度）
    text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # 将文本添加到图像上
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
    
    # 保存并返回图像路径

    img_name = os.path.split(image_path)[1]
    save_jpg_name = save_root_path + pre_fix + img_name
    save_txt_name = save_jpg_name[:-3] + "txt" 
    cv2.imwrite(save_jpg_name, image)
    save_yolo_annotations(annotations, save_txt_name)
    return 

def add_noise_to_image(image_path, txt_path, save_root_path = "aug_data/",pre_fix = "noise"):
    """
    Adds noise to an image. Supports "gaussian" and "salt_pepper" noise types.
    """
    # 使用OpenCV读取图像
    image = cv2.imread(image_path)
    annotations = load_yolo_annotations(txt_path)
    height, width, channels = image.shape
    noise_type = random.choice(["speckle", "poisson", "uniform","gaussian","salt_pepper"])
    if noise_type == "gaussian":
        mean = 0
        var = random.randint(2,4)
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (height, width, channels))  # 生成高斯噪声
        gaussian = gaussian.reshape(height, width, channels)
        noisy_image = cv2.add(image, gaussian.astype('uint8'))  # 添加噪声到图像
        
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.004
        row, col, _ = image.shape
        s_vs_p = 0.5
        out = np.copy(image)

        # Salt (白色噪声)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper (黑色噪声)
        num_pepper = np.ceil(amount * image.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[tuple(coords)] = 0
        noisy_image = out
    elif noise_type == "speckle":
        gauss = np.random.randn(height, width, channels)
        gauss = gauss.reshape(height, width, channels)    
        noisy_image = image + image * gauss * random.uniform(0.05, 0.2)

    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image * vals) / float(vals)

    elif noise_type == "uniform":
        uniform_noise = np.random.uniform(-15, 15, (height, width, channels))
        noisy_image = cv2.add(image, uniform_noise.astype('uint8'))
    else:
        raise ValueError("Incorrect noise type. Please choose 'gaussian' or 'salt_pepper'.")
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    # 保存并返回图像路径
    img_name = os.path.split(image_path)[1]
    save_jpg_name = save_root_path + pre_fix + noise_type + img_name
    save_txt_name = save_jpg_name[:-3] + "txt" 
    cv2.imwrite(save_jpg_name, noisy_image)
    save_yolo_annotations(annotations, save_txt_name)
    
    return



# 示例使用
if __name__ == "__main__":
    image_path = '/home/dancer/data/cook_match/offical_train_data/data_example/dog/ele_072eb872fdab29c1d8161223afdeddbb.jpg'
    annotation_path = '/home/dancer/data/cook_match/offical_train_data/data_example/dog/ele_072eb872fdab29c1d8161223afdeddbb.txt'
    clear_images_in_folder("aug_data")
    # hori_flip(image_path, annotation_path)
    #     
    # veri_flip(image_path, annotation_path)

    #smart_random_translate_image_and_annotations(image_path, annotation_path)

    #rand_scale = round(random.uniform(0.3, 1.2),3)
    #add_scaled_bbox_to_image(image_path, annotation_path,image_path, annotation_path, rand_scale)

    #roate_image_bbox_v2(image_path, annotation_path, angle)
    #rotate_image_and_bboxes(image_path, annotation_path, angle)
    #convert_to_black_white_rgb(image_path, annotation_path)
    #add_random_text_to_image_cv2(image_path, annotation_path, pre_fix="addtextnum"+str(i)+'-')

    #add_noise_to_image(image_path, annotation_path, save_root_path = "aug_data/",pre_fix = "noise"+str(i))
    for i in range(20):
        add_noise_to_image(image_path, annotation_path, save_root_path = "aug_data/",pre_fix = "noise"+str(i))