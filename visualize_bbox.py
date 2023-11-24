from tkinter import image_names
import cv2
import random
import os
import glob

def draw_bounding_boxes(image_path, annotation_path, output_path, classes):
    # 随机颜色生成器
    def get_random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # 为每个类别生成颜色
    colors = {class_id: get_random_color() for class_id in classes.keys()}

    # 读取图像
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # 读取标注
    with open(annotation_path, 'r') as file:
        annotations = file.readlines()

    # 绘制每个边界框
    for annotation in annotations:
        class_id, x_center, y_center, box_width, box_height = map(float, annotation.split())
        x_center, y_center, box_width, box_height = x_center * width, y_center * height, box_width * width, box_height * height

        top_left_x = int(x_center - box_width / 2)
        top_left_y = int(y_center - box_height / 2)

        bottom_right_x = int(x_center + box_width / 2)
        bottom_right_y = int(y_center + box_height / 2)

        # 获取类别名称和颜色
        class_name = classes[int(class_id)]
        color = colors[int(class_id)]

        # 画框和类别名称
        cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
        cv2.putText(image, class_name, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 保存带标注的图像
    jpg_name = os.path.split(image_path)[1]
    cv2.imwrite(os.path.join(output_path, jpg_name), image)

def clear_images_in_folder(folder_path):
    # 检查文件夹中的图像文件（假设图像格式为jpg, png, jpeg）
    image_files = glob.glob(folder_path + '/*.jpg') + glob.glob(folder_path + '/*.png') + glob.glob(folder_path + '/*.txt')
    # 检查是否有图像文件
    if len(image_files) > 0:
        print(f"找到 {len(image_files)} 个图像文件，即将清空文件夹...")
        # 删除每个图像文件
        for file in image_files:
            os.remove(file)
        print("文件夹已清空。")
    else:
        print("文件夹中没有图像文件。")
def according_img_find_txt(images_path, txt_path, save_path, classes):
    image_name_list = [x for x in os.listdir(images_path) if x[-3:]=="jpg"]
    for image_name in image_name_list:
        i_path = os.path.join(images_path, image_name)
        t_path = os.path.join(txt_path, image_name[:-3]+"txt")
        if not os.path.isfile(t_path):
            print("warning! {} does not exist".format(t_path))
            continue
        draw_bounding_boxes(i_path, t_path, save_path, classes)

def according_txt_find_img(images_path, txt_path, save_path, classes):
    text_name_list = [x for x in os.listdir(txt_path) if x[-3:]=="txt"]
    for image_name in text_name_list:
        t_path = os.path.join(txt_path, image_name)
        i_path = os.path.join(images_path, image_name[:-3]+"jpg")
        if not os.path.isfile(i_path):
            print("warning! {} does not exist".format(t_path))
            continue
        draw_bounding_boxes(i_path, t_path, save_path, classes)


if __name__ == "__main__":
    # 类别字典
    classes = {
        0: "smoke",
        1: "nake",
        2: "rat",
        3: "cat",
        4: "dog",
        5: "no_mask",
        6: "trash_can",
        7: "occulusion"
    }
    coco_classes = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
    }
    mouth_class = {0:"mouth"}
    smoking_class = {0:"no",
                     1:"smkoing"}
    stage1_class = {0:"smoking",
                    1:"nake",
                    2:"rat",
                    3:"cat",
                    4:"dog"}
    # 使用示例
    nake_class = {0:"nake",
                  1:"no_maks"}
    images_path = "/home/dancer/data/cook_match/offical_train_data/v4/images"
    txt_path = "/home/dancer/data/cook_match/offical_train_data/v4/labels"
    save_path = "see_box"
    clear_images_in_folder(save_path)
    according_txt_find_img(images_path, txt_path, save_path, stage1_class)
