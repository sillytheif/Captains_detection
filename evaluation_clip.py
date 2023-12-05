from ultralytics import YOLO
import os
import csv
import cv2
import numpy as np
import torch
from PIL import Image
import clip
from visualize_bbox import clear_images_in_folder
import random
def read_first_words(file_path):
    first_words = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = line.split()
            if words:
                first_words.add(words[0])
    return first_words

def set_to_decimal(word_set):
    word_indices = [int(word) for word in word_set]
    binary_num = sum([1 << index for index in word_indices])
    return binary_num


def transform_img(img_path):
    image_cv2 = cv2.imread(img_path)

    # 将BGR格式转换为RGB格式
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)

    # 将NumPy数组转换为PIL图像
    image_pil = Image.fromarray(image_rgb)

    # 应用CLIP的预处理函数
    input_tensor = preprocess(image_pil).unsqueeze(0).to(device)  # 添加一个批次维度
    return input_tensor

def write_lists_on_image(image_path, list1, list2, save_path):
    """
    读取图像，将两组整型列表的内容写在图像上（不重叠），并保存。

    :param image_path: 要读取的图像的路径。
    :param list1: 第一组整型列表。
    :param list2: 第二组整型列表。
    :param save_path: 图像保存的路径。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at {image_path}")
        return

    # 设置字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    color1 = (255, 0, 0)  # 蓝色
    color2 = (0, 255, 0)  # 绿色
    font_scale = 1
    thickness = 2
    line_type = cv2.LINE_AA

    # 将第一组列表写在图像上
    vertical_pos = 30
    for value in list1:
        position = (10, vertical_pos)
        cv2.putText(image, str(value), position, font, font_scale, color1, thickness, line_type)
        vertical_pos += 30

    # 将第二组列表写在图像上，确保不重叠
    vertical_pos += 30  # 在第一组和第二组之间添加额外的间隔
    for value in list2:
        position = (10, vertical_pos)
        cv2.putText(image, str(value), position, font, font_scale, color2, thickness, line_type)
        vertical_pos += 30

    # 保存图像
    try:
        cv2.imwrite(save_path, image)
    except:
        import pdb;pdb.set_trace()
    print(f"Image saved to {save_path}")


def crop_and_save_images(model, image_path, tensor, text_features):
    """
    Crops regions from an image based on a tensor in xyxy format and saves them.

    Parameters:
    image_path (str): The path to the source image.
    tensor (list of list or numpy.ndarray): The tensor with coordinates in xyxy format for multiple boxes.
    target_directory (str): The directory where cropped images will be saved.
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the filename and extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Ensure the target directory exists

    # Iterate over each box in the tensor
    similarity = torch.Tensor().cuda()
    for i, box in enumerate(tensor):
        # Ensure box is a numpy array
        box = np.array(box, dtype=np.int32)

        # Extract coordinates
        x1, y1, x2, y2 = box

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]
        image_pil = Image.fromarray(cropped_image)

        # 应用CLIP的预处理函数
        input_tensor = preprocess(image_pil).unsqueeze(0).to(device)  # 添加一个批次维度
        # Save the cropped image
        image_features = model.encode_image(input_tensor)
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        #similarity = (80*image_features @ text_features.T).view(4,5).softmax(dim=1).mean(dim=0).max(dim=0)
        tmp = (100 * image_features @ text_features.T).softmax(dim=-1)
        similarity = torch.cat([similarity, tmp], dim=0)
    return similarity

def crop_draw_det_clip(image_path, tensor, cls_list, classes, clip_output, list2,save_file_name):
    """
    Crops regions from an image based on a tensor in xyxy format and saves them.

    Parameters:
    image_path (str): The path to the source image.
    tensor (list of list or numpy.ndarray): The tensor with coordinates in xyxy format for multiple boxes.
    target_directory (str): The directory where cropped images will be saved.
    """
    # Load the image
    def get_random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    colors = {class_id: get_random_color() for class_id in classes.keys()}
    image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the filename and extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Ensure the target directory exists

    # Iterate over each box in the tensor
    similarity = torch.Tensor().cuda()
    for i, box in enumerate(tensor):
        # Ensure box is a numpy array
        box = np.array(box, dtype=np.int32)

        # Extract coordinates
        x1, y1, x2, y2 = box

        # Crop the image
        class_id = cls_list[i]
        color = colors[int(class_id)]
        class_name = classes[int(class_id)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        position = (int((x1+x2)/2), y1 - 5)
        temp = clip_output[i].cpu().tolist()
        temp = [round(x,2) for x in temp]
        cv2.putText(image, str(temp), position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color1 = (255, 0, 0)  # 蓝色
    color2 = (0, 255, 0)  # 绿色
    font_scale = 1
    thickness = 2
    line_type = cv2.LINE_AA
    vertical_pos = 30
    for value in list2:
        position = (10, vertical_pos)
        cv2.putText(image, str(value), position, font, font_scale, color2, thickness, line_type)
        vertical_pos += 30
    cv2.imwrite(save_file_name, image)
    return 

def fusion_logit(confidence_list, classname_list, clip_similarity, threshold=0.8):
    '''
    confidence_list = float list, n
    classname_list = int list, n
    clip_similarity [n * class_name] tensor
    '''
    enhance_sim = clip_similarity.clone()
    final_result = []
    for i,confidence in enumerate(confidence_list):
        if classname_list[i] == 0:
            final_result.append(classname_list[i])
            continue
        if confidence <= threshold: #大于阈值，直接按检测
            value, indice = clip_similarity[i].max(dim=0)  #找出max值
            if indice.item() == classname_list[i]:  #一致性，按照检测结果
                final_result.append(classname_list[i])
            else:
                enhance_sim[i][classname_list[i]]  = (confidence + clip_similarity[i][classname_list[i]])/2
                if enhance_sim[i][classname_list[i]] > value.item():  # 等价于 检测 + clip_目标 > 2 * clip_max
                    final_result.append(classname_list[i])
                else:
                    if value > 0.5:
                        if indice.item() ==0 or indice.item() == 1:
                            continue
                        else:
                            final_result.append(indice.item())
        else:
            final_result.append(classname_list[i])    
    return final_result
if __name__ == "__main__":
    stage1_class = {0:"smoking",
                1:"nake",
                2:"rat",
                3:"cat",
                4:"dog",
                5:"no_mask"}
    images_path = "/home/dancer/data/cook_match/picture"
    video_path = "/home/dancer/data/cook_match/video"
    save_name = "answer_1205_clip.csv"
    save_folder = "current_cut_images"
    clear_images_in_folder(save_folder)
    # Load a model
    model = YOLO('/home/dancer/code/my_yolo_v8/runs/detect/train20/weights/best.pt')  # pretrained YOLOv8n model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset

    label_class = ["cigarette", "nake man", "rat", "cat", "dog"]
    prompt_1 = ["a a photo of a " + x for x in label_class]
    # prompt_2 = ["A detailed illustration of " + x for x in label_class]
    # prompt_3 = ["This picture contains a " + x for x in label_class]
    # prompt_4 = ["An environment where {} can be found".format(x) for x in label_class]
    # all_prompts = prompt_1 + prompt_2 + prompt_3 + prompt_4

    # 对所有提示词进行Tokenize，并发送到设备
    text_inputs = torch.cat([clip.tokenize(prompt) for prompt in prompt_1]).to(device)
    text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    # Run batched inference on a list of images
    clip_begin_threshold = 0.7

    # # 创建或打开 answer.csv 文件
    results = model(images_path, save=True, classes = [0,1,2,3,4], stream=True)  # return a generator of Results objects
    count=0
    adjust_path = []
    with torch.no_grad():
        with open(save_name, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            for result in results:
                class_id_list = result.boxes.cls.tolist()
                class_id_list = [int(x) for x in class_id_list]
                class_conf_list = result.boxes.conf.tolist()
                similarity = crop_and_save_images(clip_model, result.path, result.boxes.xyxy.cpu().numpy(), text_features)
                first_words_set = fusion_logit(class_conf_list, class_id_list, similarity)
                #first_words_set = result.boxes.cls.tolist()
                # 将首个单词集合转换为十进制数
                file_name = os.path.split(result.path)[1]
                decimal_number = set_to_decimal(set(first_words_set))
                oririganl_first_words_set = result.boxes.cls.tolist()
                oririganl_first_words_set = [int(x) for x in oririganl_first_words_set]
                original_decimal_number = set_to_decimal(set(oririganl_first_words_set))
                if decimal_number != original_decimal_number:
                    print(result.path)
                    adjust_path.append(file_name)
                    save_img_path = os.path.join(save_folder, file_name)
                    #write_lists_on_image(result.path, set(oririganl_first_words_set), set(first_words_set), save_img_path)
                    crop_draw_det_clip(result.path, result.boxes.xyxy.cpu().numpy(), class_id_list, stage1_class, similarity, set(first_words_set), save_img_path)
                    count+=1
                # 将十进制数写入 CSV
                #os.path.split()
                csv_writer.writerow([file_name, decimal_number])
    print("adjust num is:{}".format(count))
    for x in adjust_path:
        print(x)
    print(count)
    results = model(video_path, classes = [0,1,2,3,4], stream=True)  # return a generator of Results objects

    last_name = None
    count_list = [0 for i in range(5)]

    with open(save_name, 'a+', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        for result in results:
            first_words_set = result.boxes.cls.tolist()
            current_list = set(first_words_set)
            # 将首个单词集合转换为十进制数
            current_name = result.path
            if last_name == current_name:
                for x in current_list:
                    count_list[int(x)] = count_list[int(x)] +1
            else:
                if last_name is not None:
                    # 结算
                    first_words_set = [i for i,x in enumerate(count_list) if x >=4]
                    file_name = os.path.split(last_name)[1]
                    decimal_number = set_to_decimal(set(first_words_set))
                    csv_writer.writerow([file_name, decimal_number])
                count_list = [0 for i in range(5)]
                last_name = current_name
                for x in current_list:
                    count_list[int(x)] = count_list[int(x)] +1