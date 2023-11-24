from ultralytics import YOLO
import os
import csv
images_path = "/home/dancer/data/cook_match/picture"
video_path = "/home/dancer/data/cook_match/video"
save_name = "answer_1122_m1.csv"


# Load a model
model = YOLO('/home/dancer/code/my_yolo_v8/runs/detect/train9/weights/best.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images

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

# # 创建或打开 answer.csv 文件
results = model(images_path, save=True,classes = [0,1,2,3,4], stream=True)  # return a generator of Results objects
with open(save_name, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    for result in results:
        first_words_set = result.boxes.cls.tolist()
        # 将首个单词集合转换为十进制数
        file_name = os.path.split(result.path)[1]
        decimal_number = set_to_decimal(first_words_set)
        # 将十进制数写入 CSV
        #os.path.split()
        csv_writer.writerow([file_name, decimal_number])

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
                first_words_set = [i for i,x in enumerate(count_list) if x >=3]
                file_name = os.path.split(last_name)[1]
                decimal_number = set_to_decimal(first_words_set)
                csv_writer.writerow([file_name, decimal_number])
            count_list = [0 for i in range(5)]
            last_name = current_name
            for x in current_list:
                count_list[int(x)] = count_list[int(x)] +1