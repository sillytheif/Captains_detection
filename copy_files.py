import os
import shutil

def copy_files_flat(src_folder, dest_folder_jpg, dest_folder_txt):
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder_jpg):
        os.makedirs(dest_folder_jpg)
    if not os.path.exists(dest_folder_txt):
        os.makedirs(dest_folder_txt)

    # 遍历源文件夹及其所有子文件夹
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            # 检查文件扩展名
            if file.lower().endswith('.jpg'):
                shutil.copy(os.path.join(root, file), os.path.join(dest_folder_jpg, file))
            elif file.lower().endswith('.txt'):
                shutil.copy(os.path.join(root, file), os.path.join(dest_folder_txt, file))

# 使用示例
src_folder = '/home/dancer/data/cook_match/offical_train_data/v3_prepare'  # 源文件夹路径
dest_folder_jpg = '/home/dancer/data/cook_match/offical_train_data/v3/images'  # JPG文件的目标文件夹路径
dest_folder_txt = '/home/dancer/data/cook_match/offical_train_data/v3/labels'  # TXT文件的目标文件夹路径
copy_files_flat(src_folder, dest_folder_jpg, dest_folder_txt)
