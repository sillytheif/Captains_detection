import os
from my_au import hori_flip, veri_flip, smart_random_translate_image_and_annotations, add_scaled_bbox_to_image
from my_au import rotate_image_and_bboxes, convert_to_black_white_rgb, add_random_text_to_image_cv2, add_noise_to_image


if __name__ == "__main__":
    class_id = ["cat", "dog", "nake", "nothing", "rat", "smoke"]  # 90 images
    original_path = "/home/dancer/data/cook_match/offical_train_data/data_example"
    
    # stage1 augmentation by 上下、左右、黑夜、加文字
    save_path_v1 = "/home/dancer/data/cook_match/offical_train_data/v3_prepare"
    #"apply 水平、竖直和黑化"
    for class_name in class_id:
        sub_path = os.path.join(original_path, class_name)
        save_sub_path = os.path.join(save_path_v1, class_name) + "/"
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)
        jpg_name = [x for x in os.listdir(sub_path) if x[-3:]=="jpg"]
        for img_name in jpg_name:
            img_path = os.path.join(sub_path, img_name)
            txt_path = img_path[:-3] + "txt"
            hori_flip(img_path, txt_path, save_root_path=save_sub_path)
            veri_flip(img_path, txt_path, save_root_path=save_sub_path)
            convert_to_black_white_rgb(img_path, txt_path, save_root_path=save_sub_path)
    
    original_path = save_path_v1
    
    for class_name in class_id:
        sub_path = os.path.join(original_path, class_name)
        save_sub_path = os.path.join(save_path_v1, class_name) + "/"
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)
        jpg_name = [x for x in os.listdir(sub_path) if x[-3:]=="jpg"]
        for img_name in jpg_name:
            img_path = os.path.join(sub_path, img_name)
            txt_path = img_path[:-3] + "txt"
            add_random_text_to_image_cv2(img_path, txt_path, save_root_path = save_sub_path,pre_fix="addtextnum"+str(0)+'-')
    
    for class_name in class_id:
        sub_path = os.path.join(original_path, class_name)
        save_sub_path = os.path.join(save_path_v1, class_name) + "/"
        if not os.path.exists(save_sub_path):
            os.makedirs(save_sub_path)
        jpg_name = [x for x in os.listdir(sub_path) if x[-3:]=="jpg"]
        for img_name in jpg_name:
            img_path = os.path.join(sub_path, img_name)
            txt_path = img_path[:-3] + "txt"
            smart_random_translate_image_and_annotations(img_path, txt_path, save_root_path = save_sub_path)
