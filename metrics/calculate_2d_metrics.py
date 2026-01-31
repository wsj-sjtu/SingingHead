import os
from tqdm import tqdm
import cv2
import json
import shutil
from metric_2d.talking_head_metrics import *


img_size = 256  # or 512
downsample_interval = 6

data_root = '/path/to/eval_folder'
generated_video_root = os.path.join(data_root, 'generated')


## Step1: preprocess generated video to images
# Unify the fps of the videos (30fps), and save the video as images.
if 1:
    tmp_root = os.path.join(data_root, 'video_30fps')
    os.makedirs(tmp_root, exist_ok=True)
    method_list = os.listdir(generated_video_root)

    ## resave generated videos with 30fps
    for method in tqdm(method_list):
        sub_folder = os.path.join(data_root, method)
        tar_folder = os.path.join(tmp_root, method)
        os.makedirs(tar_folder, exist_ok=True)
        file_list = os.listdir(sub_folder)
        for name in file_list:
            src_path = os.path.join(sub_folder, name)
            tar_path = os.path.join(tar_folder, name)
            cmd = "ffmpeg -i " + src_path + " -r 30" + " -qscale 0 -q:v 1 -c:a aac " + tar_path
            os.system(cmd)  

    ## extract images from video
    img_folder_name = f'imgs_{img_size}'
    video_img_root = os.path.join(data_root, img_folder_name)
    method_list = os.listdir(tmp_root)
    for method in tqdm(method_list):
        sub_folder = os.path.join(tmp_root, method)
        tar_folder = os.path.join(video_img_root, method)
        os.makedirs(tar_folder, exist_ok=True)
        file_list = os.listdir(sub_folder)
        for name in file_list:
            video_path = os.path.join(sub_folder, name)
            video = cv2.VideoCapture(video_path)
            cnt = 0
            while True:
                ret, frame = video.read()
                if ret is False:
                    break
                # resize
                if frame.shape[0] != img_size:
                    frame = cv2.resize(frame, (img_size,img_size))
                img_save_path = os.path.join(tar_folder, name.replace('.mp4', '_'+str(cnt).zfill(3)+'.jpg'))
                cv2.imwrite(img_save_path, frame)
                cnt += 1
            video.release()   

    ## downsample images
    sampled_imgs_root = os.path.join(data_root, f'{img_folder_name}_sampled_{downsample_interval}')
    method_list = os.listdir(video_img_root)
    for method in tqdm(method_list, desc='Down sampling'):
        sampled_folder = os.path.join(sampled_imgs_root, method)
        os.makedirs(sampled_folder, exist_ok=True)

        img_folder = os.path.join(video_img_root, method)
        img_name_list = os.listdir(img_folder)
        seq_name_list = [name.replace(name.split('_')[-1], '')[:-1] for name in img_name_list]
        seq_name_list = list(set(seq_name_list))

        for seq_name in tqdm(seq_name_list, desc='Seq'):
            frame_list = [name for name in img_name_list if name.replace(name.split('_')[-1], '')[:-1]==seq_name]
            frame_list.sort(key=lambda x: [int(y) for y in x.split('_')[-1].replace('.jpg','')])    # time order

            for i in range(len(frame_list)):
                if i % downsample_interval == 0:
                    src_path = os.path.join(img_folder, frame_list[i])
                    tar_path = os.path.join(sampled_folder, frame_list[i])
                    shutil.copyfile(src_path, tar_path)



## Step2: calculate metrics 
if 1:       # 计算LMD比较慢所以单独拿出来
    imgs_root = os.path.join(data_root, f'imgs_{img_size}_sampled_{downsample_interval}')
    video_root = generated_video_root
    audio_folder = os.path.join(data_root, 'input_audio')
    result_root = os.path.join(data_root, f'metric_results_{img_size}_{downsample_interval}')

    os.makedirs(result_root, exist_ok=True)
    gt_folder = os.path.join(imgs_root, 'gt')

    method_list = os.listdir(imgs_root)

    for method in tqdm(method_list, desc='Method'):

        img_folder = os.path.join(imgs_root, method)

        ## compute metrics
        fid = compute_fid(img_folder, gt_folder)
        psnr, ssim = compute_psnr_ssim(img_folder, gt_folder)
        cpbd = compute_cpbd(img_folder)
        lpips = compute_lpips(img_folder, gt_folder)
        flmd, mlmd = compute_lmd(img_folder, gt_folder)

        ## compute ba
        video_folder = os.path.join(video_root, method)
        tmp_folder = os.path.join('./tmp/', str(img_size), method)
        os.makedirs(tmp_folder, exist_ok=True)
        ba = compute_ba(video_folder, img_size, audio_folder=audio_folder) if method=='gt' else compute_ba(video_folder, img_size, tmp_folder=tmp_folder)

        # save results
        result_save_path = os.path.join(result_root, method+'.json')
        result_dict = {
            'FID': fid, 'PSNR': psnr, 'SSIM':ssim, 'CPBD': cpbd, 'LPIPS':lpips,
            'F-LMD':flmd, 'M-LMD':mlmd, 'BA':ba
        }

        with open(result_save_path, "w") as file:
            json.dump(result_dict, file)
        
        


