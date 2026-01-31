"""
calculate FID, PSNR, SSIM, LPIPS, CPBD, CSIM, NIQE, BRISQUE; LMD; Diversity, BA
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio
import skimage
import torch

from cleanfid import fid
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import cpbd
import lpips
import face_alignment
import essentia
from essentia.standard import *
from metric_2d.beat_align_score import calc_ba_score
import moviepy.editor as mov 


def compute_fid(img_folder, gt_folder):
    
    fid_val = fid.compute_fid(img_folder, gt_folder)
    return fid_val



def compute_psnr_ssim(img_folder, gt_folder):

    name_list = os.listdir(img_folder)
    img_num = len(name_list)
    
    psnr_sum = 0.0
    ssim_sum = 0.0
    for name in tqdm(name_list, desc='Calculating PSNR and SSIM'):
        img = cv2.imread(os.path.join(img_folder, name))
        gt_img = cv2.imread(os.path.join(gt_folder, name))

        # PSNR
        psnr = peak_signal_noise_ratio(gt_img, img)
        psnr_sum += psnr
        # SSIM
        ssim = structural_similarity(gt_img, img, channel_axis=2)
        ssim_sum += ssim

    psnr = psnr_sum / img_num
    ssim = ssim_sum / img_num
    return psnr, ssim



def compute_cpbd(img_folder):

    name_list = os.listdir(img_folder)
    img_num = len(name_list)
    
    cpbd_sum = 0.0
    for name in tqdm(name_list, desc='Calculating CPBD'):
        img = imageio.imread(os.path.join(img_folder, name), pilmode='L')

        cpbd_val = cpbd.compute(img)
        cpbd_sum  += cpbd_val

    cpbd_val = cpbd_sum / img_num
    return cpbd_val
    


def compute_lpips(img_folder, gt_folder):

    lpips_model = lpips.LPIPS(net='alex')

    name_list = os.listdir(img_folder)
    img_num = len(name_list)
    
    lpips_sum = 0.0
    for name in tqdm(name_list, desc='Calculating LPIPS'):
        img = Image.open(os.path.join(img_folder, name))
        gt_img = Image.open(os.path.join(gt_folder, name))
        # to tensor
        img = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        gt_img = torch.tensor(np.array(gt_img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        lpips_val = lpips_model(img, gt_img).item()
        lpips_sum += lpips_val

    lpips_val = lpips_sum / img_num
    return lpips_val




def compute_lmd(img_folder, gt_folder, img_size=512):
    """
    calculate LMD for mouth landmarks and all landmarks
    """

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

    name_list = os.listdir(img_folder)
    img_num = len(name_list)
    
    flmd_sum = 0.0
    mlmd_sum = 0.0
    for name in tqdm(name_list, desc='Calculating LPIPS'):
        real_img = skimage.io.imread(os.path.join(gt_folder, name))
        gen_img = skimage.io.imread(os.path.join(img_folder, name))

        # detect lmk
        try:
            lmk_real = fa.get_landmarks(real_img)[0] / img_size
            lmk_gen = fa.get_landmarks(gen_img)[0] / img_size
            lmk_real_lip = lmk_real[48:]
            lmk_gen_lip = lmk_gen[48:]
        except:
            continue

        # normalize
        lmk_real -= lmk_real.mean(0)
        lmk_gen -= lmk_gen.mean(0)
        lmk_real_lip -= lmk_real_lip.mean(0)
        lmk_gen_lip -= lmk_gen_lip.mean(0)

        # calculate distance
        flmd = np.sqrt(((lmk_real - lmk_gen)**2).sum(1)).mean() * img_size
        flmd_sum += flmd
        mlmd = np.sqrt(((lmk_real_lip - lmk_gen_lip)**2).sum(1)).mean() * img_size
        mlmd_sum += mlmd

    flmd = flmd_sum / img_num
    mlmd = mlmd_sum / img_num

    return flmd, mlmd



def compute_ba(video_folder, img_size, tmp_folder=None, audio_folder=None):

    if audio_folder is None:
        assert tmp_folder != None

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')

    seq_name_list = os.listdir(video_folder)
    seq_name_list = [name.replace('.mp4','') for name in seq_name_list]

    ba_list = []
    for seq_name in tqdm(seq_name_list, desc='Calculating BA'):
        video_path = os.path.join(video_folder, seq_name+'.mp4')
        video = cv2.VideoCapture(video_path)

        if audio_folder is None:
            audio_path = os.path.join(tmp_folder, seq_name+'.wav')
            video_clip = mov.VideoFileClip(video_path)
            audio = video_clip.audio
            audio.write_audiofile(audio_path)
        else:
            audio_path = os.path.join(audio_folder, seq_name+'.wav') 

        # ref:https://github.com/lisiyao21/Bailando/blob/main/_prepro_aistpp_music.py
        loader = essentia.standard.MonoLoader(filename=audio_path, sampleRate=16000)
        audio = loader()
        audio = np.array(audio).T

        seq_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        seq_lmk = np.zeros((seq_len, 68, 2))
        for i in range(seq_len):
            _, frame = video.read()
            if frame.shape[0] != img_size:
                frame = cv2.resize(frame, (img_size,img_size))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # detect lmk
            try:
                lmk = fa.get_landmarks(img)[0]
                seq_lmk[i] = lmk
            except:
                if i==0:
                    break
                else:
                    seq_lmk[i] = seq_lmk[i-1]

        if np.any(seq_lmk):
            ba = calc_ba_score(seq_lmk, audio)
            ba_list.append(ba)

    ba = np.mean(ba_list)
    return ba
