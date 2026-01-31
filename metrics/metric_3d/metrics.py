"""
Reference: https://github.com/Khrylx/DLow/blob/master/motion_pred/eval.py
"""


import os
import numpy as np
import pandas
import torch
import torch.nn.functional as F

from src.evaluate.metric.beat_align_score import calc_ba_score_batch



"""
for all metrics
"""
class MetricCalculator():
    def __init__(self, metric_root):
        
        # flame mask
        flame_mask = pandas.read_pickle(os.path.join(metric_root, 'FLAME_masks.pkl'))
        upper_face_mask = np.concatenate([flame_mask['forehead'], flame_mask['eye_region']], axis=0)
        self.flame_mask = {'face': flame_mask['face'], 'lip': flame_mask['lips'], 'upper_face':upper_face_mask}


    ## --------------------------- for mesh ---------------------------- ##
    ## ----------------------------------------------------------------- ##
        
    def calculate_fdd(self, pred, gt):
        """
        pred: (bs, nsample, nframe, 15069)
        gt: (bs, nframe, 15069)
        ref: https://github.com/Doubiiu/CodeTalker/blob/main/main/cal_metric.py
        """

        if len(pred.shape) != 4:      # for methods that cannot generate diverse results
            pred = pred.unsqueeze(1)

        bs, nsample, seq_len = pred.shape[0], pred.shape[1], pred.shape[2]

        pred_upper = pred.contiguous().view(bs, nsample, seq_len, 5023, 3)[:, :, :, self.flame_mask['upper_face'], :]   # (bs,nsample,nframe,nvtx,3)
        gt_upper = gt.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['upper_face'], :]

        # pred
        l2_dist = pred_upper ** 2
        l2_dist = torch.sum(l2_dist, dim=-1)    # (bs,nsample,nframe,nvtx)
        pred_std = torch.std(l2_dist, dim=-2)   # (bs,nsample,nvtx)
        pred_std = torch.mean(pred_std)
        # gt
        l2_dist = gt_upper ** 2
        l2_dist = torch.sum(l2_dist, dim=-1)
        gt_std = torch.std(l2_dist, dim=-2)
        gt_std = torch.mean(gt_std)

        fdd = gt_std - pred_std

        return fdd.item()
    

    def calculate_lve(self, pred, gt, type='mean'):
        '''
        lip vertex error 
        pred: (bs, nsample, nframe, 15069)
        gt: (bs, nframe, 15069)
        type: mean/min   mean error or min error when diverse generation
        '''

        if len(pred.shape) != 4:      # for methods that cannot generate diverse results
            pred = pred.unsqueeze(1)

        bs, nsample, seq_len = pred.shape[0], pred.shape[1], pred.shape[2]

        pred_lip = pred.contiguous().view(bs, nsample, seq_len, 5023, 3)[:, :, :, self.flame_mask['lip'], :]   # (bs,nsample,nframe,nvtx,3)
        gt_lip = gt.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['lip'], :]
        gt_lip = gt_lip.unsqueeze(1).repeat(1, nsample, 1, 1, 1)

        l2_dist = torch.norm(pred_lip - gt_lip, p=2, dim=-1)
        l2_dist, _ = torch.max(l2_dist, dim=-1) 
        l2_dist = torch.mean(l2_dist, dim=-1)

        if type == 'mean':
            lve = torch.mean(l2_dist)
        else:               
            l2_dist, _ = torch.min(l2_dist, dim=1)
            lve = torch.mean(l2_dist)

        return lve.item()
    

    def calculate_ve(self, pred, gt, type='mean'):
        '''
        face vertex error
        pred: (bs, nsample, nframe, 15069)
        gt: (bs, nframe, 15069)
        '''

        if len(pred.shape) != 4:      # for methods that cannot generate diverse results
            pred = pred.unsqueeze(1)

        bs, nsample, seq_len = pred.shape[0], pred.shape[1], pred.shape[2]

        pred_face = pred.contiguous().view(bs, nsample, seq_len, 5023, 3)[:, :, :, self.flame_mask['face'], :]   # (bs,nsample,nframe,nvtx,3)
        gt_face = gt.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['face'], :]
        gt_face = gt_face.unsqueeze(1).repeat(1, nsample,1, 1, 1)

        l2_dist = torch.norm(pred_face - gt_face, p=2, dim=-1)

        if type == 'mean':   
            ve = torch.mean(l2_dist, dim=1)
        else:                 
            ve, _ = torch.min(l2_dist, dim=1)
        
        ve = torch.mean(ve) 
        return ve.item()
    

    

    ## -------------------------- for flame parameters ------------------------- ##
    ## ------------------------------------------------------------------------- ##

    def compute_diversity(self, pred):
        """
        pred: 
        for all flame params (batch_size, nsample, nframe, 59) 
        for flame pose params (batch_size, nsample, nframe, 6)
        for vertices (batch_size, nsample, nframe, nvtx*3)
        for lmk (batch_size, nsample, nframe, 68*3)
        """
        if pred.shape[1] == 1:
            return 0.0
        
        batch_size, nsample, nframe, _ = pred.shape
        pred = pred.reshape(batch_size, nsample, -1)
        dist = 0 
        for i in range(batch_size):
            dist += F.pdist(pred[i]).mean().item()
        
        return dist / batch_size


    def compute_mindst(self, pred, gt):
        """
        Distance between the ground truth and the closest generated sample.
        """
        gt = gt.unsqueeze(1).repeat(1, pred.shape[1], 1, 1)
        diff = pred - gt
        dist = torch.norm(diff, dim=3).mean(dim=2).min(dim=1)[0] 
        return dist.mean().item()


    def compute_meandist(self, pred, gt):
        """
        Mean distance between generated samples and ground truth.
        """
        gt = gt.unsqueeze(1).repeat(1, pred.shape[1], 1, 1)
        diff = pred - gt
        dist = torch.norm(diff, dim=3).mean()
        return dist.item()

    
    def compute_mindst_pose(self, pred, gt):
        """
        Distance between the ground truth and the closest generated sample.
        Only calculate for global+neck pose (6)

        pred: (batch_size, nsample, nframe, 59)
        gt: (batch_size, nframe, 59)
        """
        pred_pose = pred[..., 50:56]
        gt_pose = gt[..., 50:56]
        gt_pose = gt_pose.unsqueeze(1).repeat(1, pred_pose.shape[1], 1, 1)
        diff = pred_pose - gt_pose
        dist = torch.norm(diff, dim=3).mean(dim=2).min(dim=1)[0]
        return dist.mean().item()


    def compute_meandist_pose(self, pred, gt):
        """
        Mean distance between generated samples and ground truth.
        Only calculate for global+neck pose (6)

        pred: (batch_size, nsample, nframe, 59)
        gt: (batch_size, nframe, 59)
        """
        pred_pose = pred[..., 50:56]
        gt_pose = gt[..., 50:56]
        gt_pose = gt_pose.unsqueeze(1).repeat(1, pred_pose.shape[1], 1, 1)
        diff = pred_pose - gt_pose
        dist = torch.norm(diff, dim=3).mean()
        return dist.item()
    


    ## -------------------------- audio related metric ------------------------- ##
    ## ------------------------------------------------------------------------- ##

    def calculate_ba_lmk(self, pred_lmk, audio, sr=16000):
        """
        audio: (bs, n)
        pred_lmk: (bs, nsample, nframe, 68, 3)
        """

        if len(pred_lmk.shape) != 5:      # for methods that cannot generate diverse results
            pred_lmk = pred_lmk.unsqueeze(1)

        nsample = pred_lmk.shape[1]
        ba_list = []
        for i in range(nsample):
            ba = calc_ba_score_batch(pred_lmk[:,i,:,:,:], audio, sr=sr)
            ba_list.append(ba)

        return np.mean(ba_list)
    

    def calculate_ba_pose(self, pred_pose, audio, sr=16000):
        """
        audio: (bs, n)
        pred_pose: (bs, nsample, nframe, 6)
        """

        if len(pred_pose.shape) != 4:      # for methods that cannot generate diverse results
            pred_pose = pred_pose.unsqueeze(1)

        nsample = pred_pose.shape[1]
        ba_list = []
        for i in range(nsample):
            ba = calc_ba_score_batch(pred_pose[:,i,:,:], audio, sr=sr)
            ba_list.append(ba)

        return np.mean(ba_list)