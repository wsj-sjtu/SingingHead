import os
import torch
from .metrics import *
from metric_3d.FLAME.decalib.utils.config import cfg as deca_cfg
from metric_3d.FLAME.decalib.models.FLAME import FLAME
from metric_3d.metrics import MetricCalculator


metric = MetricCalculator('./metric_3d/')
flame = FLAME(deca_cfg.model).to('cuda:0')


def calculate_metrics(audio, shape, motion_gt, motion):
    """
    audio: (bs, seq_len)
    shape: (bs, 100)
    motion_gt: (bs, nframe, 59)
    motion: generated motion (bs, nsample, nframe, 59)
    """

    nsample = motion.shape[1]

    # --- for flame params ---
    mindist = metric.compute_mindst(motion, motion_gt)
    meandist = metric.compute_meandist(motion, motion_gt)
    diversity = metric.compute_diversity(motion)
    diversity_pose = metric.compute_diversity(motion[...,50:56])
    mindist_pose = metric.compute_mindst_pose(motion, motion_gt)
    meandist_pose = metric.compute_meandist_pose(motion, motion_gt)

    # --- for mesh ---
    bs, seq_len = motion.shape[0], motion.shape[2]
    # pred
    motion_tmp = motion.contiguous().view(bs*nsample*seq_len, -1)
    full_poses = torch.cat([torch.zeros_like(motion_tmp[:, 50:56]), motion_tmp[:, 56:59], torch.zeros_like(motion_tmp[..., :6])], dim=1).cuda()
    exps = motion_tmp[:, :50]
    shape_rep = shape.unsqueeze(1).unsqueeze(2).repeat(1, nsample, seq_len, 1)
    shape_rep = shape_rep.contiguous().view(bs*nsample*seq_len, -1)
    verts_p, _, _ = flame(shape_params=shape_rep.cuda(),
                                expression_params=exps.cuda(),
                                full_pose=full_poses.cuda())
    pred_verts = verts_p.contiguous().view(bs, nsample, seq_len, 5023*3) 
    # gt
    motion_tmp = motion_gt.contiguous().view(bs*seq_len, -1)
    full_poses = torch.cat([torch.zeros_like(motion_tmp[:, 50:56]), motion_tmp[:, 56:59], torch.zeros_like(motion_tmp[..., :6])], dim=1).cuda()
    exps = motion_tmp[:, :50]
    shape_rep = shape.unsqueeze(1).repeat(1, seq_len, 1)
    shape_rep = shape_rep.contiguous().view(bs*seq_len, -1)
    verts_p, _, _ = flame(shape_params=shape_rep.cuda(),
                                expression_params=exps.cuda(),
                                full_pose=full_poses.cuda())
    gt_verts = verts_p.contiguous().view(bs, seq_len, 5023*3) 

    # metrics
    lve_mean = metric.calculate_lve(pred_verts, gt_verts, type='mean')
    lve_min = metric.calculate_lve(pred_verts, gt_verts, type='min')
    ve_mean = metric.calculate_ve(pred_verts, gt_verts, type='mean')
    ve_min = metric.calculate_ve(pred_verts, gt_verts, type='min')
    fdd = metric.calculate_fdd(pred_verts, gt_verts)

    # BA
    motion_tmp = motion.contiguous().view(bs*nsample*seq_len, -1)
    full_poses = torch.cat([motion_tmp[:, 50:59], torch.zeros_like(motion_tmp[..., :6])], dim=1).cuda()
    exps = motion_tmp[:, :50]
    shape_rep = shape.unsqueeze(1).unsqueeze(2).repeat(1, nsample, seq_len, 1)
    shape_rep = shape_rep.contiguous().view(bs*nsample*seq_len, -1)
    verts_p, _, pred_lmk3d = flame(shape_params=shape_rep.cuda(),
                                expression_params=exps.cuda(),
                                full_pose=full_poses.cuda())
    pred_verts = verts_p.contiguous().view(bs, nsample, seq_len, 5023*3)
    pred_lmk3d = pred_lmk3d.contiguous().view(bs, nsample, seq_len, -1, 3)   # (bs,nsample,nframe,68,3)

    ba_lmk = metric.calculate_ba_lmk(pred_lmk3d, audio)
    ba_pose = metric.calculate_ba_pose(motion[..., 50:56], audio)

    # diversity for mesh
    pred_verts_face = pred_verts.contiguous().view(bs, nsample, seq_len, 5023, 3)[:, :, :, metric.flame_mask['face'], :].view(bs, nsample, seq_len, -1)
    diversity_face = metric.compute_diversity(pred_verts_face)
    diversity_verts = metric.compute_diversity(pred_verts)
    diversity_lmk = metric.compute_diversity(pred_lmk3d.contiguous().view(bs, nsample, seq_len, -1))

    metric_dict = {
        'lve_mean': lve_mean,
        'lve_min': lve_min,
        've_mean': ve_mean,
        've_min': ve_min,
        'fdd': fdd,
        'meandist_pose': meandist_pose,
        'ba_pose': ba_pose,
        'diversity_lmk': diversity_lmk
    }
    return metric_dict