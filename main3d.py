from segment_anything import sam_model_registry
from segment_anything.medpredictor import SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from transforms import Photometric_transform, Geometric_transform, Tensor_transform, Tensor2d_raw, affine_point, shrink_border
from segment_anything.utils.amg import build_point_grid
from evidence_tools import evidence2opinion, combin_two, opinion2evidence
from vis_tools import point_selection, show_points, show_mask, patch_select_point
import torch.nn.functional as F
import pdb
from cluster import get_faiss_module, get_init_centroids   
import argparse
import os
from tqdm import tqdm
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data/npy')
    parser.add_argument('--outdir', type=str, default='./output')
    parser.add_argument('--sam_type', type=str, default='vit_b')
    parser.add_argument('--dataset', type=str, default='CT_Abd')
    
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--num_grid_point', type=int, default=8)
    parser.add_argument('--num_anchors_pos', type=int, default=5)
    parser.add_argument('--num_anchors_neg', type=int, default=10)
    parser.add_argument('--num_anchors_back', type=int, default=5)
    parser.add_argument('--num_point_pos', type=int, default=1)
    parser.add_argument('--num_point_neg', type=int, default=1)
    parser.add_argument('--num_points_back', type=int, default=5)

    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()
    return args

def read_img(path):
    img = np.load(path)
    img = torch.from_numpy(img).permute(2,0,1)
    return img

def read_lbl(path, ref=False):
    lbl = np.load(path)
    lbl = torch.from_numpy(lbl)
    lbl[lbl>0] = 1
    # if ref:
    #     lbl = lbl*2
    return lbl

@torch.no_grad()
def main():
    args = get_arguments()
    args.device = torch.device("cuda")
    print("Args:", args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    output_path = os.path.join(args.outdir, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    prior_mask = torch.zeros((1, 2, 128, 128)).cuda()
    pp = 30
    prior_mask[:,:,pp:-pp,pp:-pp] = 1.

    # build model
    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'weights/sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=args.device)
    elif args.sam_type == 'vit_b':
        sam_type, sam_ckpt = 'vit_b', 'weights/sam_vit_b_01ec64.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=args.device)
    elif args.sam_type == 'med_sam':
        sam_type, sam_ckpt = 'vit_b', 'weights/medsam_vit_b.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=args.device)



    mask_Predictor = SamPredictor(sam)

    # build transforms
    phot = Photometric_transform()
    geot = Geometric_transform()
    totensor = Tensor_transform()
    getraw = Tensor2d_raw()

    faiss_module_pos = get_faiss_module()
    faiss_module_neg = get_faiss_module()
    faiss_module_back = get_faiss_module()
    
    print('======> Start Testing')
    test_img_folder = os.path.join(args.root, args.dataset, 'imgs')
    test_gt_folder = os.path.join(args.root, args.dataset, 'gts')
    ref_img_folder = os.path.join(args.root, args.dataset, 'refimgs3')
    ref_gt_folder = os.path.join(args.root, args.dataset, 'refgts3')
    ref_length = len(os.listdir(ref_img_folder)) -1 
    DSCs, NSDs = [], []

    weight = torch.ones((2,1,13,13)).cuda() / 169

    for img in tqdm(os.listdir(test_img_folder)):
        #torch.cuda.empty_cache()  # 释放显存
        depth = int(img[-7:-4])
        depth = min(depth, ref_length)
        depth = str(depth).zfill(3)

        tgt_img = read_img(os.path.join(test_img_folder, img))
        tgt_gt = read_lbl(os.path.join(test_gt_folder, img))
        ref_img = read_img(os.path.join(ref_img_folder, f'{depth}.npy'))
        ref_gt = read_lbl(os.path.join(ref_gt_folder, f'{depth}.npy'),True)

        ref_img = totensor(ref_img)
        pho_img = phot(tgt_img)
        geo_img, affine1_to_2, affine2_to_1= geot(tgt_img)
        tgt_img = totensor(tgt_img)

        imageup = torch.cat([tgt_img, ref_img], axis=2)
        imagebutt = torch.cat([pho_img, geo_img], axis=2)
        image = torch.cat([imageup, imagebutt], 1)

        # get grid points
        points = build_point_grid(args.num_grid_point)
        points_scale = np.array((args.img_size*2,args.img_size*2))[None, ::-1]
        points = points*points_scale
        points_lbl = np.ones(points.shape[0])

        # get features
        emb = mask_Predictor.set_image(image)
        masks_np, iou_predictions_np, low_res_masks_np, _ = mask_Predictor.predict(emb, point_coords=points, point_labels=points_lbl)
        feature = mask_Predictor.features
        featuremap = torch.nn.functional.normalize(feature, 2, dim=0) 
        c, h, w = featuremap.shape
        feature_size = h // 2

        # get logit
        anchor_feat = featuremap[:, :feature_size,feature_size:]
        tensor_mask = F.interpolate(ref_gt.unsqueeze(0).unsqueeze(0), size=(feature_size, feature_size), mode = "nearest").squeeze()
        # pos_anchor = anchor_feat[:, tensor_mask==2]
        pos_anchor = anchor_feat[:, tensor_mask==1]
        neg_anchor = anchor_feat[:, tensor_mask==0]

        
        #pdb.set_trace()
        if pos_anchor.shape[0] <= 100:
            pos_anchor = pos_anchor.mean(1, keepdim=True)
            pos_logit = torch.einsum('dn,dhw->hwn', pos_anchor, featuremap)
            pos_logit = pos_logit.squeeze()

        elif pos_anchor.shape[0] <= 200:
            args.num_anchors_pos = 2
            pos_anchor = get_init_centroids(32, args.num_anchors_pos, pos_anchor, faiss_module_pos)
            pos_logit =  torch.einsum('dn,dhw->hwn', pos_anchor, featuremap)
            pos_logit,_ = torch.max(pos_logit,-1)
        else:
            pos_anchor = get_init_centroids(32, args.num_anchors_pos, pos_anchor, faiss_module_pos)
            pos_logit =  torch.einsum('dn,dhw->hwn', pos_anchor, featuremap)
            pos_logit,_ = torch.max(pos_logit,-1)
        #pdb.set_trace()
        neg_anchor = get_init_centroids(32, args.num_anchors_neg, neg_anchor, faiss_module_neg)
        #neg_anchor = neg_anchor.mean(1, keepdim=True)
        neg_logit =  torch.einsum('dn,dhw->hwn', neg_anchor, featuremap)
        neg_logit,_ = torch.max(neg_logit,-1)

        
    
        # back_anchor = get_init_centroids(32, args.num_anchors_back, back_anchor, faiss_module_back)        
        # back_logit =  torch.einsum('dn,dhw->hwn', back_anchor, featuremap)
        # back_logit,_ = torch.max(back_logit,-1)

        # get evidence
        evidence = torch.stack([neg_logit, pos_logit], -1)
        evidence = torch.relu(evidence)

       
        #pdb.set_trace()

        col1, col2 = torch.chunk(evidence, 2, 0)
        e1, _ = torch.chunk(col1, 2, 1)
        e2, e3 = torch.chunk(col2, 2, 1)
        e3 = e3.permute(2,0,1)
        e3 = geot.restore(e3,affine2_to_1) 
        e3 = e3.permute(1,2,0)

        # multi view fusion
        b1, u1 = evidence2opinion(e1.reshape(-1, e1.shape[-1]))
        b2, u2 = evidence2opinion(e2.reshape(-1, e1.shape[-1]))
        b3, u3 = evidence2opinion(e3.reshape(-1, e1.shape[-1]))
        b, u = combin_two(*combin_two(b1,u1, b2,u2), b3,u3) 
        b = b.reshape(feature_size,feature_size, -1)
        
        u = u.reshape(feature_size,feature_size, -1)
        e = opinion2evidence(b, u).permute(2,0,1).unsqueeze(0)

        e = e*prior_mask#.unsqueeze(-1)
        e = F.conv2d(e,weight,None,1,6,groups=2).squeeze().permute(1,2,0)
        b, u = evidence2opinion(e.reshape(-1, e.shape[-1]))
        b = b.reshape(feature_size,feature_size, -1)
        #pdb.set_trace()


        # for jsrt
        topk_xy_pos, _ = patch_select_point(b[:,:,1], args.patch_size, args.num_point_pos)
        topk_xy_neg, _ = patch_select_point(b[:,:,0], args.patch_size, args.num_point_neg)
        # pdb.set_trace()
        # print(topk_xy_neg)

        # topk_xy_bc, topk_label_neg, _, _ = point_selection(b[:,:,0], args.num_points_back)
        #print(topk_xy_pos * 8)
        #pdb.set_trace()
        points_tgt = np.concatenate([topk_xy_pos, topk_xy_neg], 0 )*8
        point_lbl = np.concatenate([np.ones(args.num_point_pos, dtype=int), np.zeros(args.num_point_neg, dtype=int)],0)

        tgt_img_up = F.interpolate(tgt_img.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True).squeeze()
        emb = mask_Predictor.set_image(tgt_img_up)
        # pdb.set_trace()
        masks_np, scores, low_res_masks, high_res_masks = mask_Predictor.predict(emb, point_coords=points_tgt, point_labels=point_lbl)
        
        best_idx = 0
        best_idx = np.argmax(scores)
        if masks_np[best_idx].sum() == 0:
            best_idx = np.argsort(scores)[-2]
        if masks_np[best_idx].sum() == 0:
            input_box = None
        else:
            y, x = np.nonzero(masks_np[best_idx])
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])

            if args.sam_type == 'med_sam':
                masks_np, scores, low_res_masks, high_res_masks = mask_Predictor.predict(
                    emb,
                    #point_coords=points_tgt,
                    #point_labels=point_lbl,
                    box=input_box[None, :],
                    multimask_output=True)
            else:
                masks_np, scores, low_res_masks, high_res_masks = mask_Predictor.predict(
                    emb,
                    point_coords=points_tgt,
                    point_labels=point_lbl,
                    box=input_box[None, :],
                    multimask_output=True)
            
        best_idx = np.argmax(scores)
        masks = F.interpolate(high_res_masks.unsqueeze(0), (512, 512), mode="bilinear", align_corners=False).squeeze()
        masks = (masks>0.0).detach().cpu().numpy()
        masks = masks[best_idx]
        score = scores[best_idx]

        surface_distances = compute_surface_distances(tgt_gt.bool().cpu().numpy(), masks, (1, 1))
        DSC = compute_dice_coefficient(tgt_gt.cpu().numpy(), masks)
        NSD = compute_surface_dice_at_tolerance(surface_distances, 1)
        DSCs.append(DSC)
        NSDs.append(NSD)

        
    DSCs, NSDs = np.array(DSCs).mean(), np.array(NSDs).mean()
    print(f'final results: DSC: {DSCs*100.:.2f}, NSD: {NSDs*100.:.2f}')


if __name__ == "__main__":
    main()

