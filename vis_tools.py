import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pdb

def patch_select_point_rl(mask_sim, pathchsize = 4, topk=1):
    w, h = mask_sim.shape
    stride = w//2
    mask_sim1 = mask_sim[:, :stride]
    mask_sim2 = mask_sim[:, stride:]
    point1, plbl1 = patch_select_point(mask_sim1, pathchsize, topk)
    point2, plbl2 = patch_select_point(mask_sim2, pathchsize, topk) 
    # import pdb
    # pdb.set_trace()
    point = np.concatenate((point1, point2+ np.array([stride,0])), axis=0)
    plbl = np.concatenate((plbl1, plbl2), axis=0)
    return point, plbl

def patch_select_point(mask_sim, pathchsize = 4, topk=1):

    H, W = mask_sim.shape
    new_sx = H - H % pathchsize
    num_px = H // pathchsize

    new_sy = W - W % pathchsize
    num_py = W // pathchsize

    mask_sim = mask_sim[:new_sx, :new_sy]
    x = mask_sim.view(num_px, pathchsize, num_py, pathchsize)
    x_sum = x.sum((1,3))
    topk_xy = x_sum.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // num_py).unsqueeze(0)
    topk_y = (topk_xy - topk_x * num_py)

    all_points = []
    for t in range(topk):
        t_x, t_y = topk_x[0, t], topk_y[0, t]
        patch = mask_sim[t_x*pathchsize:t_x*pathchsize+pathchsize, t_y*pathchsize:t_y*pathchsize+pathchsize]
        patch_xy = patch.flatten(0).topk(1)[1]
        patch_x = (patch_xy // pathchsize).unsqueeze(0)
        patch_y = (patch_xy - patch_x * pathchsize)
        patch_x = patch_x+t_x*pathchsize
        patch_y = patch_y+t_y*pathchsize
        patch_xy = torch.cat((patch_y, patch_x), dim=0).permute(1, 0)
        all_points.append(patch_xy)
    all_points = torch.cat(all_points,0).cpu().numpy()
    label = np.array([1] * topk)
    return all_points, label

# def patch_select_point(mask_sim, pathchsize = 4, topk=1):
#     pdb.set_trace()
#     H, W = mask_sim.shape
#     new_s = H - H % pathchsize
#     num_p = H // pathchsize
#     mask_sim = mask_sim[:new_s, :new_s]
#     x = mask_sim.view(num_p, pathchsize, num_p, pathchsize)
#     x_sum = x.sum((1,3))
#     topk_xy = x_sum.flatten(0).topk(topk)[1]
#     topk_x = (topk_xy // num_p).unsqueeze(0)
#     topk_y = (topk_xy - topk_x * num_p)

#     all_points = []
#     for t in range(topk):
#         t_x, t_y = topk_x[0, t], topk_y[0, t]
#         patch = mask_sim[t_x*pathchsize:t_x*pathchsize+pathchsize, t_y*pathchsize:t_y*pathchsize+pathchsize]
#         patch_xy = patch.flatten(0).topk(1)[1]
#         patch_x = (patch_xy // pathchsize).unsqueeze(0)
#         patch_y = (patch_xy - patch_x * pathchsize)
#         patch_x = patch_x+t_x*pathchsize
#         patch_y = patch_y+t_y*pathchsize
#         patch_xy = torch.cat((patch_y, patch_x), dim=0).permute(1, 0)
#         all_points.append(patch_xy)
#     all_points = torch.cat(all_points,0).cpu().numpy()
#     label = np.array([1] * topk)
#     return all_points, label

def random_point_rl(lbl, num_pos, num_neg):
    w, h = lbl.shape
    stride = w//2
    lbl1 = lbl[:, :stride]
    lbl2 = lbl[:, stride:]
    point1, plbl1 = random_point(lbl1, num_pos, num_neg)
    point2, plbl2 = random_point(lbl2, num_pos, num_neg) 
    # import pdb
    # pdb.set_trace()
    point = np.concatenate((point1, point2+ np.array([stride,0])), axis=0)
    plbl = np.concatenate((plbl1, plbl2), axis=0)
    return point, plbl

def random_point(lbl, num_pos, num_neg):
    w, h = lbl.shape

    pos_point =  np.where(lbl.flatten() > 0)[0]  
    np.random.shuffle(pos_point)
    pos_point = pos_point[:num_pos]

    pos_x = (pos_point // h)[None,:]
    pos_y = (pos_point - pos_x * h)
    pos_point = np.concatenate((pos_y, pos_x), axis=0).T

    neg_point =  np.where(lbl.flatten() < 1)[0]   
    np.random.shuffle(neg_point)
    neg_point = neg_point[:num_neg]

    neg_x = (neg_point // h)[None,:]
    neg_y = (neg_point - neg_x * h)
    neg_point = np.concatenate((neg_y, neg_x), axis=0).T

    point = np.concatenate((pos_point, neg_point), axis=0)
    lbl = np.array([1] * num_pos + [0]*num_neg)
    return point, lbl


# def point_selection_leftright(mask_sim, topk=1):
#     # Top-1 point selection
#     mask_sim = np.

#     w, h = mask_sim.shape
#     topk_xy = mask_sim.flatten(0).topk(topk)[1]
#     topk_x = (topk_xy // h).unsqueeze(0)
#     topk_y = (topk_xy - topk_x * h)
#     topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
#     topk_label = np.array([1] * topk)
#     topk_xy = topk_xy.cpu().numpy()
        
#     # Top-last point selection
#     last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
#     last_x = (last_xy // h).unsqueeze(0)
#     last_y = (last_xy - last_x * h)
#     last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
#     last_label = np.array([0] * topk)
#     last_xy = last_xy.cpu().numpy()
    
#     return topk_xy, topk_label, last_xy, last_label

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   