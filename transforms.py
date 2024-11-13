import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchio
from torchio.transforms import RandomBiasField, RandomBlur, RandomGamma
from torchvision import transforms
import random
import torchvision.transforms.functional as TF

def affine_point(pos, affine):
    pos = (pos - 256)/256
    pos_pad = np.ones((pos.shape[0],1))
    affine_pad = np.array([0., 0., 1.])[None,:]
    pos = np.concatenate([pos,pos_pad],1)
    affine =  np.concatenate ([affine,affine_pad],0)
    pos =  (affine @ pos.T).T
    pos = pos * 256 + 256
    return pos[:,:2]

def random_affine(img, min_rot=-10, max_rot=10, min_shear=-2,
                  max_shear=2, min_scale=0.8, max_scale=1.1):
   
    a = np.radians(np.random.rand() * (max_rot - min_rot) + min_rot)
    shear = np.radians(np.random.rand() * (max_shear - min_shear) + min_shear)
    scale = np.random.rand() * (max_scale - min_scale) + min_scale

    affine1_to_2 = np.array([[np.cos(a) * scale, - np.sin(a + shear) * scale, 0.],
                            [np.sin(a) * scale, np.cos(a + shear) * scale, 0.],
                            [0., 0., 1.]], dtype=np.float32)  # 3x3

    affine2_to_1 = np.linalg.inv(affine1_to_2).astype(np.float32)

    affine1_to_2, affine2_to_1 = affine1_to_2[:2, :], affine2_to_1[:2, :]  # 2x3
    affine1_to_2, affine2_to_1 = torch.from_numpy(affine1_to_2).cuda().unsqueeze(dim=0), \
                                torch.from_numpy(affine2_to_1).cuda().unsqueeze(dim=0)

    img = perform_affine_tf(img, affine1_to_2)
    return img, affine1_to_2, affine2_to_1


def perform_affine_tf(data, tf_matrices):
    if (len(data.shape) == 3):
            data = data.unsqueeze(0)
    if (len(tf_matrices.shape) == 2):
            tf_matrices = tf_matrices.unsqueeze(0)
    # expects 4D tensor, we preserve gradients if there are any
    n_i, k, h, w = data.shape
    n_i2, r, c = tf_matrices.shape
    assert (n_i == n_i2)
    assert (r == 2 and c == 3)
    grid = F.affine_grid(tf_matrices, data.shape, align_corners=True)  # output should be same size
    data_tf = F.grid_sample(data, grid, padding_mode="zeros",align_corners=True)  # this can ONLY do bilinear
    return data_tf


class Geometric_transform(object):
    def __init__(self, min_rot=-10, max_rot=10, min_shear=-1,
                  max_shear=1, min_scale=1.1, max_scale=1.2, 
                  mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.min_shear = min_shear
        self.max_shear = max_shear
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        img = img.float().cuda()
        img, affine1_to_2, affine2_to_1 = random_affine(img, min_rot=self.min_rot, max_rot=self.max_rot, 
                                                        min_shear=self.min_shear,max_shear=self.max_shear, 
                                                        min_scale=self.min_scale, max_scale=self.max_scale)
        img = self.normalize(img.squeeze())
        return img, affine1_to_2, affine2_to_1

    def restore(self, img, affine):
        if (len(img.shape) == 3):
            img = img.unsqueeze(0)
        return perform_affine_tf(img, affine).squeeze()

class Photometric3d_transform(object):
    def __init__(self, coefficients=0.5, std=(0,2), log_gamma=(-0.3, 0.3)):
        self.bias_field = RandomBiasField(coefficients)
        self.blur = RandomBlur(std)
        self.gamma = RandomGamma(log_gamma)

    def __call__(self, img):
        # if (len(img.shape) == 3):
        #     img = img.unsqueeze(0)
        img = self.bias_field(img)
        img = self.blur(img)
        img = self.gamma(img)
        return img
    
class Photometric_transform(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.jitter = transforms.ColorJitter(brightness=brightness,contrast=contrast, saturation=saturation, hue=hue)
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        img = img.float().cuda()
        img = self.jitter(img)
        img = self.normalize(img)
        return img

class Tensor_transform(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.normalize = transforms.Normalize(mean=mean, std=std)
    
    def __call__(self, image):
        image = image.float().cuda()
        image = self.normalize(image)
        return image.cuda()
    

    
class Tensor2d_raw(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = torch.as_tensor(mean).reshape(-1, 1, 1).cuda()
        self.std = torch.as_tensor(std).reshape(-1, 1, 1).cuda()
    def __call__(self, image):
        raw = (image*self.std+self.mean)*255
        raw = raw.permute(1,2,0).cpu().numpy()
        return raw


from scipy.ndimage import binary_erosion
def shrink_border(mask, num_pixels=1):
    """
    Shrink the border of a segmentation mask by a specified number of pixels.

    Parameters:
        mask (numpy.ndarray): The input boolean mask.
        num_pixels (int): The number of pixels to shrink the border. Default is 1.

    Returns:
        numpy.ndarray: The mask with the border shrunk.
    """
    # Perform binary erosion on the mask
    eroded_mask = binary_erosion(mask, iterations=num_pixels)
    # Remove the eroded mask from the original mask to get the shrunk border
    shrunk_border = mask & eroded_mask

    return shrunk_border