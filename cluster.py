import faiss
import numpy as np
import torch

def get_faiss_module(in_dim=32):
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False 
    cfg.device     = 0 #NOTE: Single GPU only. 
    #idx = faiss.GpuIndexFlatIP(res, in_dim, cfg)
    idx = faiss.GpuIndexFlatL2(res, in_dim, cfg)
    return idx

def get_init_centroids(in_dim, K, featlist, index):
    index.reset()
    featlist = featlist.t().cpu().numpy()
    clus = faiss.Clustering(in_dim, K)
    clus.seed  = np.random.randint(100)
    clus.niter = 30
    clus.max_points_per_centroid = 10000000
    clus.train(featlist, index)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(K, in_dim).astype('float32')
    centroids = torch.tensor(centroids, requires_grad=False).cuda().t()
    return centroids