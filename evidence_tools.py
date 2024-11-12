import torch

#evidence (n, num_classes)
def evidence2opinion(evidence):
    num_classes = evidence.shape[1]
    s = torch.sum(evidence+1, dim=1, keepdim=True)
    b = evidence/s
    u = num_classes/s
    return b, u

def combin_two(b1, u1, b2, u2):
    bb = torch.bmm(b1.unsqueeze(-1), b2.unsqueeze(1))
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=1, dim2=2).sum(-1)
    c = bb_sum - bb_diag
    b = (b1*b2 + b1*u2 + b2*u1) / (1-c).unsqueeze(1)
    u = (u1*u2) / (1-c).unsqueeze(1)
    return b, u

def opinion2evidence(b, u):
    num_classes = b.shape[1]
    s = num_classes / u
    evidence = b*s        
    return evidence