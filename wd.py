import torch
import torch.nn.functional as F


def wd(x, y, tau):
    cos_distance = cost_matrix_batch(torch.transpose(x, 2, 1), torch.transpose(y, 2, 1), tau)
    cos_distance = cos_distance.transpose(1, 2)

    beta = 0.1
    min_score = cos_distance.min()
    max_score = cos_distance.max()
    threshold = min_score + beta * (max_score - min_score)
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)

    wd = OT_distance_batch(cos_dist, x.size(0), x.size(1), y.size(1), 40)
    return wd


def OT_distance_batch(C, bs, n, m, iteration=50):
    C = C.float().cuda()
    T = OT_batch(C, bs, n, m, iteration=iteration)
    temp = torch.bmm(torch.transpose(C, 1, 2), T)
    distance = batch_trace(temp, m, bs)
    return distance


def OT_batch(C, bs, n, m, beta=0.5, iteration=50):
    sigma = torch.ones(bs, int(m), 1).cuda() / float(m)
    T = torch.ones(bs, n, m).cuda()
    A = torch.exp(-C / beta).float().cuda()
    for t in range(iteration):
        Q = A * T
        for k in range(1):
            delta = 1 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q, 1, 2), delta)
            sigma = 1 / (float(m) * a)
        T = delta * Q * sigma.transpose(2, 1)
    return T


def cost_matrix_batch(x, y, tau=0.5):
    bs = list(x.size())[0]
    D = x.size(1)
    assert (x.size(1) == y.size(1))
    x = x.contiguous().view(bs, D, -1)
    x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
    y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)

    cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)
    cos_dis = torch.exp(- cos_dis / tau)
    return cos_dis.transpose(2, 1)


def batch_trace(input_matrix, n, bs):
    a = torch.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
    b = a * input_matrix
    return torch.sum(torch.sum(b, -1), -1).unsqueeze(1)