import torch.nn as nn


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean') 
    def forward(self, im_1, im_2):

        grad1_2 = im_1[..., 1:] - im_1[..., 0:-1]
        grad1_1 = im_1[..., 1:, :] - im_1[..., 0:-1, :]
        grad1_0 = im_1[..., 1:, :, :] - im_1[..., 0:-1, :, :] 
        grad2_2 = im_2[..., 1:] - im_2[..., 0:-1]
        grad2_1 = im_2[..., 1:, :] - im_2[..., 0:-1, :]
        grad2_0 = im_2[..., 1:, :, :] - im_2[..., 0:-1, :, :] 
        loss = (self.loss(grad1_2, grad2_2) + self.loss(grad1_1, grad2_1) +self.loss(grad1_0, grad2_0)) / 3.
        return loss

