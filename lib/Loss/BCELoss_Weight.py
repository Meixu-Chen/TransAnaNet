# import torch
# from torch.nn.modules.loss import _Loss
#
#
# # class BCELoss_ClassWeight(_Loss):
# class BCELossClassWeight(_Loss):
#     def __init__(self, args, class_weights=[1, 1e4]):
#         super(BCELossClassWeight, self).__init__()
#         self.class_weights = class_weights
#         self.args = args
#
#     def forward(self, input: torch.Tensor, target: torch.Tensor):
#
#         if not torch.is_tensor(self.class_weights):
#             self.class_weights = torch.Tensor(self.class_weights)
#
#         else:
#             pass
#
#         input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
#         bce = self.class_weights[0]*target * torch.log(input) - self.class_weights[1]*(1 - target) * torch.log(1 - input)
#         weighted_bce = (bce * self.class_weights).sum() / self.class_weights.sum()[0]
#         print('bce weight ', weighted_bce.size())
#         final_reduced_over_batch = weighted_bce.mean(axis=0)
#         print('final: ', final_reduced_over_batch.size())
#
#         # input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
#         # bce = - target * torch.log(input) - (1 - target) * torch.log(1 - input)
#         # print('bce: ', bce.size())
#         # weighted_bce = (bce * self.class_weights).sum(axis=1) / self.class_weights.sum(axis=1)[0]
#         # print('bce weight ', weighted_bce.size())
#         # final_reduced_over_batch = weighted_bce.mean(axis=0)
#         # print('final: ', final_reduced_over_batch.size())
#
#         return final_reduced_over_batch
