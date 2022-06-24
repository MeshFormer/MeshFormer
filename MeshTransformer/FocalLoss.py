from torch import nn
import torch
from torch.nn import functional as F

class focal_loss(nn.Module):
    def __init__(self, alpha, num_classes, gamma=2, size_average=False):
        super(focal_loss,self).__init__()
        self.size_average = size_average

        assert len(alpha)==num_classes   
        self.alpha = torch.Tensor(alpha)

        self.gamma = gamma

    def forward(self, preds, labels, weight=None):
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) 
        preds_softmax = torch.exp(preds_logsoft)    

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if weight is not None:
            loss = torch.mul(weight, loss.t())


        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss