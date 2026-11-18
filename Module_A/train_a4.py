"""A4: A3 model plus original asymmetric risk-weighted SmoothL1."""
from torch.nn import functional as F
from train import run
def risk_loss(pred,target):
    base=F.smooth_l1_loss(pred*100,target*100,reduction='none')
    extra=((target<=-.05)&(pred.detach()>target)).to(base.dtype)
    return (base*(1+extra)).mean()
if __name__=='__main__':run('model_a4','A4 风险加权',.001,loss_fn=risk_loss,order='buffered')
