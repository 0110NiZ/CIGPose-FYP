"""A2 uses A1 model plus its original top-1 ranking loss."""
from torch.nn import functional as F
from train import run,smooth_l1
def ranking_loss(pred,target):
    p=pred*100;t=target*100;j=t.argmax(1,keepdim=True)
    mask=(t.gather(1,j)-t>.05).to(p.dtype)
    ranking=(F.softplus(-(p.gather(1,j)-p))*mask).sum()/mask.sum().clamp(min=1.)
    return smooth_l1(pred,target)+.2*ranking
if __name__=='__main__':run('model_a2','A2 排序监督',.001,loss_fn=ranking_loss,order='buffered')
