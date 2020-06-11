import torch
import time
from model.model import parsingNet
# torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = True
net = parsingNet(pretrained = False, backbone='18',cls_dim = (100+1,56,4),use_aux=False).cuda()
# net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,18,4),use_aux=False).cuda()

net.eval()

x = torch.zeros((1,3,288,800)).cuda() + 1
for i in range(10):
    y = net(x)
t_all = 0
for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all += t2 - t1

print('avg_time:',t_all / 100)
print('avg_fps:',100 / t_all)