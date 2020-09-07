# Thanks for the contribution of KopiSoftware https://github.com/KopiSoftware

import torch
import time
import numpy as np
from model.model import parsingNet
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt
from PIL import Image


img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def resize(x, y):
    global cap
    cap.set(3,x)
    cap.set(4,y)

def test_practical_without_readtime():
    global cap
    for i in range(10):
        _,img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1
        y = net(x)
        
    print("pracrical image input size:",img.shape)
    print("pracrical tensor input size:",x.shape)
    t_all = []
    for i in range(100):
        _,img = cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1

        t1 = time.time()
        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
        
    print("practical with out read time:")
    print('\taverage time:', np.mean(t_all) / 1)
    print('\taverage fps:',1 / np.mean(t_all))
    
    # print('fastest time:', min(t_all) / 1)
    # print('fastest fps:',1 / min(t_all))
    
    # print('slowest time:', max(t_all) / 1)
    # print('slowest fps:',1 / max(t_all))
 
    
def test_practical():
    global cap
    for i in range(10):
        _,img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1
        y = net(x)
        
    print("pracrical image input size:",img.shape)
    print("pracrical tensor input size:",x.shape)
    t_all = []
    t_capture = []
    t_preprocessing = []
    t_net = []
    for i in range(100):
        t1 = time.time()
        
        t3 = time.time()
        _,img = cap.read()
        t4 = time.time()
        
        t5 = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = Image.fromarray(img)
        x = img_transforms(img2)
        x = x.unsqueeze(0).cuda()+1
        t6 = time.time()

        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
        t_capture.append(t4 - t3)
        t_preprocessing.append(t6 - t5)
        t_net.append(t2 - t6)
        
    print("practical with read time:")
    print('\taverage time:', np.mean(t_all) / 1)
    print('\taverage fps:',1 / np.mean(t_all))
    print('\tcapture time:', np.mean(t_capture) / 1)
    print('\tpre-processing time:', np.mean(t_preprocessing) / 1)
    print('\tdetect time:', np.mean(t_net) / 1)
    
    # print('fastest time:', min(t_all) / 1)
    # print('fastest fps:',1 / min(t_all))
    
    # print('slowest time:', max(t_all) / 1)
    # print('slowest fps:',1 / max(t_all))
    
###x = torch.zeros((1,3,288,800)).cuda() + 1
def test_theoretical():
    x = torch.zeros((1,3,288,800)).cuda() + 1
    for i in range(10):
        y = net(x)
    
    t_all = []
    for i in range(100):
        t1 = time.time()
        y = net(x)
        t2 = time.time()
        t_all.append(t2 - t1)
    print("theortical")
    print('\taverage time:', np.mean(t_all) / 1)
    print('\taverage fps:',1 / np.mean(t_all))
    
    # print('fastest time:', min(t_all) / 1)
    # print('fastest fps:',1 / min(t_all))
    
    # print('slowest time:', max(t_all) / 1)
    # print('slowest fps:',1 / max(t_all))
    



if __name__ == "__main__":
    ###captrue data from camera or video
    #cap = cv2.VideoCapture("video.mp4") #uncommen to activate a video input
    cap = cv2.VideoCapture(0) #uncommen to activate a camera imput
    #resize(480, 640) #ucommen to change input size
    
    
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    net = parsingNet(pretrained = False, backbone='18',cls_dim = (100+1,56,4),use_aux=False).cuda()
    # net = parsingNet(pretrained = False, backbone='18',cls_dim = (200+1,18,4),use_aux=False).cuda()
    net.eval()
    

    test_practical_without_readtime()
    test_practical()
    cap.release()
    test_theoretical()    