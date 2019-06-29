from __future__ import print_function 
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import WIDERFace_ROOT , WIDERFace_CLASSES as labelmap
from data import WIDERFaceDetection, WIDERFaceAnnotationTransform, WIDERFace_CLASSES, WIDERFace_ROOT, BaseTransform , TestBaseTransform
from data import *
from PIL import Image

import torch.utils.data as data

from face_ssd import build_ssd
#from resnet50_ssd import build_sfd
import pdb
import pandas as pd
import numpy as np

from FaceAttr_baseline_model import FaceAttrModel

import cv2
import math
import matplotlib.pyplot as plt
import time

plt.switch_backend('agg')

"""
parser = argparse.ArgumentParser(description='DSFD:Dual Shot Face Detector')
parser.add_argument('--trained_model', default='/data2/faceAttr/FaceAttr-Analysis-App/WIDERFace_DSFD_RES152.pth', # 人脸识别预训练模型
                    type=str, help='Trained state_dict file path to open') 
parser.add_argument('--save_folder', default='save_folder/', type=str,
                    help='Dir to save results')  # 图片保存的文件夹
parser.add_argument('--visual_threshold', default=0.1, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()
"""

trained_model = "/data2/faceAttr/FaceAttr-Analysis-App/WIDERFace_DSFD_RES152.pth"
save_folder = "save_folder/"
visual_threshold = 0.3
cuda = True

CUDA_DEVICE_1 = 0  # 运行人脸检测模型  并行1
CUDA_DEVICE_2 = 4  # 运行人脸检测模型  并行2
CUDA_DEVICE_3 = 7  # 运行人脸属性分类模型
# 人脸属性分类的路径
faceAttr_path = "/data2/faceAttr/FaceAttr-Analysis-App/se_resnet101.pth"

# 预加载人脸检测模型
cfg = widerface_640
num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD
net = nn.DataParallel(net, device_ids=[CUDA_DEVICE_1, CUDA_DEVICE_2])
state_dict = torch.load(trained_model)
from collections import OrderedDict
new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if 'module' not in k:
        k = 'module.'+k
    else:
        k = k.replace('features.module.', 'module.features.')
    new_state_dict[k]=v

net.load_state_dict(new_state_dict)
# net = net.cuda(CUDA_DEVICE_1)

# 预加载属性分类模型
all_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
            'Bangs', 'Big_Lips', 'Big_Nose','Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
            'Wearing_Hat','Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young' ]

attr_nums = [i for i in range(len(all_attrs))] 
selected_attrs = []
for num in attr_nums:
    selected_attrs.append(all_attrs[num])
attr_threshold = [0.5 for i in range(len(all_attrs))]  
pretrained = False
model = FaceAttrModel("se_resnet101", pretrained, selected_attrs)
model = model.cuda(CUDA_DEVICE_3)
model.load_state_dict(torch.load(faceAttr_path))

print("loaded the two models...")

#os.environ["CUDA_VISIBLE_DEVICES"] = "3"  
if cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(CUDA_DEVICE_1)
    print("set default cuda float tensor")
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    print("set default float tensor")
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# 对检测框进行投票
def bbox_vote(det):
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)
        # get needed merge det and delete these det
        merge_index = np.where(o >= 0.3)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)
        if merge_index.shape[0] <= 1:
            continue
        det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
        max_score = np.max(det_accu[:, 4])
        det_accu_sum = np.zeros((1, 5))
        
        for i in range(det_accu.shape[0]):
            for j in range(det_accu.shape[1]):
                if isinstance(det_accu[i, j], torch.Tensor):
                    det_accu[i, j] = float(det_accu[i, j])

        det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
        if isinstance(max_score, float):
            det_accu_sum[:, 4] = max_score
        else:
            det_accu_sum[:, 4] = max_score.cpu()
        try:
            dets = np.row_stack((dets, det_accu_sum))
        except:
            dets = det_accu_sum
    dets = dets[0:750, :]
    return dets

def infer(net , img , transform , thresh , cuda , shrink):
    if shrink != 1:
        img = cv2.resize(img, None, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)

    x = Variable(x.unsqueeze(0))
    
    x = x.cuda(CUDA_DEVICE_1)
    #print (shrink , x.shape)
    y = None
    x = x.type(torch.cuda.FloatTensor).cuda(CUDA_DEVICE_1)
    # print(x.device)
    # net = net.cuda(CUDA_DEVICE_1)
    # print(net)
    net = net.cuda()
    with torch.no_grad():
        y = net(x)
    # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([ img.shape[1]/shrink, img.shape[0]/shrink,
                        img.shape[1]/shrink, img.shape[0]/shrink] )
    det = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            #label_name = labelmap[i-1]
            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1], pt[2], pt[3]) 
            det.append([pt[0], pt[1], pt[2], pt[3], score])
            j += 1
    if (len(det)) == 0:
        det = [ [0.1,0.1,0.2,0.2,0.01] ]
    det = np.array(det)

    keep_index = np.where(det[:, 4] >= 0)[0]
    det = det[keep_index, :]
    return det

def infer_flip(net , img , transform , thresh , cuda , shrink):
    img = cv2.flip(img, 1)
    det = infer(net , img , transform , thresh , cuda , shrink)
    det_t = np.zeros(det.shape)
    det_t[:, 0] = img.shape[1] - det[:, 2]
    det_t[:, 1] = det[:, 1]
    det_t[:, 2] = img.shape[1] - det[:, 0]
    det_t[:, 3] = det[:, 3]
    det_t[:, 4] = det[:, 4]
    return det_t


def infer_multi_scale_sfd(net , img , transform , thresh , cuda ,  max_im_shrink):
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net , img , transform , thresh , cuda , st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    bt = min(2, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net , img , transform , thresh , cuda , bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > 2:
        bt *= 2
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , bt)))
            bt *= 2
        det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , max_im_shrink) ))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    return det_s, det_b


def vis_detections(im,  dets, image_name, picture_name, thresh=0.5, output_path=None):
    """
    im: 即将可视化的图片
    dets:
    image_name: 即将保存的图片名
    thresh: 检测的阈值
    picture_name: 将要进行标注的图片
    output_path:  图片输出的路径
    """

    """Draw detected bounding boxes."""

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return "no face in image"
    #print (len(inds))
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    attr_dict = {}

    for i in inds:

        img=Image.open(picture_name)
        bbox = dets[i, :4]
        # score = dets[i, -1]
        bbox0=bbox[0]
        bbox1=bbox[1]
        bbox2=bbox[2]
        bbox3=bbox[3]
        if bbox1-(bbox3 - bbox1)*0.5>=0:
            bbox[1]=bbox1-(bbox3 - bbox1)*0.5
        else:
            bbox[1]=0   
        if bbox3+(bbox3 - bbox1)*0.15<=img.size[1]:
            bbox[3]=bbox3+(bbox3 - bbox1)*0.15
        else:
            bbox[3]=img.size[1]        
        if bbox2+(bbox2 - bbox0)*0.25<=img.size[0]:
            bbox[2]=bbox2+(bbox2 - bbox0)*0.3
        else:
            bbox[2]=img.size[0] 
        if bbox0-(bbox2 - bbox0)*0.25>=0:
            bbox[0]=bbox0-(bbox2 - bbox0)*0.3
        else:
            bbox[0]=0        

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                            bbox[2] - bbox[0],
                            bbox[3] - bbox[1], fill=False,
                            edgecolor='red', linewidth=2.5)
                )
        torch.set_default_tensor_type('torch.FloatTensor')
        loader = transforms.Compose([transforms.Resize(size=(224, 224)), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])        

        box=(bbox[0],bbox[1],bbox[2],bbox[3])
        roi=(img.crop(box))
        roi_t = loader(roi).unsqueeze(0)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        global model

        model.eval()
        outputs = None
        roi_t = roi_t.type(torch.cuda.FloatTensor).cuda(CUDA_DEVICE_3)

        with torch.no_grad():
            outputs = model(roi_t)
            
        if len(inds) == 1:
            global all_attrs
            for i in range(len(all_attrs)):
                attr_dict[all_attrs[i]] = outputs[0].data[i].cpu().item()
            print("Only one face in image")
            break
        
        else:
            if_Bald=outputs[0].data[4]
            if_Bald = 1 if if_Bald > attr_threshold[4]-0.15 else 0
                
            if_Eyeglasses=outputs[0].data[15]
            if_Eyeglasses = 1 if if_Eyeglasses > attr_threshold[15] else 0

            if_Male=outputs[0].data[20]
            if_Male= 1 if if_Male > attr_threshold[15]+0.4 else 0

            if_Wearing_Hat=outputs[0].data[35]
            if_Wearing_Hat = 1 if if_Wearing_Hat > attr_threshold[35] else 0


            print('Bald:{:.1f} Eyeglasses:{:.2f} Male:{:.2f} Wearing_Hat:{:.2f}'.format(if_Bald, if_Eyeglasses,if_Male,if_Wearing_Hat))
            ax.text(bbox[0], bbox[1] - 5,
                        'Bald:{:.1f} Eyeglasses:{:.1f} \n Male:{:.1f} Wearing_Hat:{:.1f}'.format(if_Bald, if_Eyeglasses,if_Male,if_Wearing_Hat),
                        bbox=dict(facecolor='white', alpha=0.5),
                        fontsize=10, color='black')
            
    plt.axis('off')
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path,dpi=fig.dpi)
    else:
        plt.savefig(save_folder+image_name, dpi=fig.dpi)
    
    if len(inds) == 1:
        print("only one face")
        print(attr_dict)
        return attr_dict
    elif len(inds) > 1:
        if output_path == None:
            return save_folder+image_name
        else:
            print("output_path {}".format(output_path))
            return output_path

def detect_image(path, output_path=None):
    """
        @params:
            path: 要识别分类的目标图片
        @return:
            result_image: 识别结果图片
    """
    # load net
    global net
    net.eval()

    # evaluation
    cuda = True
    transform = TestBaseTransform((104, 117, 123))
    thresh=cfg['conf_thresh']

    img_id = 'face'
    start = time.time()
    print("Start time: {}".format(start))
    #print(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #print(img)
    max_im_shrink = ( (2000.0*2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
    shrink = max_im_shrink if max_im_shrink < 1 else 1

    det0 = infer(net , img , transform , thresh , cuda , shrink)
    det1 = infer_flip(net , img , transform , thresh , cuda , shrink)
    # shrink detecting and shrink only detect big face
    st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
    det_s = infer(net , img , transform , thresh , cuda , st)
    index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
    det_s = det_s[index, :]
    # enlarge one times
    factor = 2
    bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
    det_b = infer(net , img , transform , thresh , cuda , bt)
    # enlarge small iamge x times for small face
    if max_im_shrink > factor:
        bt *= factor
        while bt < max_im_shrink:
            det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , bt)))
            bt *= factor
        det_b = np.row_stack((det_b, infer(net , img , transform , thresh , cuda , max_im_shrink) ))
    # enlarge only detect small face
    if bt > 1:
        index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
        det_b = det_b[index, :]
    else:
        index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
        det_b = det_b[index, :]
    det = np.row_stack((det0, det1, det_s, det_b))
    det = bbox_vote(det)
    mid = time.time()
    result = vis_detections(img , det , img_id, path, visual_threshold, output_path)
    end = time.time()
    print("end time: {}".format(end))
    print("classifying time: {:.2f} s".format(end-mid))
    print("total time : {:.2f} s".format(end-start))
    return result


if __name__ == '__main__':
    output_path = "output.jpg"
    detect_image("test1.jpg", "output1.jpg")
    detect_image("test2.jpg", "output2.jpg")
    detect_image("test3.jpg", "output3.jpg")
