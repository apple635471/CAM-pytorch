# simple implementation of CAM in PyTorch 
# only support one by one image verify

import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import cv2
import pdb
import torch
import os


# feature extractor hook (forward hook)
def feature_extractor(module, input, output):
    features.append(output.data.cpu().numpy())


# generate CAM(class activation mapping) result for each prediction  
def CAM_generator(feature, weight_fc, class_idx):
    b, c, h, w = feature.shape
    output_cam = []
    for idx in class_idx:
        # weight dot feature -> [512] dot [512, 7*7] = [49]
        cam = weight_fc[idx].dot(feature.reshape((c, h*w)))
        print('cam size:',cam.size)
        cam = cam.reshape(h, w)
        # normalize
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # [0, 1] map to [0, 255]
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
    return output_cam


# load pytorch model
net = torch.load('resnet18.pt')
net = net.to('cuda')
net.eval()


# get the fc weight
#   name_params[what params?][name or data]
#   ex: name_params[-2][0] is name of fc.weight
#       name_params[-2][1] is data of fc.weight
#   or you can use net.parameters() // just data
# size of fc.weight is [3, 512]
name_params = list(net.named_parameters())
weight_fc = np.squeeze(name_params[-2][1].data.cpu().numpy())


# define preprocessing work
preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   transforms.Normalize((0.57420, 0.46984, 0.35669), (0.27862, 0.28578, 0.29748))
])


# hook the feature extractor
#   extract the output feature tensor of resnet18-layer4
#   shape of resnet18-layer4 output - (1, 512, 7, 7)
target_layer = 'layer4'
net._modules.get(target_layer).register_forward_hook(feature_extractor)


path = './test/'
target_path = './cam/'
subdirs = ['Code_0001', 'Code_0002', 'Code_0003']
classes = {0:'Code_0001', 1:'Code_0002', 2:'Code_0003'}
class_table = list(list(0 for j in range(3)) for i in range(3))


for subdir in subdirs:
    folder = os.path.join(path, subdir)
    target_folder = os.path.join(target_path, subdir)
    for filename in os.listdir(folder):


        # read image & do preprocessing
        img = Image.open(os.path.join(folder, filename))
        img = img.convert('RGB')
        inputs = preprocess(img)
        inputs = inputs.to('cuda').unsqueeze(0)


        # forward
        features = []
        preds = net(inputs)


        # predict 
        softmax = nn.Softmax(dim=1)
        score = softmax(preds).data.squeeze()
        probs, rank = score.sort(0, True)
        probs = probs.cpu().numpy()
        rank = rank.cpu().numpy()

        
        # analyze the prediction result
        ## build confuse table -- class_table[truth][pred]
        if subdir == 'Code_0001':
            class_table[0][rank[0]] += 1
        elif subdir == 'Code_0002':
            class_table[1][rank[0]] += 1
        elif subdir == 'Code_0003':
            class_table[2][rank[0]] += 1    


        # output the prediction
        print('--------------------------------------------------------')
        print()
        print('The correct class is {}'.format(subdir))
        print('The predict result is:')
        for i in range(0, 3):
            print('    {:.3f} -> {}'.format(probs[i], classes[rank[i]]))


        # generate CAM(class activation mapping) result for the top1 prediction
        CAMs = CAM_generator(features[0], weight_fc, [rank[0]])


        # -------------------render the CAM and output------------------------


        print('save CAM images for the top1 prediction: %s...'%classes[rank[0]]) 
        img = cv2.imread(os.path.join(folder, filename))
        h, w, _ = img.shape


        # build a pseudocolor heatmap
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(int( 224 * ( h / 256.0 ) ), int( 224 * ( w / 256.0 ) ))), cv2.COLORMAP_JET)


        # because we have do some preprocess before feeding image in model, like: resize, centercrop
        # in this case, we suppose doing resize(256) & centercrop(224)
        # so, the heatmap only cover the center region of image
        # the size of heatmap is 224 * (original / 256)
        # the location of heatmap is central (position by offset)  
        zeromap = np.zeros((h, w, 3))
        offset_rh = int( ( h - int( 224 * ( h / 256.0 ) ) ) / 2 )
        offset_rw = int( ( w - int( 224 * ( w / 256.0 ) ) ) / 2 )
        offset_lh = int( ( h - offset_rh - int( 224 * ( h / 256.0 ) ) ) * ( -1 ) )
        offset_lw = int( ( w - offset_rw - int( 224 * ( w / 256.0 ) ) ) * ( -1 ) )
        zeromap[offset_rh:offset_lh,offset_rw:offset_lw,:] = heatmap
        heatmap = zeromap
        result = heatmap * 0.3 + img * 0.5


        # put some text description about classification result on image
        text1 = 'Correct class is {}'.format(subdir)
        text2 = 'predict class is {}'.format(classes[rank[0]])
        cv2.putText(result, text1, (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if subdir == classes[rank[0]]: cv2.putText(result, text2, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else: cv2.putText(result, text2, (10, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        # save image in target folder
        cv2.imwrite(os.path.join(target_folder, filename), result)
        print('complete saving {} !'.format(filename))
        print()
        break
    break
print('--------------------------------------------------------')
print()


# print the overall accuracy
all_total = 0
all_correct = 0
for truth_cls in classes:
    all_correct += class_table[truth_cls][truth_cls]
    for pred_cls in classes:
        all_total += class_table[truth_cls][pred_cls]
print('Overall accuracy is %2d %%' % (100 * all_correct / all_total))


# print the accuracy for each class
for truth_cls in classes:
    each_total = 0
    for pred_cls in classes:
        each_total += class_table[truth_cls][pred_cls]
    each_correct = class_table[truth_cls][truth_cls]
    print('Accuracy of %7s : %2d %%' % (classes[truth_cls], 100 * each_correct / each_total))


# print confuse matrix
print()
print('                 Actual class')
print('         {} | {} | {}'.format(classes[0], classes[1], classes[2]))
for pred_cls in classes:
    print('{}{:^9d} | {:^9d} | {:^9d}'.format(classes[pred_cls], class_table[0][pred_cls]
                                      ,class_table[1][pred_cls], class_table[2][pred_cls]))
    



 


