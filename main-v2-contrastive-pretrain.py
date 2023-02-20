import time, os
import glob
from torch.utils.data import DataLoader
import json
import torch
from os.path import join
from tensorboardX import SummaryWriter
import torch.nn as nn
from utils.dataloader import OAI_pretrain
from utils.parser import parse
from utils.Metric import Metric
import torch.optim as optim
from utils.helpers import Progressbar, add_scalar_dict
from utils.metrics_segmentation import SegmentationCrossEntropyLoss
from utils.losses_pytorch.dice_loss import TverskyLoss, FocalTversky_loss, SoftDiceLoss, AsymLoss
from utils.losses_pytorch.lovasz_loss import LovaszSoftmax
from utils.losses_pytorch.hausdorff import HausdorffDTLoss
import torch.nn.functional as F

class Classifier():
    def __init__(self, args, net):
        self.weights = args.weights
        self.backbone = args.backbone
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.model = self.network_map(net)
        self.model.train()
        self.model.cuda()

        # self.loss = self.pick_loss(args.loss).cuda()
        self.loss = self.pick_loss(args.loss)
        self.optim_model = optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)

    def set_lr(self, lr):
        for g in self.optim_model.param_groups:
            g['lr'] = lr


    def train(self):
        self.model.train()

    def save(self, path):
        states = {
            'model': self.model.state_dict(),
            'optim_model': self.optim_model.state_dict(),
        }
        torch.save(states, path)

    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'model' in states:
            self.model.load_state_dict(states['model'])
        if 'optim_model' in states:
            self.optim_model.load_state_dict(states['optim_model'])

    def network_map(self, net):
        if self.backbone.__contains__('resnext'):
            print('Weights set as instagram.')
            self.weights = 'instagram'
        # from utils.upernet import UperNet
        # model = UperNet(5, 2)
        
        # states = torch.load('experiments/pretrain.pth')
        # model.load_state_dict(states['model'])

        from utils.upernet import PretrainNet
        
        return PretrainNet(1, 5)
        # from newtwork.unet_model import UNet
        # return UNet(1, 5)

    def pick_loss(self, loss_name='sce'):
        if loss_name == 'sce':
            loss = SegmentationCrossEntropyLoss()
        if loss_name == 'tversky':
            loss = TverskyLoss()
        if loss_name == 'focal_tversky':
            loss = FocalTversky_loss()
        if loss_name == 'lovasz':
            loss = LovaszSoftmax()
        if loss_name == 'haus':
            loss = HausdorffDTLoss()
        if loss_name == 'sdl':
            loss = SoftDiceLoss()
        if loss_name == 'asym':
            loss = AsymLoss()
        return loss



if __name__ == "__main__": # '/media/ExtHDD01/Dataset/OAI_DESS_segmentation/ZIB_3D/train_masks/png/*'
    train_filenames = glob.glob('/media/ExtHDD01/Dataset/OAI_DESS_segmentation/ZIB_3D/original/*')
    # val_filenames = glob.glob('/media/ExtHDD01/Dataset/ICTS/ji/val-3d/*')
    # test_filenames = glob.glob('/media/ExtHDD01/Dataset/ICTS/ji/test-3d/*')
    
    filenames = train_filenames[:300] # + val_filenames + test_filenames
    test_filenames = train_filenames[300:400]
    
    dst_root = '/media/ExtHDD01/OAISeg_log/'
    # Training Arguments
    args = parse()
    args.lr_base = args.lr
    args.betas = (args.beta1, args.beta2)
    progressbar = Progressbar()

    os.makedirs(join(dst_root, args.experiment_name), exist_ok=True)
    os.makedirs(join(dst_root, args.experiment_name, 'checkpoint'), exist_ok=True)
    writer = SummaryWriter(join(dst_root, args.experiment_name, 'summary'))

    with open(join(dst_root, args.experiment_name, 'setting.txt'), 'w') as f:
        f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

    # Dataloader
    train_set = OAI_pretrain(filenames)
    train_loader = DataLoader(train_set, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=0, drop_last=True)
    test_set = OAI_pretrain(test_filenames)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=0, drop_last=True)
    print('Length of training set/ testing set')
    print(len(train_set), len(test_set))

    classifier = Classifier(args, net=args.net)
    it = 0
    it_per_epoch = len(train_set) // (args.batch_size_per_gpu)
    for epoch in range(args.epochs):
        lr = args.lr_base * (0.95 ** epoch)
        classifier.set_lr(lr)
        classifier.train()
        writer.add_scalar('LR/learning_rate', lr, it + 1)
        total_acc = 0
        test_total_acc = 0
        total_count = 0
        for img_a, img_b in progressbar(train_loader):
            img_a = img_a.reshape(img_a.shape[0] * img_a.shape[1], 1, img_a.shape[2], img_a.shape[3])
            img_b = img_b.reshape(img_b.shape[0] * img_b.shape[1], 1, img_b.shape[2], img_b.shape[3])
            all_img = torch.cat([img_a, img_b], dim=0).cuda()

            features = classifier.model(all_img)
            
            B2 = features.shape[0]

            feature_a, feature_b = features[:B2//2], features[B2//2:]


            sim_map = feature_a @ feature_b.T
            label = torch.arange(B2//2, device=sim_map.device)
            
            loss_a = F.cross_entropy(sim_map * 10, label)
            loss_b = F.cross_entropy(sim_map.T * 10, label)
            
            loss = loss_a + loss_b
            
            classifier.optim_model.zero_grad()
            loss.backward()
            classifier.optim_model.step()
            
            with torch.no_grad():
                pred = torch.argmax(sim_map, dim=1)
                acc = (pred == label).float().mean()
            
            total_acc += float(acc)
            total_count += 1
            progressbar.say(epoch=epoch, d_loss=float(loss), acc=total_acc / total_count)
            it += 1

        if epoch % 5 == 0:
            classifier.save(os.path.join(
                dst_root, args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))
        for img_a, img_b in progressbar(test_loader):
            img_a = img_a.reshape(img_a.shape[0] * img_a.shape[1], 1, img_a.shape[2], img_a.shape[3])
            img_b = img_b.reshape(img_b.shape[0] * img_b.shape[1], 1, img_b.shape[2], img_b.shape[3])
            all_img = torch.cat([img_a, img_b], dim=0).cuda()
            with torch.no_grad():
                features = classifier.model(all_img)

            B2 = features.shape[0]

            feature_a, feature_b = features[:B2//2], features[B2//2:]
            sim_map = feature_a @ feature_b.T
            label = torch.arange(B2//2, device=sim_map.device)
            pred = torch.argmax(sim_map, dim=1)
            acc = (pred == label).float().mean()

            test_total_acc += float(acc)
            total_count += 1
            progressbar.say(epoch=epoch, acc=test_total_acc / total_count)
