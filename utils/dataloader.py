import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn.functional as F
import tifffile as tiff
import random
import torchio as tio
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_root, mode):
        ids = os.listdir(img_root) # farm type
        self.img_root = img_root
        self.mask_root = img_root.replace('images-3d', 'masks-3d')
        train_set, test_set = train_test_split(ids, test_size=0.25, random_state=42)
        self.mode = mode
        if mode == 'train':
            dataset = train_set
        elif mode == 'test':
            dataset = test_set

        self.CHANNEL = 5

        self.data = []
        for filename in tqdm(dataset):
            img = os.path.join(self.img_root, filename)
            mask = os.path.join(self.mask_root, filename)
            data = np.load(img)
            for i in range(len(data)-self.CHANNEL+1):
                self.data.append((img, mask, i))
            



    def __getitem__(self, index):
        img, mask, start = self.data[index]
        img = np.load(img)
        mask = np.load(mask)
        
        min = img.min()
        max = img.max() + 1e-6
        
        img = torch.FloatTensor(img[start:start+self.CHANNEL])
        mask = torch.FloatTensor(mask[start:start+self.CHANNEL])

        img = (img - min) / (max - min)


        img = F.interpolate(img[None], (128, 128), mode='bilinear')[0]
        mask = F.interpolate(mask[None], (128, 128), mode='bilinear')[0]
        # img = np.resize(img, (self.CHANNEL, 128, 128))
        # mask = np.resize(mask, (self.CHANNEL, 128, 128))
        
        mask = torch.where(mask > 0.25, 1, 0)
        # return img, mask, 0
        return img, mask[self.CHANNEL//2], 0

    def __len__(self):
        return len(self.data)




class CustomShapeDataset(Dataset):
    def __init__(self, img_root, shape):
        dataset = os.listdir(img_root) # farm type
        self.img_root = img_root
        self.mask_root = img_root.replace('images-3d', 'masks-3d')

        self.CHANNEL = 5

        self.data = []
        for filename in tqdm(dataset):
            img = os.path.join(self.img_root, filename)
            mask = os.path.join(self.mask_root, filename)
            data = np.load(img)
            for i in range(len(data)-self.CHANNEL+1):
                self.data.append((img, mask, i))


    def __getitem__(self, index):
        img, mask, start = self.data[index]
        img = np.load(img)
        mask = np.load(mask)

        min = img.min()
        max = img.max() + 1e-6

        img = torch.FloatTensor(img[start:start+self.CHANNEL])
        mask = torch.FloatTensor(mask[start:start+self.CHANNEL])

        img = (img - min) / (max - min)

        # img = np.resize(img, (self.CHANNEL, 128, 128))
        # mask = np.resize(mask, (self.CHANNEL, 128, 128))

        mask = torch.where(mask > 0.25, 1, 0)
        # return img, mask, 0
        return img, mask[self.CHANNEL//2], 0

    def __len__(self):
        return len(self.data)




class PretrainDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

        self.CHANNEL = 5

    def __getitem__(self, index):
        img = np.load(self.filenames[index])

        min = img.min()
        max = img.max() + 1e-6

        img = (img - min) / (max - min)

        images = []

        for _ in range(2):
            if _ == 0:
                dim = 0
            else:
                dim = np.random.randint(0, 3)

            start = np.random.randint(0, img.shape[dim] - self.CHANNEL + 1)

            if dim == 0:
                img1 = torch.FloatTensor(img[start:start+self.CHANNEL])
            elif dim == 1:
                img1 = torch.FloatTensor(img[:, start:start+self.CHANNEL])
                if np.random.rand() < 0.5:
                    img1 = img1.permute(1, 0, 2).contiguous()
                else:
                    img1 = img1.permute(1, 2, 0).contiguous()

            elif dim == 2:
                img1 = torch.FloatTensor(img[:, :, start:start+self.CHANNEL])
                if np.random.rand() < 0.5:
                    img1 = img1.permute(2, 0, 1).contiguous()
                else:
                    img1 = img1.permute(2, 1, 0).contiguous()

            else:
                print("???")

            img1 = F.interpolate(img1[None], (128, 128), mode='bilinear')[0]
            images.append(img1)


        return images[0], images[1]

    def __len__(self):
        return len(self.filenames)


class MixPretrainDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

        self.CHANNEL = 5

    def __getitem__(self, index):
        img = np.load(self.filenames[index])
        
        try:
            mask = np.load(self.filenames[index].replace('images-3d', 'masks-3d'))
        except:
            mask = None
        
        min = img.min()
        max = img.max() + 1e-6
        
        img = (img - min) / (max - min)
        
        images = []

        for _ in range(2):
            if _ == 0:
                dim = 0
            else:
                dim = np.random.randint(0, 3)

            start = np.random.randint(0, img.shape[dim] - self.CHANNEL + 1)

            if dim == 0:
                img1 = torch.FloatTensor(img[start:start+self.CHANNEL])
                if (_ == 0) and (mask is not None):
                    mask = torch.FloatTensor(mask[start+self.CHANNEL//2][None])
                
            elif dim == 1:
                img1 = torch.FloatTensor(img[:, start:start+self.CHANNEL])
                if np.random.rand() < 0.5:
                    img1 = img1.permute(1, 0, 2).contiguous()
                else:
                    img1 = img1.permute(1, 2, 0).contiguous()
                    
            elif dim == 2:
                img1 = torch.FloatTensor(img[:, :, start:start+self.CHANNEL])
                if np.random.rand() < 0.5:
                    img1 = img1.permute(2, 0, 1).contiguous()
                else:
                    img1 = img1.permute(2, 1, 0).contiguous()

            else:
                print("???")

            img1 = F.interpolate(img1[None], (128, 128), mode='bilinear')[0]
            images.append(img1)

        if mask is None:
            mask = torch.ones_like(images[0][0]).long() * -1
        else:
            mask = F.interpolate(mask[None], (128, 128), mode='bilinear')[0]
            mask = torch.where(mask > 0.25, 1, 0).long()[0]

        return images[0], images[1], mask

    def __len__(self):
        return len(self.filenames)


class OAI_SEG(Dataset):
    def __init__(self, filenames, mode):
        self.img_root = filenames
        # self.mask_root = self.img_root.replace('original', 'train_masks/png')
        self.size = 256
        self.aug = ['ela', 'aff', 'bias', 'gam']
        self.mode = mode


    def __getitem__(self, index):
        raw_img = tiff.imread(self.img_root[index])
        raw_mask = tiff.imread(self.img_root[index].replace('original', 'train_masks/png')).astype(np.long) #(384, 384)

        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min()).astype(np.float32)
        if self.mode == 'train':
            y, z = raw_img.shape
            y, z = y-self.size, z-self.size
            y, z = random.randint(0, y), random.randint(0, z)
            raw_img = torch.from_numpy(raw_img[y:y+self.size, z:z+self.size])
            raw_mask = torch.from_numpy(raw_mask[y:y+self.size, z:z+self.size])
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(raw_img, dim=-1).unsqueeze(0)),
                label=tio.LabelMap(tensor=torch.unsqueeze(raw_mask, dim=-1).unsqueeze(0)),
            )
            augment = self.get_augmentation_transform()
            transformed = augment(subject)
            raw_img = transformed.image.data.view((raw_img.shape[0], raw_img.shape[1]))
            raw_mask = transformed.label.data.type(torch.long).view((raw_mask.shape[0], raw_mask.shape[1]))
        else:
            raw_img = torch.from_numpy(raw_img)
            raw_mask = torch.from_numpy(raw_mask)
        raw_img = raw_img.unsqueeze(0)
        raw_img = transforms.Normalize((0.5), (0.5))(raw_img)

        # raw_mask = raw_mask.unsqueeze(0)
        # raw_mask = F.one_hot(raw_mask, num_classes=5)

        return raw_img, raw_mask

    def get_augmentation_transform(self):
        FUNCTION_MAP = {'ela': tio.RandomElasticDeformation(max_displacement=(7, 7, 0), p=0.5),
                        'aff': tio.RandomAffine(p=0.5),
                        'bias': tio.RandomBiasField(p=0.5),
                        'gam': tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
                        'noi': tio.RandomNoise(p=0.5),
                        'mot': tio.RandomMotion(p=0.5),
                        'bia': tio.RandomBiasField(p=0.5),
                        'spa': tio.OneOf({tio.RandomElasticDeformation(): 0.5,
                                     tio.RandomAffine(): 1-0.5}
                                     )}
        aug_list = []

        for i in self.aug:
            aug = FUNCTION_MAP[i]
            aug_list.append(aug)
        augment = tio.Compose(aug_list)

        return augment

    def __len__(self):
        return len(self.img_root)


class OAI_MIX_SEG(Dataset):
    def __init__(self, filenames, mode):
        self.img_root = filenames
        # self.mask_root = self.img_root.replace('original', 'train_masks/png')
        self.size = 256
        self.aug = ['bias', 'gam', 'ela', 'aff'] # ,
        self.mode = mode


    def __getitem__(self, index):
        raw_img = tiff.imread(self.img_root[index])
        raw_img[:, :10] = 0
        raw_img[:, -10:] = 0
        raw_mask = tiff.imread(self.img_root[index].replace('original', 'train_masks/png')).astype(np.long) #(384, 384)

        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min()).astype(np.float32)
        if self.mode == 'train':
            y, z = raw_img.shape
            y, z = y-self.size, z-self.size
            y, z = random.randint(0, y), random.randint(0, z)
            raw_img = torch.from_numpy(raw_img[y:y+self.size, z:z+self.size])
            raw_mask = torch.from_numpy(raw_mask[y:y+self.size, z:z+self.size])
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.unsqueeze(raw_img, dim=-1).unsqueeze(0)),
                label=tio.LabelMap(tensor=torch.unsqueeze(raw_mask, dim=-1).unsqueeze(0)),
            )
            augment = self.get_augmentation_transform()
            transformed = augment(subject)
            aug_img = transformed.image.data.view((raw_img.shape[0], raw_img.shape[1]))
            aug_mask = transformed.label.data.type(torch.long).view((raw_mask.shape[0], raw_mask.shape[1]))

            aug_img = aug_img.unsqueeze(0)
            aug_img = transforms.Normalize((0.5), (0.5))(aug_img)
            raw_img = raw_img.unsqueeze(0)
            raw_img = transforms.Normalize((0.5), (0.5))(raw_img)
            return raw_img, raw_mask, aug_img, aug_mask
        else:
            raw_img = torch.from_numpy(raw_img)
            raw_mask = torch.from_numpy(raw_mask)
            raw_img = raw_img.unsqueeze(0)
            raw_img = transforms.Normalize((0.5), (0.5))(raw_img)

            return raw_img, raw_mask

    def get_augmentation_transform(self):
        FUNCTION_MAP = {'ela': tio.RandomElasticDeformation(num_control_points=(7, 7, 7), max_displacement=(7, 7, 0), p=0.5),
                        'aff': tio.RandomAffine(p=0.5),
                        'bias': tio.RandomBiasField(p=0.5),
                        'gam': tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
                        'noi': tio.RandomNoise(p=0.5),
                        'mot': tio.RandomMotion(p=0.5),
                        'bia': tio.RandomBiasField(p=0.5),
                        'spa': tio.OneOf({tio.RandomElasticDeformation(): 0.5,
                                     tio.RandomAffine(): 1-0.5}
                                     )}
        aug_list = []

        for i in self.aug:
            aug = FUNCTION_MAP[i]
            aug_list.append(aug)
        augment = tio.Compose(aug_list)

        return augment

    def __len__(self):
        return len(self.img_root)


class OAI_pretrain(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames
        self.size = 128
        self.aug = ['ela', 'aff', 'bias', 'gam'] # , 'noi'

    def __getitem__(self, index):
        raw_img = tiff.imread(self.filenames[index])
        raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min()).astype(np.float32)
        x, y, z = raw_img.shape
        x, y, z = x-self.size, y-self.size, z-self.size
        x, y, z = random.randint(0, x), random.randint(0, y), random.randint(0, z)
        raw_img = torch.from_numpy(raw_img[x:x+self.size, y:y+self.size, z:z+self.size])
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=torch.unsqueeze(raw_img, dim=-1)),
        )
        augment = self.get_augmentation_transform()
        transformed = augment(subject)
        img = transformed.image.data.view((raw_img.shape[0], raw_img.shape[1], raw_img.shape[2]))

        rand_choose = random.randint(0, 7)
        raw_list = []
        trans_list = []
        for slice in range(int(raw_img.shape[0]/8)):
            slice_num = slice * 8
            raw_list.append(raw_img[slice_num + rand_choose:slice_num + rand_choose+1, ::])
            trans_list.append(img[slice_num + rand_choose: slice_num + rand_choose+1, ::])
        raw_img = torch.cat(raw_list, 0)
        img = torch.cat(trans_list, 0)
        # raw_img = transforms.Normalize((0.5), (0.5))(raw_img)
        # img = transforms.Normalize((0.5), (0.5))(img)
        return raw_img, img

    def get_augmentation_transform(self):
        FUNCTION_MAP = {'ela': tio.RandomElasticDeformation(p=0.5),
                        'aff': tio.RandomAffine(degrees=(-10, 10, -10, 10, -10, 10), p=0.5),
                        'bias': tio.RandomBiasField(coefficients=(0, 0.3), p=0.5),
                        'gam': tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
                        'noi': tio.RandomNoise(p=0.5),
                        'mot': tio.RandomMotion(p=0.5),
                        'bia': tio.RandomBiasField(p=0.5),
                        'spa': tio.OneOf({tio.RandomElasticDeformation(): 0.5,
                                     tio.RandomAffine(): 1-0.5}
                                     )}
        aug_list = []

        for i in self.aug:
            aug = FUNCTION_MAP[i]
            aug_list.append(aug)
        augment = tio.Compose(aug_list)

        return augment

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    import glob
    from torch.utils.data import DataLoader
    train_filenames = glob.glob('/media/ExtHDD01/Dataset/OAI_DESS_segmentation/ZIB_3D/original/*')
    train_set = OAI_pretrain(train_filenames)
    train_loader = DataLoader(train_set, batch_size=3, shuffle=True, num_workers=10,
                              drop_last=True)