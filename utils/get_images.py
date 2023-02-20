import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

def see_test_images(img, pred, slice_num=None):
    if slice_num == None:
        num = img.size[0]
    else:
        num = slice_num
    _, pred = torch.max(pred, 1)
    pred = torch.unsqueeze(pred, 1)
    part_list = []
    for i in range(num):
        img_i = (img[i:i + 1, 0:1, ::] - img[i:i + 1, 0:1, ::].min()) / (
                img[i:i + 1, 0:1, ::].max() - img[i:i + 1, 0:1, ::].min())
        pred_i = (pred[i:i + 1] - pred[i:i + 1].min()) / (
                pred[i:i + 1].max() - pred[i:i + 1].min())
        img_i = torch.cat([img_i, img_i, img_i], 1)
        pred_i = torch.cat([pred_i, pred_i, pred_i], 1)
        grid = torch.cat([img_i, pred_i], 3)  # , COM_B
        part_list.append(grid)
    grid = torch.cat(part_list, 2)
    grid = make_grid(grid)
    file_path = 'seeee.png'
    save_image(grid, file_path)