import argparse
import datetime


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='# of epochs')
    parser.add_argument('--batch_size_per_gpu', dest='batch_size_per_gpu', type=int, default=20) # training batch size
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.9)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)

    parser.add_argument('--net', dest='net', default='unet')
    parser.add_argument('--backbone', dest='backbone', default='resnet34')
    parser.add_argument('--loss', dest='loss', default='sce')
    parser.add_argument('--weights', dest='weights', default='imagenet')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    parser.add_argument('--shape', dest='shape', type=int, nargs="+")


    return parser.parse_args(args)