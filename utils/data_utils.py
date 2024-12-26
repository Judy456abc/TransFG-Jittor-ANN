import logging
from PIL import Image
import os

# import torch

# from torchvision import transform
# from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import CUB, CarsDataset, dogs, NABirds, INat2017
# from .dataset import NABirds, INat2017
from .autoaugment import AutoAugImageNetPolicy

import jittor as jt
from jittor.dataset import DataLoader
from jittor import transform

logger = logging.getLogger(__name__)


def get_loader(args):

    if args.dataset == 'CUB_200_2011':
        train_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.RandomCrop((448, 448)),
                                    transform.RandomHorizontalFlip(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.CenterCrop((448, 448)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = CUB(root=args.data_root, is_train=True, transform=train_transform)
        testset = CUB(root=args.data_root, is_train=False, transform = test_transform)
    
    elif args.dataset == 'car':
        trainset = CarsDataset(os.path.join(args.data_root,'devkit/cars_train_annos.mat'),
                            os.path.join(args.data_root,'cars_train'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned.dat'),
                            transform=transform.Compose([
                                    transform.Resize((600, 600), Image.BILINEAR),
                                    transform.RandomCrop((448, 448)),
                                    transform.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
        testset = CarsDataset(os.path.join(args.data_root,'cars_test_annos_withlabels.mat'),
                            os.path.join(args.data_root,'cars_test'),
                            os.path.join(args.data_root,'devkit/cars_meta.mat'),
                            # cleaned=os.path.join(data_dir,'cleaned_test.dat'),
                            transform=transform.Compose([
                                    transform.Resize((600, 600), Image.BILINEAR),
                                    transform.CenterCrop((448, 448)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                            )
    elif args.dataset == 'dog':
        train_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.RandomCrop((448, 448)),
                                    transform.RandomHorizontalFlip(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                    transform.CenterCrop((448, 448)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = dogs(root=args.data_root, train=True, cropped=False, transform=train_transform, download=False)
        testset = dogs(root=args.data_root, train=False, cropped=False, transform=test_transform, download=False)
    
    elif args.dataset == 'nabirds':
        train_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                        transform.RandomCrop((448, 448)),
                                        transform.RandomHorizontalFlip(),
                                        transform.ToTensor(),
                                        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((600, 600), Image.BILINEAR),
                                        transform.CenterCrop((448, 448)),
                                        transform.ToTensor(),
                                        transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = NABirds(root=args.data_root, train=True, transform=train_transform)
        testset = NABirds(root=args.data_root, train=False, transform=test_transform)
    
    elif args.dataset == 'INat2017':
        train_transform=transform.Compose([transform.Resize((400, 400), Image.BILINEAR),
                                    transform.RandomCrop((304, 304)),
                                    transform.RandomHorizontalFlip(),
                                    AutoAugImageNetPolicy(),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform=transform.Compose([transform.Resize((400, 400), Image.BILINEAR),
                                    transform.CenterCrop((304, 304)),
                                    transform.ToTensor(),
                                    transform.ImageNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        trainset = INat2017(root=args.data_root, split='train', transform=train_transform,)
        testset = INat2017(root=args.data_root, split='val', transform=test_transform,)
    
    else:
        raise ValueError("Dataset not supported: {}".format(args.dataset))

    train_sampler = jt.dataset.RandomSampler(trainset)
    test_sampler = jt.dataset.SequentialSampler(testset)
    
    train_loader = DataLoader(trainset, sampler=train_sampler, batch_size=args.train_batch_size)
    test_loader = DataLoader(testset, sampler=test_sampler, batch_size=args.eval_batch_size) if testset is not None else None

    return train_loader, test_loader