import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)


try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI \' for CUDA 10:\n\
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100\n\
for CUDA 11.0:\n\
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110\'")


def get_cifar100_dataloader(batch_size=128, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(cifar100_training)

    cifar100_training_loader = torch.utils.data.DataLoader(
        cifar100_training, shuffle=(train_sampler is None), num_workers=num_workers, batch_size=batch_size, sampler=train_sampler)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


    train_sampler = torch.utils.data.distributed.DistributedSampler(cifar100_test)

    cifar100_test_loader = torch.utils.data.DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=100, sampler=train_sampler)

    return cifar100_training_loader, cifar100_test_loader

def get_cifar10_dataloader(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=4, sampler=train_sampler)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(testset)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4, sampler=train_sampler)

    return trainloader, testloader

@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

def dali_get_imagenet_dataloader(batch_size=256, dataset_path=None, model_arch=None):
    img_dir = dataset_path
    traindir = os.path.join(img_dir, 'train')
    valdir = os.path.join(img_dir, 'validation')


    crop_size = 224
    val_size = 256
    if model_arch == 'inception_v3':
        val_size, crop_size = 342, 299
    elif model_arch.startswith('efficientnet_'):
        sizes = {
            'b0': (256, 224), 'b1': (256, 240), 'b2': (288, 288), 'b3': (320, 300),
            'b4': (384, 380), 'b5': (489, 456), 'b6': (561, 528), 'b7': (633, 600),
        }
        e_type = model_arch.replace('efficientnet_', '')
        val_size, crop_size = sizes[e_type]
        # interpolation = InterpolationMode.BICUBIC

    pipe = create_dali_pipeline(batch_size=batch_size,
                                num_threads=6,
                                device_id=torch.distributed.get_rank(),
                                seed=12 + torch.distributed.get_rank(),
                                data_dir=traindir,
                                crop=crop_size,
                                size=val_size,
                                shard_id=torch.distributed.get_rank(),
                                num_shards=torch.distributed.get_world_size(),
                                is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    pipe = create_dali_pipeline(batch_size=batch_size,
                                num_threads=6,
                                device_id=torch.distributed.get_rank(),
                                seed=12 + torch.distributed.get_rank(),
                                data_dir=valdir,
                                crop=crop_size,
                                size=val_size,
                                shard_id=torch.distributed.get_rank(),
                                num_shards=torch.distributed.get_world_size(),
                                is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # test_dataset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ]))
    # val_sampler = torch.utils.data.distributed.DistributedSampler(
    #     test_dataset)
    # val_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=batch_size, shuffle=False,
    #     num_workers=6, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


def get_imagenet_dataloader(batch_size=256, dataset_path=None):
    img_dir = dataset_path
    traindir = os.path.join(img_dir, 'train')
    valdir = os.path.join(img_dir, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_sampler = None

    # Partition dataset among workers using DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=6, pin_memory=True, sampler=train_sampler)

    test_dataset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    # Partition dataset among workers using DistributedSampler
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader

def get_imagenet_dataloader_official(batch_size=256, dataset_path=None):
    print('==> Using Pytorch Dataset')
    img_dir = dataset_path
    input_size = 224  # image resolution for resnets
    traindir = os.path.join(img_dir, 'train')
    valdir = os.path.join(img_dir, 'validation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=6, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=6, pin_memory=True)
    return train_loader, val_loader


def get_dataloader(name, batch_size, dataset_path, model_arch=None, *arg, **kargs):
    if name == 'cifar10':
        return get_cifar10_dataloader(batch_size, dataset_path, *arg, **kargs)
    elif name == 'cifar100':
        return get_cifar100_dataloader(batch_size, dataset_path, *arg, **kargs)
    elif name == 'imagenet':
        return dali_get_imagenet_dataloader(batch_size, dataset_path, model_arch=model_arch, *arg, **kargs)
        # return get_imagenet_dataloader(batch_size, dataset_path, *arg, **kargs)
        # return get_imagenet_dataloader_official(batch_size, dataset_path, *arg, **kargs)
    else:
        raise RuntimeError("Unkown dataloader")