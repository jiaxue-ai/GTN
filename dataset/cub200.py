import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(f1, f2):

    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    file1 = open(f1, 'r')
    file2 = open(f2, 'r')

    map1 = {}
    for line in file1:
        info = line.split()
        index = info[0]
        filename = info[1]
        map1[index] = filename

    map2 = {}
    for line in file2:
        info = line.split()
        index = info[0]
        label = info[1]
        map2[index] = label

    for key in map1.keys():
        image = map1[key]
        label = map2[key]
        gt = int(image.split('.')[0]) - 1

        if label == '1':
            train_images.append(image)
            train_labels.append(gt)
        else:
            val_images.append(image)
            val_labels.append(gt)

    return train_images, train_labels, val_images, val_labels


class CUB200Dataloder(data.Dataset):
    def __init__(self, root, image, label, transform=None):
        classes, class_to_idx = find_classes(root + '/images')
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.images, self.labels = image, label
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(os.path.join(self.root, 'images', self.images[index])).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)


class Dataloder():
    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_image, train_label, val_image, val_label = make_dataset(config.split_info1, config.split_info2)

        trainset = CUB200Dataloder(config.dataset_path, 
            train_image, train_label, transform=transform_train)
        testset = CUB200Dataloder(config.dataset_path, 
            val_image, val_label, transform=transform_test)

        kwargs = {'num_workers': 0, 'pin_memory': True}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            config.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            config.batch_size, shuffle=False, **kwargs)
        self.classes = trainset.classes
        self.trainloader = trainloader 
        self.testloader = testloader
    
    def getloader(self):
        return self.trainloader, self.testloader
