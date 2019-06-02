import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path


def make_dataset(filename, datadir, category):
    images = []
    labels = []
    with open(os.path.join(filename), "r") as lines:
        for line in lines:
            info = line.split(',')
            _path = info[0]
            _cat = info[1]
            _classes = info[2]

            if _cat == category:
                _image = os.path.join(datadir, _path)
                assert os.path.isfile(_image)
                label = _classes.find('y')
                images.append(_image)
                labels.append(label)

    return images, labels


class FashionAIDataloder(data.Dataset):
    def __init__(self, root, images, labels, transform=None):
        self.root = root
        self.images = images
        self.labels = labels
        self.transform = transform
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.images[index])
        _img = Image.open(image_path).convert('RGB')
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

        all_images, all_labels = make_dataset(config.train_source, config.dataset_path, config.category)
        split_ratio = 0.8
        cut = int(len(all_images) * split_ratio)
        train_images = all_images[:cut]
        train_labels = all_labels[:cut]
        val_images = all_images[cut:]
        val_labels = all_labels[cut:]
        # print(len(train_images), len(train_labels), len(val_images), len(val_labels))
        
        trainset = FashionAIDataloder(config.dataset_path, train_images, train_labels, transform=transform_train)
        testset = FashionAIDataloder(config.dataset_path, val_images, val_labels, transform=transform_test)

        kwargs = {'num_workers': 0, 'pin_memory': True}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
            config.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
            config.batch_size, shuffle=False, **kwargs)
        # self.classes = trainset.classes
        self.trainloader = trainloader 
        self.testloader = testloader
    
    def getloader(self):
        return self.trainloader, self.testloader
