import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

from PIL import Image
import os
import os.path
import random


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(filename, datadir, class_to_idx):
    images = []
    labels = []
    with open(os.path.join(filename), "r") as lines:
        for line in lines:
            info = line.split()
            _image = info[0]
            _class = _image.split('_')[1]
            _length = info[1]
            _label = info[2]
            images.append((_image, _class))
            labels.append((_length, _label))

    return images, labels


class UCF101Dataloder(data.Dataset):
    def __init__(self, root, filename, transform=None):
        classes, class_to_idx = find_classes(root + '/frames')
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.images, self.labels = make_dataset(filename, root, class_to_idx)
        # print(len(self.images), len(self.labels))
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        clip_name = self.images[index][0]
        class_name = self.images[index][1]
        clip_path = os.path.join(self.root, 'frames', class_name, clip_name)

        _length = self.labels[index][0]
        _label = int(self.labels[index][1])
        frame_num = random.randint(1, int(_length))
        image_name = 'image_%04d.jpg' % (frame_num)
        image_path = os.path.join(clip_path, image_name)

        if not os.path.exists(image_path):
            # print(image_path)
            image_path = '/media/yyhome/second/UCF101/frames/PullUps/v_PullUps_g22_c04/image_0001.jpg'
            _label = 69

        _img = Image.open(image_path).convert('RGB')

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

        trainset = UCF101Dataloder(config.dataset_path, 
            config.train_source, transform=transform_train)
        testset = UCF101Dataloder(config.dataset_path, 
            config.test_source, transform=transform_test)

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
