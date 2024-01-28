from DeepLearning.ImageFns import *

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


import os



image_path = "DeepLearning/ImageDatasets/ImageNet"


def DLWrap(x, y):
    return x.cuda(), y.cuda()

class datasetHandler:
    def __init__(self):
        """
         dataset_path=None, dataset_type="Images"
        An initializer for a dataset handler.
        :param dataset_path: (optional) path to the dataset folder
        :param dataset_type: (optional) the type of dataset (default: Images)
        """
        self.name = None

        self.training_data = None
        self.testing_data = None

        self.training_loader = None
        self.testing_loader = None

        def to_cuda(x):
            return x.to("cuda:0")

        def label_to_cuda(y):
            return torch.tensor(y).cuda()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            to_cuda
        ])
        self.target_transform = transforms.Compose([
            label_to_cuda
        ])


    def load_dataset(self,name = "mnist", batch_size=64):
        if name == "mnist":
            self.load_mnist(batch_size)
        elif name == "fmnist":
            self.load_fmnist(batch_size)
        elif name == "cifar10":
            self.load_cifar10(batch_size)
        elif name == "cifar100":
            self.load_cifar100(batch_size)
        elif name == "imagenet":
            self.load_imagenet(batch_size)
        elif name == "imagenet_val":
            self.load_imagenet_val(batch_size)

    def load_imagenet(self, batch_size=64):
        self.name = 'imagenet'
        map_file = os.path.join(image_path,'imagenet1000_folders_to_clsidx.txt')
        evaluation_folder = os.path.join(image_path,'val')
        training_folder = os.path.join(image_path,'train')

        image_size = 224
        num_workers = 0

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # CutoutPIL(cutout_factor=0.5),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


        self.training_data = Custom_ImageFolder(root = training_folder, map_file = map_file, transform = train_transform)
        self.testing_data = Custom_ImageFolder(root = evaluation_folder, map_file = map_file, transform = val_transform)
        print("length train dataset: {}".format(len(self.training_data)))
        print("length val dataset: {}".format(len(self.testing_data)))

        sampler_train = None
        sampler_val = None
        # if num_distrib() > 1:
        #     sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #     sampler_val = OrderedDistributedSampler(val_dataset)

        # Pytorch Data loader
        train_loader = torch.utils.data.DataLoader(
            self.training_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, sampler=sampler_train)

        val_loader = torch.utils.data.DataLoader(
            self.testing_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, sampler=sampler_val)

        self.training_loader = iter(PrefetchLoader(train_loader))
        self.testing_loader = iter(PrefetchLoader(val_loader))



    def load_imagenet_val(self, batch_size=64):
        self.name = 'imagenet'
        map_file = os.path.join(image_path,'imagenet1000_folders_to_clsidx.txt')
        evaluation_folder = os.path.join(image_path,'val')
        training_folder = None

        image_size = 224
        num_workers = 0

        train_transform = None
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])


        self.training_data = None
        self.testing_data = Custom_ImageFolder(root = evaluation_folder, map_file = map_file, transform = val_transform)
        print("train dataset not imported")
        print("length val dataset: {}".format(len(self.testing_data)))

        sampler_train = None
        sampler_val = None
        # if num_distrib() > 1:
        #     sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        #     sampler_val = OrderedDistributedSampler(val_dataset)

        # Pytorch Data loader
        train_loader = None

        val_loader = torch.utils.data.DataLoader(
            self.testing_data, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False, sampler=sampler_val)


        self.testing_loader = iter(PrefetchLoader(val_loader))
        self.training_loader = self.testing_loader


    def load_mnist(self,batch_size=64):
        self.name = 'mnist'

        self.training_data = datasets.MNIST(image_path, train=True, transform=self.transform)
        self.testing_data = datasets.MNIST(image_path, train=False, transform=self.transform)

        self.training_loader = iter(DataLoader(self.training_data,batch_size=batch_size,shuffle=True))
        self.testing_loader = iter(DataLoader(self.testing_data, batch_size=batch_size, shuffle=True))

    def load_fmnist(self,batch_size=64):
        self.name = 'fmnist'

        self.training_data = datasets.FashionMNIST(image_path, train=True, transform=self.transform)
        self.testing_data = datasets.FashionMNIST(image_path, train=False, transform=self.transform)

        self.training_loader = iter(DataLoader(self.training_data,batch_size=batch_size,shuffle=True))
        self.testing_loader = iter(DataLoader(self.testing_data, batch_size=batch_size, shuffle=True))

    def load_cifar10(self,batch_size=64):
        self.name = 'cifar10'

        self.training_data = datasets.CIFAR10(image_path, train=True, transform=self.transform, target_transform=self.target_transform)
        self.testing_data = datasets.CIFAR10(image_path, train=False, transform=self.transform, target_transform=self.target_transform)


        self.training_loader = iter(DataLoader(self.training_data,batch_size=batch_size,shuffle=True))
        self.testing_loader = iter(DataLoader(self.testing_data, batch_size=batch_size, shuffle=True))


    def load_cifar100(self,batch_size=64):
        self.name = 'cifar100'

        self.training_data = datasets.CIFAR100(image_path, train=True, transform=self.transform, target_transform=self.target_transform)
        self.testing_data = datasets.CIFAR100(image_path, train=False, transform=self.transform, target_transform=self.target_transform)

        self.training_loader = iter(DataLoader(self.training_data,batch_size=batch_size,shuffle=True))
        self.testing_loader = iter(DataLoader(self.testing_data, batch_size=batch_size, shuffle=True))




    # Download and preprocessing functions
    def download_small_Ds(self):
        self.download_mnist()
        self.download_fmnist()
        self.download_cifar10()
        self.download_cifar100()

    def download_mnist(self):
        """
        Uses tensorflow to download and then preprocesses the MNIST dataset.
        :param save: Set to true to save the dataset.
        :return: None
        """
        datasets.MNIST(image_path, download=True)

    def download_fmnist(self):
        """
        Uses tensorflow to download and then preprocesses the FashionMNIST dataset.
        :param save: Set to true to save the dataset.
        :return: None
        """
        datasets.FashionMNIST(image_path, download=True)

    def download_cifar10(self, save=False, save_dir=None):
        """
        Uses tensorflow to download and then preprocesses the Cifar10 dataset.
        :param save: Set to true to save the dataset.
        :return: None
        """
        datasets.CIFAR10(image_path, download=True)

    def download_cifar100(self, save=False, save_dir=None):
        """
        Uses tensorflow to download and then preprocesses the Cifar100 dataset.
        :param save: Set to true to save the dataset.
        :return: None
        """
        datasets.CIFAR100(image_path, download=True)


if __name__ == '__main__':
    DS = datasetHandler()
    # DS.download_small_Ds()

    check = [152,165,166,167,175,241,248,262,268,619,665,837,850]

    DS.load_imagenet()

    x, y = next(DS.training_loader)
    i = 0
    checks = (y==(check[i]+1))

    while torch.sum(checks) < 1:
        x, y = next(DS.training_loader)
        checks = (y==(check[i]+1))


    plot_grid(x[checks],imagenet_names(y[checks].detach().cpu().numpy()))


g = 0
