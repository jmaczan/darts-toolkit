import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config["data"]["data_dir"]
        self.batch_size = config["data"]["batch_size"]
        self.num_workers = config["data"]["num_workers"]

        self.transform = transforms.Compose(
            [
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),  # CIFAR10 - see https://github.com/kuangliu/pytorch-cifar/issues/8
            ]
        )

        # transforms without data augmentation
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    def prepare_data(self):
        datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(
                root=self.data_dir, train=True, transform=self.transform
            )
            train_size = int(0.8 * len(cifar_full))
            val_size = len(cifar_full) - train_size

            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [train_size, val_size]
            )

            search_size = int(0.5 * len(self.cifar_train))
            self.cifar_train, self.cifar_search = random_split(
                self.cifar_train, [len(self.cifar_train) - search_size, search_size]
            )

        if stage == "test" or stage is None:
            self.cifar_test = datasets.CIFAR10(
                root=self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return {
            "train": DataLoader(
                self.cifar_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            ),
            "search": DataLoader(
                self.cifar_search,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True,
            ),
        }

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )
