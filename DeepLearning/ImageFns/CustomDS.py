import json
import os
import random

import torchvision
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import numpy as np

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def check_valid_from_file(file,additional_check):
    fp = open(file)
    data = json.load(fp).keys()
    def check_valid(file_path):
        check = file_path.split('/')[-2]
        extension = file_path.split('/')[-1].split('.')[-1]
        if extension == 'JPEG':
            if check in data:
                if additional_check is None:
                    return True
                else:
                    return additional_check(file_path)
            else:
                return False
        else:
            return False

    return check_valid

def find_classes_from_map(map_file):
    fp = open(map_file)
    class_to_idx = json.load(fp)
    classes = list(class_to_idx.keys())
    sorted_classes = sorted(classes)

    return sorted_classes, class_to_idx


def find_all_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class Custom_ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self,
                root: str,
                map_file: Optional[str] = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                loader: Callable[[str], Any] = torchvision.datasets.folder.default_loader,
                is_valid_file: Optional[Callable[[str], bool]] = None,
        ) -> None:
        self.map_file = map_file
        self.all_samples = None
        if self.map_file is not None:
            is_valid_file = check_valid_from_file(self.map_file,is_valid_file)

        super(Custom_ImageFolder, self).__init__(root, transform=transform, target_transform=target_transform,
                                                   loader=loader, is_valid_file=is_valid_file)


    def find_classes(self, directory):
        if self.map_file is not None:
            return find_classes_from_map(self.map_file)

        return find_all_classes(directory)

    def select(self, N):
        if self.all_samples is None:
            self.all_samples = self.samples

        self.samples = random.sample(self.all_samples,N)
