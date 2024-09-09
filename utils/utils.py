import os
import gc
import yaml
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset
from typing import Type, Union, Tuple, TypeVar, ParamSpec, Callable, List


def setup():
    gc.collect()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def get_loader(root_dir: str, batch_size: int, transforms) -> DataLoader:
    # Methods for dealing with imbalanced datasets:
    # 1. Oversampling (probably preferable)
    # 2. Class weighting
    dataset = ImageFolder(root=root_dir, transform=transforms)

    subdirectories = dataset.classes
    class_weights = []

    for subdir in subdirectories:
        files = os.listdir(os.path.join(root_dir, subdir))
        class_weights.append(1 / len(files))

    sample_weights = [0] * len(dataset)

    for idx, (_, label) in enumerate(dataset):
        sample_weights[idx] = class_weights[label]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True),
        num_workers=8,
        pin_memory=True,
    )


def get_mean_std(loader: DataLoader):
    channels_sum: torch.Tensor = 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])

    mean: torch.Tensor = channels_sum / len(loader)
    std: torch.Tensor = torch.std(loader.dataset)

    return mean, std


def seed_everything(seed=420):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import contextlib


@contextlib.contextmanager
def ignore_warning(warning: Type[Warning]):
    """Context manager to ignore a specific warning type.
    Args:
        warning (Type[Warning]): The type of warning to ignore.
    """
    import warnings

    with warnings.catch_warnings():  # Catches all warnings that occur within the context
        warnings.filterwarnings("ignore", category=warning)  # Ignores the specified warning type
        yield  # Yields control back to the caller while ignoring the warning


from functools import wraps

T = TypeVar("T")
P = ParamSpec("P")


def add_logging(f: Callable[P, T]) -> Callable[P, T]:
    """A type-safe decorator to add logging to a function.
    Args:
        f: Function to be decorated.
    Returns:
        Decorated function.
    """

    @wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T:
        class_name = args[0].__class__.__name__ if args and args[0] else None  # Get the class name if available
        method_name = f.__name__  # Get the name of the method
        print(f"{class_name}.{method_name} was called" if class_name else f"{method_name} was called")  # Log the function call
        return f(*args, **kwargs)  # Call the original function

    return inner


class ImageMaskDataset(Dataset):
    def __init__(self, folder: str, size: List[int], transform=None, target_transform=None):
        """Initialize an ImageMaskDataset instance.
        Args:
            folder (str): The path to the dataset folder.
            size (tuple): The expected size of the images and masks.
            transform (callable, optional): The data transformation function for images. Defaults to None.
            target_transform (callable, optional): The data transformation function for masks. Defaults to None.
        """
        self.dataset_folder = os.path.join(folder, f"{size[0]}_{size[1]}")  # Set the dataset folder path based on the size
        self.image_files = [filename for filename in os.listdir(self.dataset_folder) if not "_mask" in filename]  # Get the image files in the dataset folder
        self.transform = transform  # Set the data transformation function for images
        self.target_transform = target_transform  # Set the data transformation function for masks
        self.expected_size = size  # Set the expected size of the images and masks

    def __len__(self):
        """Get the length of the dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int):
        """Get an item from the dataset.
        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image and mask tensors.
        """
        image_path = os.path.join(self.dataset_folder, self.image_files[idx])  # Get the path of the image
        mask_path = os.path.join(self.dataset_folder, self.image_files[idx].split(".")[0] + "_mask." + self.image_files[idx].split(".")[1])  # Get the path of the corresponding mask
        image = to_tensor(Image.open(image_path).convert("RGB"))  # Open the image and convert it to a tensor
        mask = to_tensor(Image.open(mask_path).convert("L"))  # Open the mask and convert it to a tensor
        assert [image.shape[1], image.shape[2]] == self.expected_size  # Check if the image size matches the expected size
        assert [mask.shape[1], mask.shape[2]] == self.expected_size  # Check if the mask size matches the expected size
        if self.transform:
            image = self.transform(image)  # Apply the data transformation function to the image
        if self.target_transform:
            mask = self.target_transform(mask)  # Apply the data transformation function to the mask
        return image, mask


class CustomDataset(Dataset):
    def __init__(self, folder: str, size: List[int], transform=None) -> None:
        """Initialize a CustomDataset instance.
        Args:
            folder (str): The path to the dataset folder.
            size (tuple, optional): The expected size of the images. Defaults to None.
            transform (callable, optional): The data transformation function. Defaults to None.
        """
        self.data = pd.read_csv(os.path.join(folder, "image_labels.csv"))  # Read the image labels from a CSV file
        self.transform = transform  # Set the data transformation function
        self.base_dir = folder  # Set the base directory of the dataset
        self.class_names = self.data.columns[1:].to_list()  # Get the class names from the dataset columns
        self.image_folder = f"{size[0]}_{size[1]}"  # Set the image folder based on the size
        self.expected_size = size  # Set the expected size of the images

    def __len__(self):
        """Get the length of the dataset.
        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.
        Args:
            idx (int or torch.Tensor): The index of the item to retrieve.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image and label tensors.
        """
        if torch.is_tensor(idx):
            idx: list = idx.tolist()  # Convert the index to a list if it's a tensor
        img_path = os.path.join(self.base_dir, self.image_folder, self.data.iloc[idx, 0])  # Get the image path

        labels = self.data.iloc[idx, 1:].values  # Get the labels for the item
        labels = np.array(labels, dtype=np.float32)

        label_tensor = torch.from_numpy(labels)  # Convert the labels to a tensor

        img = to_tensor(Image.open(img_path).convert("RGB"))  # Open the image and convert it to a tensor

        # Check if the image size matches the expected size
        assert [img.shape[1], img.shape[2]] == self.expected_size

        if self.transform:
            img = self.transform(img)  # Apply the data transformation function to the image

        return img, label_tensor


class ConfigParser:
    def __init__(self, config_file):
        """Initialize a ConfigParser instance.
        Args:
            config_file (str): The path to the configuration file.
        """
        self.config_file = config_file
        self.variables = {}

    def parse(self):
        """Parse the configuration file and extract the variables."""
        with open(self.config_file, "r") as file:
            config: dict = yaml.safe_load(file)  # Load the YAML configuration file
        self.variables = self._extract_variables(config)  # Extract the variables from the configuration

    def _extract_variables(self, config: dict):
        """Extract the variables from the configuration dictionary.
        Args:
            config (dict): The configuration dictionary.
        Returns:
            dict: The extracted variables.
        """
        config_varibles = {}
        training_variables = {}
        for section, section_values in config.items():
            if section == "Boris":
                config_varibles.update(section_values)  # Update the Boris section variables
            else:
                training_variables.update(section_values)  # Update the Training section variables
        variables = {}
        variables["Boris"] = self._process_section_variables(config_varibles)  # Process the Boris section variables
        variables["Training"] = self._process_section_variables(training_variables)  # Process the Training section variables
        return variables

    def _process_section_variables(self, section_variables: dict):
        """Process the variables in a section.
        Args:
            section_variables (dict): The variables in a section.
        Returns:
            dict: The processed variables.
        """
        variables = {}
        for key, value in section_variables.items():
            if isinstance(value, dict):
                nested_variables = self._process_section_variables(value)  # Recursively process nested variables
                variables.update(nested_variables)  # Update the variables dictionary
            else:
                if isinstance(value, str) and value.lower() == "none":
                    value = None  # Convert "None" string to None
                variables[key] = value  # Add the variable to the dictionary
        return variables

    def get_variables(self):
        """Get the extracted variables.
        Returns:
            dict: The extracted variables.
        """
        return self.variables
