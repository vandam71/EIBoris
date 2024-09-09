from __future__ import annotations
import torch
import numpy as np
from tqdm import tqdm
from ss import StorageSystem
from ls import LearningSystem
from cs import ComputationSystem
from torch.utils.data import DataLoader
from typing import Tuple, Optional, List, Union
from utils import ImageMaskDataset, CustomDataset, ConfigParser


class Boris(object):
    def __init__(
        self,
        segmentation: Optional[str] = None,
        classification: Optional[str] = None,
        size: List[int, int] = (224, 224),
        wm_size: int = 7,
        num_classes: Optional[int] = None,
        attention: bool = True,
        net_type: str = "resnet50",
        use_segmentation: bool = True,
        min_samples_split: int = 8,
        max_depth: Optional[int] = None,
    ) -> None:
        """Boris object constructor.
        Args:
            segmentation (str, optional): Path to the segmentation dataset. Defaults to None.
            classification (str, optional): Path to the classification dataset. Defaults to None.
            size (List[int, int], optional): Image size. Defaults to (224, 224).
            wm_size (int, optional): Size of the working memory. Defaults to 7.
            num_classes (int, optional): Number of output classes. Required if classification dataset is not provided. Defaults to None.
            attention (bool, optional): Flag to indicate attention usage. Defaults to True.
            net_type (str, optional): Network type for ComputationSystem. Defaults to "resnet50".
            use_segmentation (bool, optional): Flag to indicate whether segmentation is used. Defaults to True.
            min_samples_split (int, optional): Minimum number of samples required to split a node in ComputationSystem. Defaults to 8.
            max_depth (int, optional): Maximum depth of the decision tree in ComputationSystem. Defaults to None.
        """
        self.seg_data = ImageMaskDataset(segmentation, size) if segmentation and use_segmentation else None  # Initialize segmentation dataset if provided
        self.class_data = CustomDataset(classification, size) if classification else None  # Initialize classification dataset if provided
        assert self.class_data or num_classes  # Ensure that either classification dataset or number of classes is provided
        # Initialize the subsystems
        self._cs = ComputationSystem(len(self.class_data.class_names) if self.class_data else num_classes, attention, net_type, use_segmentation, min_samples_split, max_depth)  # this initialization needs to depend on the dataset to be used
        self._ls = LearningSystem(self._cs._bl.classifier, self._cs._bl.segmenter)
        self._ss = StorageSystem(wm_size=wm_size)

    def setNumClasses(self, num_classes: int) -> None:
        """Set the number of output classes in the classifier model.
        Args:
            num_classes (int): Number of output classes.
        """
        self._cs._bl.classifier.model.num_classes = num_classes

    def setNumChannels(self, num_channels: int) -> None:
        """Set the number of input channels in both the segmenter and classifier models.
        Args:
            num_channels (int): Number of input channels.
        """
        self._cs._bl.segmenter.model.num_channels = num_channels
        self._cs._bl.classifier.model.num_channels = num_channels

    def __call__(self, X: torch.Tensor) -> List[int]:
        """Perform the forward pass of the Boris instance.
        Args:
            X (torch.Tensor): Input tensor.
        Returns:
            list: Final decision based on probabilities, labels, and working memory influence.
        """
        # TODO: this needs to be updated to LMS computation and have somekind of training process
        (probabilities, labels) = self._cs(X)  # Call the ComputationSystem to obtain probabilities and labels
        wm_influence = self._ss.working_memory.get_influence()  # Get the influence from the working memory
        alpha, beta, omega = 0.33, 0.33, 0.33  # Define weights for combining the outputs
        # Calculate and store the final decision in the current node's data

        StorageSystem().most_recent_node.data.final_decision = ((np.array(probabilities) * alpha + np.array(labels) * beta + np.array(wm_influence) * omega) / 3).tolist()
        return StorageSystem().most_recent_node.data.final_decision  # Return the final decision

    def fit(self, epochs: Union[Tuple[int, int], int], batch_size: int = 32, weighted: bool = False) -> None:
        """Trains the model on the provided datasets.
        Args:
            epochs (Tuple[int, int]): The number of epochs to train the segmentation and classification models.
            batch_size (int, optional): The batch size for training. Defaults to 32.
        """
        imagelabel_dataloader = DataLoader(self.class_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        pos_weight = None

        if weighted:
            num_classes = len(self.class_data.class_names)
            positive_counts = torch.zeros(num_classes)
            for batch in tqdm(imagelabel_dataloader):
                inputs, targets = batch
                positive_counts += targets.sum(dim=0)
            total_samples = len(imagelabel_dataloader.dataset)
            negative_counts = total_samples - positive_counts
            pos_weight = negative_counts / (positive_counts + 1e-5)
            # pos_weight = pos_weight.view(-1, 1)
            # pos_weight = pos_weight.expand(batch_size, num_classes, 1)

            # print(pos_weight.shape, pos_weight)
            # exit()
            print('Class Weights', pos_weight.tolist())

        # Create dataloaders for image masks and image labels
        if self.seg_data:
            imagemask_dataloader = DataLoader(self.seg_data, batch_size=batch_size, shuffle=True, pin_memory=True)
            epochs_seg, epochs_class = (epochs, epochs) if not isinstance(epochs, tuple) else epochs
            # Fit the computation system using the dataloaders and epochs
            self._cs.fit((imagemask_dataloader, epochs_seg), (imagelabel_dataloader, epochs_class, pos_weight))
        else:
            self._cs.fit(None, (imagelabel_dataloader, epochs, pos_weight))
        # TODO: this should have memory training somehow, maybe define how many images do we want to train on and random sample them

    def save(self, save_file: str) -> None:
        """Save the current object to a file.
        Args:
            save_file (str): The file path to save the object to.
        """
        torch.save(self, save_file)

    @classmethod
    def load(cls, path: str) -> Boris:
        """Load a Boris instance from a saved file.
        Args:
            path (str): The file path to load the instance from.
        Returns:
            Boris: The loaded Boris instance.
        """
        instance: Boris = torch.load(path)
        return instance

    @classmethod
    def from_config(cls, config_file: str, load_only: bool = False) -> Boris:
        """Create a Boris instance from a configuration file.
        Args:
            config_file (str): The path to the configuration file.
        Returns:
            Boris: The created Boris instance.
        """
        parser = ConfigParser(config_file)  # Create a ConfigParser instance with the provided configuration file
        parser.parse()  # Parse the configuration file
        params = parser.get_variables()  # Get the variables from the parsed configuration file
        obj = cls(**params["Boris"])  # Create a Boris instance with the parameters extracted from the config
        print("New Boris instance started")

        # Print all configurations
        print("\n--- Configuration Details ---")
        for section, settings in params.items():
            print(f"[{section}]")
            for key, value in settings.items():
                print(f"{key}: {value}")
            print()
        if load_only is True:
            return obj
        if params.get("Training", {}):
            # If there are training parameters specified in the config
            if type(params["Training"]["epochs"]) == int:
                print(f"Training with epochs: {params['Training']['epochs']}, batch_size: {params['Training']['batch_size']}, weighted: {params["Training"].get('weighted', False)}\n")
                obj.fit(params["Training"]["epochs"], params["Training"]["batch_size"], params["Training"].get('weighted', False))  # Fit the Boris instance with the specified epochs and batch size
            else:
                print(f"Training with epochs: {tuple(params['Training']['epochs'])}, batch_size: {params['Training']['batch_size']}, weighted: {params["Training"].get('weighted', False)}")
                obj.fit(tuple(params["Training"]["epochs"]), params["Training"]["batch_size"], params["Training"].get('weighted', False))  # Fit the Boris instance with the specified epochs and batch size
            print(f"Saving trained model to: {params['Training']['save_file']}")
            obj.save(params["Training"]["save_file"])  # Save the trained Boris instance to the specified save file
        return obj  # Return the created Boris instance
