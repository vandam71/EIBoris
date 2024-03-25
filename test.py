import torch
from utils import ImageMaskDataset, CustomDataset, setup
from torch.utils.data import DataLoader
from cs.bl.segmentation import SegmentationNetwork
from cs.bl.classification import ClassificationNetwork
from cs.bl import BottomLevel
from ls import LearningSystem
import numpy as np


def get_metrics():
    bl: BottomLevel = torch.load("results\\bottom_level_224_224_8_16_pneumonia.pt")
    print(bl.segmenter.prev_training)
    metrics = bl.segmenter.prev_training.numpy()
    np.savetxt("segmenter_metrics.txt", metrics, "%f")
    metrics = bl.classifier.prev_training.numpy()
    np.savetxt("classifier_metrics.txt", metrics, "%f")


def training_progression():
    setup()

    i_set = CustomDataset("K:\\Dataset\\pneumonia", [224, 224])
    i_loader = DataLoader(i_set, batch_size=32, shuffle=True, pin_memory=True)

    im_set = ImageMaskDataset("K:\\Dataset\\masks", [224, 224])
    im_loader = DataLoader(im_set, batch_size=32, shuffle=True, pin_memory=True)

    bl = BottomLevel(len(i_set.class_names), True, "resnet50", True)
    ls = LearningSystem(bl.classifier, bl.segmenter)

    bl.fit((im_loader, 8), (i_loader, 16))

    torch.save(bl, "results\\bottom_level_224_224_8_16_pneumonia.pt")


def segmenter_test_from_file():
    filename = "results\\segmenter_480_480_32.pt"

    segmenter: SegmentationNetwork = torch.load(filename)

    im_set = ImageMaskDataset("K:\\Dataset\\masks", [512, 512])
    im_loader = DataLoader(im_set, batch_size=16, shuffle=True, pin_memory=True)

    segmenter.fit(im_loader, 1)

    torch.save(segmenter, "results\\segmenter_512_512_32.pt")


def segmenter_test():
    segmenter = SegmentationNetwork()

    segmenter.optim = torch.optim.Adam(segmenter.model.parameters(), lr=1e-4)
    segmenter.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(segmenter.optim, "min", patience=4, verbose=True)

    im_set = ImageMaskDataset("K:\\Dataset\\masks", [224, 224])
    im_loader = DataLoader(im_set, batch_size=32, shuffle=True, pin_memory=True)

    segmenter.fit(im_loader, 8)

    torch.save(segmenter, "results\\segmenter_test.pt")


def classifier_test():
    i_set = CustomDataset("K:\\Dataset\\pneumonia", [224, 224])
    i_loader = DataLoader(i_set, batch_size=32, shuffle=True, pin_memory=True)

    classifier = ClassificationNetwork(len(i_set.class_names), "resnet50")
    segmenter: SegmentationNetwork = torch.load("results\\segmenter_224_224_32.pt")
    classifier.segmenter = segmenter

    classifier.fit(i_loader, 4)

    torch.save(classifier, "results\\classifier_224_224_16_pneumonia.pt")


def classifier_test_from_model():
    i_set = CustomDataset("K:\\Dataset\\cxr", [480, 480])
    i_loader = DataLoader(i_set, batch_size=32, shuffle=True, pin_memory=True)
    classifier: ClassificationNetwork = torch.load("results\\classifier_480_480_32_pneumonia.pt")
    segmenter: SegmentationNetwork = torch.load("results\\segmenter_480_480_32.pt")
    classifier.segmenter = segmenter

    classifier.optim = torch.optim.SGD(classifier.model.parameters(), lr=1e-4, momentum=0.9)
    classifier.scheduler = torch.optim.lr_scheduler.StepLR(classifier.optim, step_size=8, gamma=0.1, verbose=True)

    classifier.model.num_classes = len(i_set.class_names)

    classifier.fit(i_loader, 16)

    torch.save(classifier, "results\\classifier_480_480_16_nih.pt")


def classifier_without_segmenter_from_model():
    i_set = CustomDataset("K:\\Dataset\\pneumonia", [512, 512])
    i_loader = DataLoader(i_set, batch_size=32, shuffle=True, pin_memory=True)

    filename = "results\\classifier_no_seg_480_480.pt"

    classifier: ClassificationNetwork = torch.load(filename)

    classifier.fit(i_loader, 2)

    torch.save(classifier, "results\\classifier_no_seg_512_512.pt")


def classifier_without_segmenter():
    i_set = CustomDataset("K:\\Dataset\\pneumonia", [480, 480])
    i_loader = DataLoader(i_set, batch_size=32, shuffle=True, pin_memory=True)

    classifier = ClassificationNetwork(len(i_set.class_names), "resnet50")

    classifier.fit(i_loader, 32)

    torch.save(classifier, "results\\classifier_no_seg_480_480.pt")


def df_from_model():
    from cs.tl.induction import DecisionForest
    from cs.bl import BottomLevel

    i_set = CustomDataset("K:\\Dataset\\pneumonia", [224, 224])
    i_loader = DataLoader(i_set, batch_size=32, shuffle=True, pin_memory=True)

    segmenter: SegmentationNetwork = torch.load("results\\segmenter_224_224_32.pt")
    classifier: ClassificationNetwork = torch.load("results\\classifier_224_224_32_pneumonia.pt")

    bl = BottomLevel(3, True, "resnet50", True)
    bl.segmenter = segmenter
    bl.classifier = classifier
    classifier.segmenter = segmenter
    with torch.no_grad():
        X, y = bl(i_loader)

    df = DecisionForest(8, None)
    df.fit(X, y)
    torch.save(df, "results\\forest_224_224_32_16_pneumonia_8mss.pt")


if __name__ == "__main__":
    get_metrics()
