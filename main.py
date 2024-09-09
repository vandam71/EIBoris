import torch
import random
from Boris import Boris
from pprint import pprint
from copy import deepcopy
from utils import setup
from torch.utils.data import DataLoader, TensorDataset


def random_boris():
    # seed_everything()
    def generate_random(data: torch.Tensor):
        channel = random.randint(0, data.shape[0] - 1)
        row = random.randint(0, data.shape[1] - 1)
        col = random.randint(0, data.shape[2] - 1)
        data[channel, row, col] += random.uniform(-0.15, 0.15)
        return data

    SAMPLES = 8
    BATCH_SIZE = 2
    NUM_CLASSES = 3
    input_data = torch.randn(3, 224, 224)

    data_array = []

    for i in range(SAMPLES):
        input_data = generate_random(input_data)
        data_array.append(deepcopy(input_data))

    random.shuffle(data_array)

    x = torch.stack(data_array).float()

    # x = torch.randn(SAMPLES, 3, 224, 224).float()
    y_mask = torch.randint(0, 1, (SAMPLES, 1, 224, 224)).float()
    y_label = torch.randint(0, 1, (SAMPLES, NUM_CLASSES)).float()

    mask_loader = DataLoader(TensorDataset(x, y_mask), batch_size=BATCH_SIZE)
    image_loader = DataLoader(TensorDataset(x, y_label), batch_size=BATCH_SIZE)

    boris = Boris(num_classes=NUM_CLASSES)

    boris._cs.fit((mask_loader, 8), (image_loader, 32))

    for i, image in enumerate(x[:]):
        boris(image.unsqueeze(0))
        print(f"Iteration {i}")
        pprint(boris._ss.working_memory.memory)
        print("\n\n")

    boris._ss.long_term_memory.draw_ltm()


if __name__ == "__main__":
    setup()
    from ss import StorageSystem
    boris = Boris.from_config("./config.yaml")
    # boris = Boris.load("./trained_pneumonia_224_s8_c16_b32_4_None_wm7.boris")
    # boris._ss = StorageSystem(7)
    # data_loader = DataLoader(boris.class_data, batch_size=64, shuffle=False)
    # for batch_idx, (images, labels) in enumerate(data_loader):
    #     for i in range(len(images)):
    #         boris(images[i].unsqueeze(0))
    #         # pprint(boris._ss.working_memory.memory)
    #     boris._ss.long_term_memory.draw_ltm()
