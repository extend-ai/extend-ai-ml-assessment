# %%
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import cv2


transform = transforms.Compose(
    [
        transforms.Resize(2592, 1952),
        transforms.ToTensor(),
    ]
)

dataset = datasets.ImageFolder("data", transform=transform)


# %%
