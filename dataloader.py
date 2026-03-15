from torchvision import datasets , transforms
import torchvision
from torch.utils.data import DataLoader , random_split
import zipfile

def get_data(path):
  with zipfile.ZipFile(path,"r") as f:
    f.extractall()
    print("Extract done")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()

def create_dataset(
    root):
  dataset = datasets.ImageFolder(
      root = root,
      transform = auto_transforms
  )
  train_size = int(0.8*len(dataset))
  test_size = len(dataset) - train_size
  train_data , test_data = random_split(
      dataset,[train_size,test_size]
  )
  class_names = dataset.classes
  return train_data , test_data , class_names,dataset

def create_dataloader(train_data, test_data):
    train_dataloader = DataLoader(
        train_data,
        batch_size = 32,
        shuffle = True
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size = 32,
        shuffle = False
    )

    return train_dataloader , test_dataloader
