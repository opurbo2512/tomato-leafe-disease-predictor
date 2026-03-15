import torch
from dataloader import create_dataset,create_dataloader,get_data
from data_visuliazation import visulize
from train import train
from utils import save_model
from torch import nn
import torch
import torchvision

get_data("data.zip")
train_data , test_data ,class_names , dataset = create_dataset("data")
train_dataloader , test_dataloader = create_dataloader(train_data,test_data)
print(dataset.class_to_idx)
visulize(train_data,class_names)
"""
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=3,
                    bias=True))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

results = train(
    model,
    train_dataloader,
    test_dataloader,
    loss_fn,
    optimizer,
    5
)

save_model(
    model,
    "models",
    "model.pth"
)
"""