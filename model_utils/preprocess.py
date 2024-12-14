from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# data transformations + augmentation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_data_dir = os.path.join(os.getcwd(), 'data', 'train')
test_data_dir = os.path.join(os.getcwd(), 'data', 'test')

# train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)

# def get_train_loader(train_batch_size):
#     train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
#     return train_loader

def get_test_loader( test_batch_size ):
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return test_loader

# class_names = train_dataset.classes
# print(f"Classes: {class_names}")
