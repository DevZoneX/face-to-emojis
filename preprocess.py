from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define data transformations (with augmentation)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (since FER-2013 is grayscale)
    transforms.RandomHorizontalFlip(),            # Data augmentation
    transforms.Resize((48, 48)),                  # Resize images to 48x48 (as in FER-2013)
    transforms.ToTensor(),                        # Convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize pixel values
])

# Load the training and test datasets using ImageFolder
train_data_dir = 'data/train'
test_data_dir = 'data/test'

train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)

def get_train_loader(train_batch_size):
    # DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    return train_loader

def get_test_loader( test_batch_size ):
    # DataLoader for batching and shuffling
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return test_loader

class_names = train_dataset.classes
print(f"Classes: {class_names}")
