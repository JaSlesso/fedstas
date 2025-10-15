import torch
import torch.nn as nn
import torch.nn.functional as F

class FastCIFAR10CNN_GN(nn.Module):
    def __init__(self, num_classes=10, groups=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(groups, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(groups, 32)
        self.pool1 = nn.MaxPool2d(2, 2)               # 32 -> 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn3   = nn.GroupNorm(groups, 64)
        self.pool2 = nn.MaxPool2d(2, 2)               # 16 -> 8
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.gn3(self.conv3(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FastCIFAR10CNN(nn.Module):
    """Ultra-fast lightweight CNN model for CIFAR-10 (32x32x3 -> 10 classes)"""
    
    def __init__(self, num_classes=10):
        super(FastCIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 32x32x3 -> 32x32x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 32x32x16 -> 32x32x32
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16x32 -> 16x16x64
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


class SimpleCIFAR10CNN(nn.Module):
    """Simple CNN model for CIFAR-10 (32x32x3 -> 10 classes)"""
    
    def __init__(self, num_classes=10):
        super(SimpleCIFAR10CNN, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # First block
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        
        # Second block
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        
        # Third block
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18CIFAR10(nn.Module):
    """ResNet-18 model adapted for CIFAR-10"""
    
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR10, self).__init__()
        self.in_planes = 64
        
        # Initial convolution - smaller kernel for CIFAR-10's 32x32 images
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def create_model(model_type="fast_cnn", num_classes=10):
    """
    Factory function to create CIFAR-10 models.
    
    Args:
        model_type (str): Either "fast_cnn", "simple_cnn", or "resnet18"
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
    
    Returns:
        torch.nn.Module: The requested model
    """
    if model_type.lower() == "fast_cnn":
        return FastCIFAR10CNN(num_classes=num_classes)
    elif model_type.lower() == "simple_cnn":
        return SimpleCIFAR10CNN(num_classes=num_classes)
    elif model_type.lower() == "resnet18":
        return ResNet18CIFAR10(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'fast_cnn', 'simple_cnn', or 'resnet18'")


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test all models
    print("Testing CIFAR-10 models...")
    
    # Test input (batch_size=1, channels=3, height=32, width=32)
    test_input = torch.randn(1, 3, 32, 32)
    
    # Test Fast CNN
    fast_model = create_model("fast_cnn")
    fast_output = fast_model(test_input)
    print(f"Fast CNN - Parameters: {count_parameters(fast_model):,}")
    print(f"Fast CNN - Output shape: {fast_output.shape}")
    
    # Test Simple CNN
    simple_model = create_model("simple_cnn")
    simple_output = simple_model(test_input)
    print(f"Simple CNN - Parameters: {count_parameters(simple_model):,}")
    print(f"Simple CNN - Output shape: {simple_output.shape}")
    
    # Test ResNet-18
    resnet_model = create_model("resnet18")
    resnet_output = resnet_model(test_input)
    print(f"ResNet-18 - Parameters: {count_parameters(resnet_model):,}")
    print(f"ResNet-18 - Output shape: {resnet_output.shape}") 
