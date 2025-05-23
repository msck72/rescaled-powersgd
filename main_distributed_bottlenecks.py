import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet18, vgg16, resnet50
from torch.utils.data import DataLoader

from powersgd import optimizer_step, optimizer_step_naive_powersgd
from powersgd.powersgd_last_layer_magics import Aggregator, PowerSGD, Config
from powersgd.utils import params_in_optimizer

from load_dataset import CIFAR10_load_dataloaders_distributedly
import matplotlib.pyplot as plt


MODEL_NAME = ''
# THRESHOLD_STANDARD_DEVIATION = 0.1


def main():
    torch.manual_seed(42)
    # define a model
    global MODEL_NAME
    
    EPOCHS = 5
    
    model = resnet18()
    # model = vgg16()
    # model = resnet50()
    model_name = 'ResNet18'
    MODEL_NAME = model_name
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # 10 classes for CIFAR-10

    train_loader, test_loader = CIFAR10_load_dataloaders_distributedly(batch_size=64)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    config = Config(rank=2, num_iters_per_step=1, start_compressing_after_num_steps=0)
    params = params_in_optimizer(optimizer)
    compressor = PowerSGD(params, config=config)

    accuracies = []

    for epoch in range(EPOCHS):
        for batch, (x, y) in enumerate(train_loader):
            # optimizer.zero_grad()
            if batch % 10 == 0 and torch.distributed.get_rank() == 0:
                print(f'batch {batch}')

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            # optimizer.step()

            optimizer_step(optimizer, compressor, 'aggregate')
    
            if batch != 0 and batch % 100 == 0:
                print(f'Testing after {batch} rounds')
                accuracies.append((epoch * len(train_loader) + batch, test(model, test_loader)))
    
        accuracies.append(((epoch + 1) * len(train_loader), test(model, test_loader)))
    
    print(config)
    print(f'accuracies = {accuracies}')
    
    plt.plot([i * 100 for i in range(len(accuracies))], [a for _, a in accuracies])
    plt.xlabel('Number of completed batches')
    plt.ylabel('Accuracy')
    plt.title('Improvement in Accuracy over the batches')
    plt.xticks()
    plt.show()
        


def test(model: nn.Module, test_loader: DataLoader):

    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for source, targets in test_loader:
        
            y_hat = model(source)
            loss = F.cross_entropy(y_hat, targets)

            test_loss += loss.item()

            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':
    torch.distributed.init_process_group()
    main()
    torch.distributed.destroy_process_group()