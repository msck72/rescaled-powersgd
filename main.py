import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

from powersgd import optimizer_step
from powersgd.ef_at_last_powersgd import Aggregator, PowerSGD, Config
from powersgd.utils import params_in_optimizer



ROUND = 0

def test(model, test_loader):

    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for source, targets in test_loader:
        
            y_hat = model(source)
            loss = F.cross_entropy(y_hat, targets)

            test_loss += loss.item()

            # Compute accuracy (assuming classification)
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")



# initialize a random model
# use the predefined model in torch
def main():
    torch.manual_seed(42)
    model = resnet18()

    # use a dataset, say cifar10
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set, test_set = CIFAR10(root='./data', download=True, train=True, transform=transform), CIFAR10(root='./data', download=True, train=False, transform=transform)
    train_loader, test_loader = DataLoader(train_set, shuffle=True, batch_size=64), DataLoader(test_set, shuffle=False, batch_size=64)

    # define a loss function
    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # 1 Estimating gradient using powerSGD

    config = Config(rank=1, num_iters_per_step=1, start_compressing_after_num_steps=1)
    params = params_in_optimizer(optimizer)
    compressor = PowerSGD(params, config=config)
    grads_powersgd, grads_rescaled_powersgd = None, None


    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()

        # compress the gradeint using different techniques
        params = params_in_optimizer(optimizer)
        grads = [p.grad.data for p in params]
        
        if batch == 0:
            grads_powersgd = [torch.zeros_like(p.grad.data) for p in params]
            grads_rescaled_powersgd = [torch.zeros_like(p.grad.data) for p in params]

        for g, gp, grp in zip(grads, grads_powersgd, grads_rescaled_powersgd):
            gp.copy_(g)
            grp.copy_(g)

        if batch > 50:
            powersgd_gradient_estimate = powersgd_compressor(grads_powersgd, compressor)
            rescaled_powersgd_gradient_estimate = rescaled_powersgd_compressor(grads_rescaled_powersgd, compressor)
        
            # # compare the quality of compression
            quality_of_compression(grads, powersgd_gradient_estimate, rescaled_powersgd_gradient_estimate)

        # optimizer_step(optimizer, compressor, 'aggregate_using_rescaled_grads')
        # optimizer_step(optimizer, compressor, 'aggregate')
        
        # print("PowerSGD gradient estimate shapes")
        # for pge in powersgd_gradient_estimate:
        #     print(pge.shape)
            
        # print("\nrescaled powersgd gradient estimate")
        # for rpge in rescaled_powersgd_gradient_estimate:
        #     print(rpge.shape)
        # print()
        
        print('one step done\n\n')
        
        if batch > 70:
            break
        # break
    # test(model, test_loader)

    

    

def create_params_like_zeros(optimizer):
    
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            params.append(torch.zeros_like(p))
            
    return params

def powersgd_compressor(grads, compressor: Aggregator):
    return compressor.aggregate(grads)

# a Estimating gradient using smaller low-rank matrices
def rescaled_powersgd_compressor(grads, compressor: Aggregator):

    return compressor.aggregate_using_rescaled_grads(grads)

# Observe the difference
def quality_of_compression(actual_grad, powersgd_grad_est, rescaled_powersgd_grad_est):
    
    global ROUND
    # compare ho good the compression is in both the cases.
    with open(f'powersgds_power_{ROUND}.txt', 'w') as f:
        powersgd_power_test = 0
        for ag, pg in zip(actual_grad, powersgd_grad_est):
            temp = torch.norm(ag - pg)
            if temp != 0 and ROUND < 10:
                f.write(f'{ag.shape} {torch.numel(ag)} {float(temp.numpy())}\n')
            powersgd_power_test += temp
    
    with open(f'rescaled_powersgds_power_{ROUND}.txt', 'w') as f:
        rescaled_powersgd_power_test = 0
        for ag, rpg in zip(actual_grad, rescaled_powersgd_grad_est):
            temp = torch.norm(ag - rpg)
            if temp != 0 and ROUND < 10:
                f.write(f'{ag.shape} {torch.numel(ag)} {float(temp.numpy())}\n')
            rescaled_powersgd_power_test += temp
        
    print(f"powersgd power test = {powersgd_power_test}")
    print(f"rescaled powersgd power test = {rescaled_powersgd_power_test}")
    
    ROUND += 1
    
    return

if __name__ == '__main__':
    main()