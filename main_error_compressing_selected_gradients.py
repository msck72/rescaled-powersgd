import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models import resnet18, vgg16, resnet50
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

from powersgd import optimizer_step
from powersgd.powersgd_shape_manipulations import Aggregator, PowerSGD, Config
from powersgd.utils import params_in_optimizer

import os
import matplotlib.pyplot as plt


ROUND, MAX_ROUND = 0, 20
MODEL_NAME = ''
SAVE_GRADIENT_AT = MAX_ROUND
THRESHOLD_STANDARD_DEVIATION = 0.1


def main():
    torch.manual_seed(42)
    # define a model
    global MODEL_NAME, SAVE_GRADIENT_AT
    
    
    model = resnet18()
    # model = vgg16()
    # model = resnet50()
    model_name = 'ResNet18'
    MODEL_NAME = model_name

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set, test_set = CIFAR10(root='./data', download=True, train=True, transform=transform), CIFAR10(root='./data', download=True, train=False, transform=transform)
    train_loader, test_loader = DataLoader(train_set, shuffle=True, batch_size=64), DataLoader(test_set, shuffle=False, batch_size=64)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    config = Config(rank=2, num_iters_per_step=1, start_compressing_after_num_steps=1)
    params = params_in_optimizer(optimizer)
    compressor = PowerSGD(params, config=config)
    grads_powersgd, grads_rescaled_powersgd = None, None

    is_compressed_mask = compressor._compressed_gradients_layers_num()
    powersgd_approx, rescaled_powersgd_approx = [], []
    
    gradients_std, powersgd_layer_wise_error = [], []
    num_grads_logically_comm, num_grads_comm = 0, 0

    # iterate through each mini-batch and compute the gradient and estimate the gradient
    for batch, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # forward-pass
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

        powersgd_gradient_estimate = powersgd_compressor(grads_powersgd, compressor)
        rescaled_powersgd_gradient_estimate = varing_compression_rate_powersgd_compressor(grads_rescaled_powersgd, compressor)
        (num_grads_logically_comm, num_grads_comm) = compressor.get_grads_comm()
    
        # compare the quality of compression
        powersgd_quality, rescaled_powersgd_quality = quality_of_compression(grads, powersgd_gradient_estimate, rescaled_powersgd_gradient_estimate, is_compressed_mask)
        
        powersgd_approx.append(powersgd_quality.item())
        rescaled_powersgd_approx.append(rescaled_powersgd_quality.item())

        # # save the gradient
        # if batch == SAVE_GRADIENT_AT:
        #     for i, (ag, pg) in enumerate(zip(grads, powersgd_gradient_estimate)):
        #         if is_compressed_mask[i]:
        #             torch.save(ag, f'./raw_gradients/{model_name}_layer_{i}.pt')
        #             torch.save(pg, f"./powersgd_gradients/{model_name}_layer_{i}.pt")
        #             # np.savetxt(f'./gradients/layer_{i}', ag.view(ag.shape[0], -1).numpy())
        
        # save the standard deviation of the gradient of each layer
        # if batch == SAVE_GRADIENT_AT:
        #     with open(f'std_gradient.txt', 'w') as std_file:
        #         for i, (ag, pg) in enumerate(zip(grads, powersgd_gradient_estimate)):
        #             if is_compressed_mask[i]:
        #                 std = torch.std(ag).item()
        #                 std_file.write(f'{str(std)}\n')
        #                 gradients_std.append(std)
        #                 # np.savetxt(f'./gradients/layer_{i}', ag.view(ag.shape[0], -1).numpy())
        
        i_gradients_std, i_powersgd_layer_wise_error = [], []
        for i, (ag, pg) in enumerate(zip(grads, powersgd_gradient_estimate)):
            if is_compressed_mask[i]:
                std = torch.std(ag).item()
                i_gradients_std.append(std)

        i_powersgd_layer_wise_error = layer_wise_error(grads, powersgd_gradient_estimate, is_compressed_mask)
        
        gradients_std.append(i_gradients_std)
        powersgd_layer_wise_error.append(i_powersgd_layer_wise_error)
        
        if batch == MAX_ROUND:
            break
        # break

    
    print(f'powersgd_approx: {powersgd_approx}')
    print(f'rescaled_powersgd_approx: {rescaled_powersgd_approx}')
    
    plt.plot(powersgd_approx, label='Entire gradient compression using PowerSGD', color='blue')
    plt.plot(rescaled_powersgd_approx, label='Selected gradient layers compression using PowerSGD', color='green')
    plt.title('Comparision of error while sharing different amounts of gradient')
    plt.xlabel('Batch number')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    plt.figure(figsize=(4, 4.8))
    plt.bar(['Total compression', 'Selected Compression'], [num_grads_comm, num_grads_logically_comm], color=['blue', 'green'])
    plt.title('Amount of Data communicated\n(sent for aggregation by each node)')
    plt.ylabel('Size of gradient communicated')
    plt.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
        
    

def create_params_like_zeros(optimizer):
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            params.append(torch.zeros_like(p))
            
    return params


def powersgd_compressor(grads, compressor: Aggregator):
    return compressor.aggregate(grads)


# a Estimating gradient using smaller low-rank matrices
def varing_compression_rate_powersgd_compressor(grads, compressor: Aggregator):
    return compressor.compress_and_aggregate_selected_grads(grads, THRESHOLD_STANDARD_DEVIATION)


# Observe the difference
def quality_of_compression(actual_grad, powersgd_grad_est, rescaled_powersgd_grad_est, is_compressed_mask):
    
    global ROUND
    global MODEL_NAME
    # compare how good the compression is in both the cases.
    if not os.path.exists('powersgd_quality'):
        os.makedirs('powersgd_quality')
        
    with open(f'powersgd_quality/{MODEL_NAME}_{ROUND}.txt', 'w') as f:
        powersgd_quality = 0
        for i, (ag, pg) in enumerate(zip(actual_grad, powersgd_grad_est)):
            temp = torch.norm(ag - pg)
            if is_compressed_mask[i]:
                f.write(f'{ag.shape} {torch.numel(ag)} {float(temp.numpy())}\n')
            
            # f.write(f'{ag.shape} {torch.numel(ag)} {float(temp.numpy())}\n')
            
            powersgd_quality += temp
    
    if not os.path.exists('rescaled_powersgd_quality'):
        os.makedirs('rescaled_powersgd_quality')
    
    with open(f'rescaled_powersgd_quality/{MODEL_NAME}_{ROUND}.txt', 'w') as f:
        rescaled_powersgd_quality = 0
        for i, (ag, rpg) in enumerate(zip(actual_grad, rescaled_powersgd_grad_est)):
            temp = torch.norm(ag - rpg)
            if is_compressed_mask[i]:
                f.write(f'{ag.shape} {torch.numel(ag)} {float(temp.numpy())}\n')
            
            # f.write(f'{ag.shape} {torch.numel(ag)} {float(temp.numpy())}\n')
            
            rescaled_powersgd_quality += temp
        
    print(f"powersgd power test = {powersgd_quality}")
    print(f"rescaled powersgd power test = {rescaled_powersgd_quality}\n\n")
    
    ROUND += 1
    
    return powersgd_quality, rescaled_powersgd_quality



def layer_wise_error(actual_grads, est_grads, is_compressed_mask):
    error = []
    for i, (ag, eg) in enumerate(zip(actual_grads, est_grads)):
        if is_compressed_mask[i]:
            error.append(float(torch.norm(ag - eg).numpy()))
            
    return error


if __name__ == '__main__':
    main()