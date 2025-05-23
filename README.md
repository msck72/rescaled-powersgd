# Selective compression using PowerSGD:

> Note: The code for PowerSGD is adopted from the PowerSGD github repo, available at https://github.com/epfml/powersgd

This repository contains the variants of the PowerSGD algorithm, by selectively varying the rank of low-rank approximation of each layer of the gradient. This also includes various batching methodologies of the gradient layers for efficient batched matrix multiplications.

This repository was developed to explore the effects of variable ranks for low-rank approximation of the gradient and reshaping the gradient layers (which changes the inherent properties of the matrix formed by the gradient) on the convergence.

## How to run
The description and the functionality of each main program is written at the beginning of the file.

After selecting a main file, run the following command to run the program in a standalone setup:
```bash
torchrun --standalone --nproc_per_node=p --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=ip_address:port_num selected_main_file
```

To run the program in a distributed setup (works for only main_distributed_bottlenecks.py):
```bash
torchrun --nnodes=n --nproc_per_node=p --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=ip_address:port_num selected_main_file
```

> Set the number of nodes and number of processes per node accordingly.

The varying rank and the layers associated with this varied rank can be modified/specified in the `powersgd/powersgd_last_layer_magics.py` program, also entire gradient chunk can be reshaped into multiple factors by modifying the `find_factors` function in the `utils.py`
