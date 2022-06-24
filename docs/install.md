## Installation

This process was executed successfully on Ubuntu 20.04 with CUDA driver of version 11.1. Please ensure your GPU memory is more than `24G` if you want to do all evaluations.

## Dependency

We assume that you have installed anaconda and created a blank python environment (python3.8).

#### Dependency Installation

* Pytorch

  * Pyorch supports sparse matrix multiplication after`1.8.0`, so please make sure your torch version is above 1.8, such as
    ```
    pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

  In this procedure, `${TORCH}` is  `1.9.1`, `${CUDA}` is `11.1`.
* Pytorch-geometric

  * Please run the following instructions.
    * `Pytorch geometric`
      ```bash
      pip install torch-geometric -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
      ```
    * `Pytorch scatter`
    * ```bash
      pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
      ```
    * `Pytorch cluster`
    * ```bash
      pip install torch-cluster  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
      ```
    * `Pytorch sparse`
    * ```bash
      `pip install torch-cluster  -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`
      ```
* Others

  * Other dependencies can be installed by running:
  * ```bash
    pip install -r requirements.txt
    ```
