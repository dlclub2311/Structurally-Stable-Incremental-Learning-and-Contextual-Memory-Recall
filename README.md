## A Method and a Metric for CIL: Structurally Stable Incremental Learning (S²IL) and Contextual Memory Recall (CMR)

#### 1. Code Dependencies
To install the required packages and to switch to the new environment: 
```bash
conda env create --file environment.yaml && conda activate s2il

```

#### 2. Data Setup

1. Cifar-100 dataset is automatically downloaded by the code and the required data setup is done by the code.
2. ImageNet-100 and Imagenet-1K datasets have to be downloaded and organized according to the paths given in the respective files in *imagenet_split* folder.

#### 3. Experiments
To reproduce the results presented in Table1 for the **INC10** setting on **CIFAR-100** with three different class orders:

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> --label S2IL_cnn_cifar100_INC10 \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_cifar100_INC10.txt \
    --ssim-p 0.1 --ssim-q 8.0  --ssim-r 8.0
```

To reproduce the results presented in Table2 for the **INC10** setting on **ImageNet-100**:

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_imagenet100.yaml options/data/imagenet100_1order.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> --label S2IL_cnn_Imagenet100_INC10 \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_Imagenet100_INC10.txt \
    --ssim-p 0.1 --ssim-q 8.0  --ssim-r 8.0
```

To reproduce the results presented in Table2 for the **INC100** setting on **ImageNet-1K**:

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_imagenet1000.yaml options/data/imagenet1000_1order.yaml \
    --initial-increment 500 --increment 100 --memory 20000 \
    --device <GPU_ID> --label S2IL_cnn_Imagenet1000_INC100 \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_Imagenet1000_INC100.txt \
    --ssim-p 0.1 --ssim-q 8.0  --ssim-r 8.0
```

To reproduce CPCMR value in Table5 for the **INC10** setting on **CIFAR100** with three different class orders:

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> --label S2IL_cnn_cifar100_INC10_CPCMR \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_cifar100_INC10_CPCMR.txt \
    --ssim-p 0.1 --ssim-q 8.0  --ssim-r 8.0 --hint-replace-prob 0.1 --calc-hint --save-model task
```

#### 4. Results
The results are saved in the specified log files(see the --log-file option), and detailed class-wise results can be found in the results folder, which is generated when the code is run.


## Acknowledgements

This repository is developed based on [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch).
