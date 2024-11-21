## A Method and a Metric for CIL: Structurally Stable Incremental Learning (S2IL) and Contextual Memory Recall (CMR)

#### 1. Code Dependencies
To install the required packages and to switch to the new environment: 
```bash
conda env create --file environment.yaml && conda activate s2il

```

#### 2. Experiments
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