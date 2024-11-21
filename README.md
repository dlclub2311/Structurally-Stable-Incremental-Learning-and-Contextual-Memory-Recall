## A Method and a Metric for CIL: Structurally Stable Incremental Learning (S2IL) and Contextual Memory Recall (CMR)

#### 1. Code Dependencies
Install required packages and switching to new environment: 
```bash
conda env create --file environment.yaml && conda activate s2il

```

#### 2. Experiments
To reproduce Table1 with 5 steps on CIFAR100 with three different class orders:

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> --label S2IL_cnn_cifar100_INC10 \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_cifar100_INC10.txt \
    --alpha 0.1 --beta 8.0  --gamma 8.0
```

To reproduce CPCMR value in Table5 with 5 steps on CIFAR100 with three different class orders:

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> --label S2IL_cnn_cifar100_INC10_CPCMR \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_cifar100_INC10_CPCMR.txt \
    --alpha 0.1 --beta 8.0  --gamma 8.0 --hint-replace-prob 0.1 --calc-hint --save-model task
```


For ImageNet100 (Table 2):

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_imagenet100.yaml options/data/imagenet100_1order.yaml \
    --initial-increment 50 --increment 10 --memory 2000 \
    --device <GPU_ID> --label S2IL_cnn_Imagenet100_INC10 \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_Imagenet100_INC10.txt \
    --alpha 0.1 --beta 8.0  --gamma 8.0
```

And for ImageNet1K (Table 2):

```bash
python3 -minclearn --options options/S2IL/S2IL_cnn_imagenet1000.yaml options/data/imagenet1000_1order.yaml \
    --initial-increment 500 --increment 100 --memory 20000 \
    --device <GPU_ID> --label S2IL_cnn_Imagenet1000_INC100 \
    --data-path <PATH/TO/DATA> --log-file S2IL_cnn_Imagenet1000_INC100.txt \
    --alpha 0.1 --beta 8.0  --gamma 8.0
```

## Acknowledgements

This repository is developed based on [PODNet](https://github.com/arthurdouillard/incremental_learning.pytorch)


