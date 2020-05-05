# Run and evaluate MultiSWAG
DATAPATH=~/datasets
DATASET=CIFAR100
MODEL=VGG16
EPOCHS=300
BASEDIR=ckpts/example
SWAG_RUNS=3

LR=0.05
WD=5e-4
SWAG_START=161
SWAG_LR=0.01
SWAG_SAMPLES=20

echo ${WD}

CKPT_FILES=""

for (( seed=1; seed <=$SWAG_RUNS; seed++ ))
do
    python experiments/train/run_swag.py --data_path=$DATAPATH --epochs=$EPOCHS --dataset=$DATASET --save_freq=$EPOCHS \
      --model=$MODEL --lr_init=${LR} --wd=${WD} --swag --swag_start=$SWAG_START --swag_lr=${SWAG_LR} --cov_mat --use_test \
      --dir=${BASEDIR}/swag_${seed} --seed=$seed

    CKPT_FILES+=" "${BASEDIR}/swag_${seed}/swag-${EPOCHS}.pt
done


python3 experiments/train/eval_multiswag.py --data_path=$DATAPATH --dataset=$DATASET --model=$MODEL --use_test --swag_ckpts \
  --swag_samples=$SWAG_SAMPLES --swag_ckpts${CKPT_FILES}  --savedir=$BASEDIR/multiswag/
