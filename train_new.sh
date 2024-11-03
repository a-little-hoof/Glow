#! /bin/bash
#SBATCH -o log/test.out
#SBATCH --partition=GPUA800
#SBATCH --job-name=02
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --time 72:00:00
#SBATCH --mem 24G

cd ..
cd ..
cd final

python train_new.py --data_type AFHQDataset \
    --results_dir ../results/gt --input_shape 3 64 64 \
    --lr_post 1e-05 --batch_size 128 \
    --gpu 0 \
    --num_bits 8 \
    --num_iters 1000000 \
    --clip 1.0 \