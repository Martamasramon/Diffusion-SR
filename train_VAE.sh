#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/autoencoder/
#$ -N Train_VAE_ADC

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

python3 train.py \
    --results_folder './ADC_VAE_grey' \
    --lr 0.00001 \
    --n_epochs 300 \
    --batch_size 32 \
    --save_every 50 \
    --down 1

date