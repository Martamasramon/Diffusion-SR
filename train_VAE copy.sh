#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/autoencoder/
#$ -N Test_VAE

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

python3 test.py \
    --checkpoint '/ADC_VAE_test/vae_209' \
    --batch_size 15 \
    --down 1

python3 test.py \
    --checkpoint '/T2W_VAE_test/vae_488' \
    --batch_size 15 \
    --use_T2W

date