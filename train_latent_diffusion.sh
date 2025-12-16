#$ -l tmem=40G,h_vmem=40G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/diffusion_latent/
#$ -N Train_latent

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

/cluster/project7/ProsRegNet_CellCount/CriDiff/monitor_resources.sh & 

python3 train.py --checkpoint './latent_prednoise/model-best.pt' --results_folder './latent_finetune_prednoise' --lr 0.000005 --n_epochs 10000 --recon_mse 0.01
python3 test.py --checkpoint './latent_finetune_prednoise/model-best.pt'

date