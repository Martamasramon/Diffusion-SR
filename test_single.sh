#$ -l tmem=128G,h_vmem=128G
#$ -l gpu=true
#$ -l h_rt=25:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Test_diffusion

date

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

cd diffusion_multitask

python3 test.py --checkpoint './basic/model-best.pt'  --use_T2W --unet_type 'multitask'

date