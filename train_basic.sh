#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=40:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Pretrain_upsample
#$ -t 1-2

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

cd diffusion_basic

CMD_LIST=(
    ### Basic ###
    "python3 train.py --results_folder './lowfield' --lowfield"
    ### T2W ###
    "python3 train.py --results_folder './lowfield_concat' --use_T2W --lowfield"
    ### T2W Offset ###
    # "python3 train.py --results_folder './concat_down8_offset20' --use_T2W --down 8 --t2w_offset 20"
    ### Controlnet ###
    # "python3 train.py --results_folder './test_controlnet' --controlnet --down 8 --checkpoint ./pretrain_down8/model-best.pt"
    ### SR beyond target ###
    # "python3 train.py --results_folder './test_x4' --use_T2W --upsample --down 4 --batch_size 8"
    ### 3D ###
    # "python3 train3d.py --results_folder './pretrain_3d'"
  )

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date