#$ -l tmem=128G,h_vmem=128G
#$ -l gpu=true
#$ -l h_rt=25:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Train_diffusion
#$ -t 1-5

date
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

CMD_LIST=(
    ### Basic ###
    "cd diffusion_basic && python3 train.py --results_folder './lowfield_perct' --lowfield"

      ### T2W ###
      "cd diffusion_basic && python3 train.py --results_folder './lowfield_concat' --use_T2W --lowfield"
      
      ### T2W Offset ###
      # "cd diffusion_basic && python3 train.py --results_folder './concat_down8_offset20' --use_T2W --down 8 --t2w_offset 20"
      
      ### Controlnet ###
      # "cd diffusion_basic && python3 train.py --results_folder './test_controlnet' --controlnet --down 8 --checkpoint ./pretrain_down8/model-best.pt"
      
      ### SR beyond target ###
      # "cd diffusion_basic && python3 train.py --results_folder './test_x4' --use_T2W --upsample --down 4 --batch_size 8"
      
      ### 3D ###
      # "cd diffusion_basic && python3 train3d.py --results_folder './pretrain_3d'"
    
    ### Attention ###
    "cd diffusion_attention && python3 train.py --results_folder './lowfield' --use_T2W --lowfield"

    ### Latent ###
    "cd diffusion_latent && python3 train.py --results_folder './lowfield' --lowfield"
    "cd diffusion_latent && python3 train.py --results_folder './lowfield_t2w' --use_T2W --lowfield"

    ### Autoencoder ###
    # "cd autoencoder &&  python3 train.py --results_folder './ADC' --lr 0.00000001 ---n_epochs 100 --batch_size 64 --save_every 50 --down 1 --greyscale"

      ### T2W ###
      # "cd autoencoder &&  python3 train.py --results_folder './T2W' --lr 0.00000001 ---n_epochs 100 --batch_size 64 --save_every 50 --use_T2W --greyscale"

    ### Finetune ###
    # "cd diffusion_basic && python3 train.py --results_folder './finetune' --finetune './pretrain/model-8.pt' --lr 0.000001 --data_folder 'HistoMRI' --n_epochs 10000 "

  )

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date