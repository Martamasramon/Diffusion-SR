#$ -l tmem=200G,h_vmem=200G
#$ -l gpu=true
#$ -l h_rt=50:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Train_diffusion
#$ -t 1-9

date

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

CMD_LIST=(
    ### Basic ###
    # "cd diffusion_basic && python3 train.py --results_folder './lowfield_perct' "

      ### T2W/HBV ###
      # "cd diffusion_basic && python3 train.py --results_folder './lowfield_concat_t2w'     --use_T2W "
      "cd diffusion_basic && python3 train.py --results_folder './lowfield_concat_t2w_hbv' --use_T2W --use_HBV"
      "cd diffusion_basic && python3 train.py --results_folder './lowfield_concat_hbv'     --use_HBV "
      
      ### T2W Offset ###
      # "cd diffusion_basic && python3 train.py --results_folder './concat_down8_offset20' --use_T2W --down 8 --t2w_offset 20"
      
      ### Controlnet ###
      # "cd diffusion_basic && python3 train.py --results_folder './test_controlnet' --controlnet --down 8 --checkpoint ./pretrain_down8/model-best.pt"
      
      ### SR beyond target ###
      # "cd diffusion_basic && python3 train.py --results_folder './test_x4' --use_T2W --upsample --down 4 --batch_size 8"
      
      ### 3D ###
      # "cd diffusion_basic && python3 train3d.py --results_folder './pretrain_3d'"
    
    ### Attention ###
    # "cd diffusion_attention && python3 train.py --results_folder './lowfield' --use_T2W "

    ### Latent ###
    # "cd diffusion_latent && python3 train.py --recon_mse 0.01 --results_folder './lowfield_recon' "
    # "cd diffusion_latent && python3 train.py --recon_mse 0.01 --results_folder './lowfield_t2w_recon' --use_T2W "
    # "cd diffusion_latent && python3 train.py --recon_mse 0.01 --objective 'pred_noise' --results_folder './lowfield_recon_prednoise' "
    # "cd diffusion_latent && python3 train.py --recon_mse 0.01 --objective 'pred_noise' --results_folder './lowfield_t2w_recon_prednoise' --use_T2W "

    ### Autoencoder ###
    # "cd autoencoder &&  python3 train.py --results_folder './ADC' --lr 0.00000001 ---n_epochs 100 --batch_size 64 --save_every 50 --down 1 --greyscale"

      ### T2W ###
      # "cd autoencoder &&  python3 train.py --results_folder './T2W' --lr 0.00000001 ---n_epochs 100 --batch_size 64 --save_every 50 --use_T2W --greyscale"

    ### Finetune ###
    # "cd diffusion_basic && python3 train.py --results_folder './finetune' --finetune './pretrain/model-8.pt' --lr 0.000001 --data_folder 'HistoMRI' --n_epochs 10000 "

    ## DisC-Diff ##
    # "cd diffusion_basic && python3 train.py --results_folder './lowfield_discdiff_basic' "
    # "cd diffusion_basic && python3 train.py --results_folder './lowfield_discdiff_multi' --unet_type 'disc_diff' --use_T2W "
    "cd diffusion_basic && python3 train.py --results_folder './lowfield_discdiff_hbv'     --unet_type 'disc_diff' --use_HBV "
    "cd diffusion_basic && python3 train.py --results_folder './lowfield_discdiff_t2w_hbv' --unet_type 'disc_diff' --use_T2W --use_HBV "


    ### Multi-task ###
    "cd diffusion_multitask && python3 train.py --results_folder './hbv'                  --use_T2W --unet_type 'multitask' --batch_size 4 --use_HBV"
    "cd diffusion_multitask && python3 train.py --results_folder './dropout'              --use_T2W --unet_type 'multitask' --batch_size 4 --modality_drop_prob 0.1"
    "cd diffusion_multitask && python3 train.py --results_folder './no_attn'              --use_T2W --unet_type 'multitask' --batch_size 4 --no_cross_attn"
    "cd diffusion_multitask && python3 train.py --results_folder './uncorrelated_noise'   --use_T2W --unet_type 'multitask' --batch_size 4 --noise_rho 0 "
    "cd diffusion_multitask && python3 train.py --results_folder './uncorrelated_no_attn' --use_T2W --unet_type 'multitask' --batch_size 4 --no_cross_attn --noise_rho 0"
  )

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date