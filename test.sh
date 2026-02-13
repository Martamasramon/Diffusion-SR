#$ -l tmem=150G,h_vmem=150G
#$ -l gpu=true
#$ -l h_rt=5:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Test_diffusion
#$ -t 1-2

date

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

CMD_LIST=(
  ### Basic ###
  # "cd diffusion_basic && python3 test.py --checkpoint './lowfield/model-best.pt'  --unet_type 'basic_old'"

  # "cd diffusion_basic && python3 test.py --checkpoint './lowfield_concat_t2w/model-20.pt'  --use_T2W "
  # "cd diffusion_basic && python3 test.py --checkpoint './lowfield_discdiff_basic/model-best.pt' "
  # "cd diffusion_basic && python3 test.py --checkpoint './lowfield_discdiff_t2w/model-best.pt' "

  ### Attention ###
  # "cd diffusion_attention && python3 test.py --checkpoint './lowfield/model-best.pt'  --use_T2W"

  ### Latent ###
  # "cd diffusion_latent && python3 test.py --checkpoint './lowfield_recon/model-best.pt' "
  # "cd diffusion_latent && python3 test.py --checkpoint './lowfield_t2w_recon/model-best.pt'  --use_T2W"

  ### Latent UQ (Added as a template, please modify the desired checkpoint path as needed) ###
  # "cd diffusion_latent && python3 test.py --checkpoint './lowfield_t2w_recon/model-best.pt'  --use_T2W --perform_uq --num_reruns 24"

  ### Autoencoder ###
  # "cd autoencoder && python3 test.py --batch_size 5 --down 1 --save_name 'ADC_pretrained'"
  # "cd autoencoder && python3 test.py --batch_size 5 --use_T2W --save_name 'T2W_pretrained'"

  ### Multi-stream DisC-Diff ###
  # "cd diffusion_basic && python3 test.py --checkpoint './lowfield_discdiff_multi/model-best.pt'  --use_T2W --unet_type 'disc_diff'"
  # "cd diffusion_basic && python3 test.py --checkpoint './lowfield_discdiff_multi_hbv/model-best.pt'  --use_T2W --use_HBV --unet_type 'disc_diff'"

  ### Multi-task ###
  "cd diffusion_multitask && python3 test.py --checkpoint './basic/model-best.pt'  --use_T2W --unet_type 'multitask'"  
  "cd diffusion_multitask && python3 test.py --checkpoint './basic/model-best.pt'  --use_T2W --unet_type 'multitask'"  
)

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date

# python3 test.py --checkpoint './lowfield/model-best.pt'
# Average MSE:  0.013098
# Average PSNR: 19.02
# Average SSIM: 0.4292