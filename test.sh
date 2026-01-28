#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=5:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Test_diffusion
#$ -t 1-11

date
hostname
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

CMD_LIST=(
  ### Basic ###
  "cd diffusion_basic && python3 test.py --checkpoint './lowfield/model-best.pt' --lowfield"
  "cd diffusion_basic && python3 test.py --checkpoint './lowfield_perct/model-best.pt' --lowfield"
  "cd diffusion_basic && python3 test.py --checkpoint './lowfield_concat/model-best.pt' --lowfield --use_T2W "
  "cd diffusion_basic && python3 test.py --checkpoint './lowfield_concat2/model-best.pt' --lowfield --use_T2W "

  ### Attention ###
  "cd diffusion_attention && python3 test.py --checkpoint './lowfield/model-best.pt' --lowfield --use_T2W"

  ### Latent ###
  "cd diffusion_latent && python3 test.py --checkpoint './lowfield/model-best.pt' --lowfield "
  "cd diffusion_latent && python3 test.py --checkpoint './lowfield_recon/model-best.pt' --lowfield "
  "cd diffusion_latent && python3 test.py --checkpoint './lowfield_recon_prednoise/model-best.pt'--lowfield  "

  "cd diffusion_latent && python3 test.py --checkpoint './lowfield_t2w/model-best.pt' --lowfield --use_T2W"
  "cd diffusion_latent && python3 test.py --checkpoint './lowfield_t2w_recon/model-best.pt' --lowfield --use_T2W"
  "cd diffusion_latent && python3 test.py --checkpoint './lowfield_t2w_recon_prednoise/model-best.pt' --lowfield --use_T2W"

  ### Autoencoder ###
  # "cd autoencoder && python3 test.py --batch_size 5 --down 1 --save_name 'ADC_pretrained'"
  # "cd autoencoder && python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_100'  --save_name 'ADC_grey_100'  --greyscale"
  
  # "cd autoencoder && python3 test.py --batch_size 5 --use_T2W --save_name 'T2W_pretrained'"
  # "cd autoencoder && python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_50'  --save_name 'T2W_grey_50'  --greyscale"
)

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date
