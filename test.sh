#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=5:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/
#$ -N Test_diffusion
#$ -t 1-14

date
hostname
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

CMD_LIST=(
  ### Basic ###
  "cd diffusion_basic && python3 test.py --checkpoint './pretrain/model-8.pt' "
  "cd diffusion_basic && python3 test.py --checkpoint './pretrain_mask/model-best.pt' --use_mask "
  "cd diffusion_basic && python3 test.py --checkpoint './test_controlnet/model-best.pt' --down 8 --controlnet"

  ### Attention ###
  "cd diffusion_attention && python3 test.py --checkpoint './concat_prob02/model-best.pt' --use_T2W"
  "cd diffusion_attention && python3 test.py --checkpoint './concat_down8_offset20/model-best.pt' --use_T2W --down 8 --t2w_offset 20"

  ### Autoencoder ###
  "cd autoencoder && python3 test.py --batch_size 5 --down 1 --save_name 'ADC_pretrained'"
  "cd autoencoder && python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_100'  --save_name 'ADC_grey_100'  --greyscale"
  
  "cd autoencoder && python3 test.py --batch_size 5 --use_T2W --save_name 'T2W_pretrained'"
  "cd autoencoder && python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_50'  --save_name 'T2W_grey_50'  --greyscale"
)

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date
