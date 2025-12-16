#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=20:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/autoencoder/
#$ -N Test_VAE_grey
#$ -t 1-14

date
hostname
nvidia-smi

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

# ---- map array index -> one command, many GPUs ----
CMD_LIST=(
  "python3 test.py --batch_size 5 --down 1 --save_name 'ADC_pretrained'"
  "python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_100'  --save_name 'ADC_grey_100'  --greyscale"
  "python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_150'  --save_name 'ADC_grey_150'  --greyscale"
  "python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_200'  --save_name 'ADC_grey_200'  --greyscale"
  "python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_250'  --save_name 'ADC_grey_250'  --greyscale"
  "python3 test.py --batch_size 5 --down 1 --checkpoint 'ADC_VAE_grey/vae_best' --save_name 'ADC_grey_best' --greyscale"

  "python3 test.py --batch_size 5 --use_T2W --save_name 'T2W_pretrained'"
  "python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_50'  --save_name 'T2W_grey_50'  --greyscale"
  "python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_100' --save_name 'T2W_grey_100' --greyscale"
  "python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_150' --save_name 'T2W_grey_150' --greyscale"
  "python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_200' --save_name 'T2W_grey_200' --greyscale"
  "python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_250' --save_name 'T2W_grey_250' --greyscale"
  "python3 test.py --batch_size 5 --use_T2W --checkpoint 'T2W_VAE_grey/vae_best' --save_name 'T2W_grey_best' --greyscale"
)

IDX=$((SGE_TASK_ID - 1))
CMD="${CMD_LIST[$IDX]}"

echo "SGE_TASK_ID=${SGE_TASK_ID}"
echo "Running: ${CMD}"
eval "${CMD}"

date
