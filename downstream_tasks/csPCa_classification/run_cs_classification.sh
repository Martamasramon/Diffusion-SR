#$ -l tmem=64G,h_vmem=64G
#$ -l gpu=true
#$ -l h_rt=25:00:00

#$ -S /bin/bash
#$ -j y
#$ -V

#$ -wd /cluster/project7/ProsRegNet_CellCount/CriDiff/downstream_tasks/csPCa_classification/
#$ -N Train_classifier

# Generate timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Create a directory named after the timestamp
dir_name="outputs/$timestamp"
mkdir -p "$dir_name"

export PATH=/share/apps/python-3.9.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.9.5-shared/lib:$LD_LIBRARY_PATH

source /cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin/activate
export PATH="/cluster/project7/ProsRegNet_CellCount/CriDiff/CriDiff_env/bin:$PATH"

python ./train.py "$dir_name" "$timestamp"