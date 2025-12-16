#!/usr/bin/bash
# HPC Cluster compute resource monitor script 
# E.Gair
# 09/04/2025
# Added GPU monitoring functionality
# Restricted GPU monitoring to user jobs/tasks
# Bug fix: get sge_shepherd pid for array tasks from gridengine
# Bug fix: removed incorrectly appended start time from logfile
# Bug fix for multiple processes single gpu slot
# Improved output formatting
# Make first command line parameter optional, set default monitor interval 1 min.
# Bug fix to GPU_MONITOR variable
# Bug fixes get_gpu_usage()
# Bug fixes get_pids()
# Bug fix gpu vram MiB to GiB conversion (include check for valid number)

#set -x

#-------------------------------------------------------------------
# Optional command line parameter: monitor interval in minutes
#-------------------------------------------------------------------
if [ -z "$1" ]; then
  echo "Sleep parameter not supplied, setting to default (1 min)"
  SLEEP_DURATION=1
else
  # Check if the input is a whole integer greater than or equal to 1
  if [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
    SLEEP_DURATION="$1"
  else
    echo "Error: <sleep_duration_in_minutes> must be a whole integer greater than or equal to 1."
    exit 1
  fi
fi

# Convert sleep duration to seconds
SLEEP_DURATION_SECONDS=$((SLEEP_DURATION * 60))

# ----------------------------------------------------------------------------
# Determine the job ID and task ID if running in an SGE job (including arrays)
# ----------------------------------------------------------------------------
JOB_ID=${JOB_ID:-}
TASK_ID=${SGE_TASK_ID:-}

# Ensure TASK_ID is empty if it's not a real task ID
if [ -z "$SGE_TASK_ID" ] || [ "$SGE_TASK_ID" = "undefined" ]; then
  TASK_ID=""
fi

# -------------------------------------------------------------------
# Set log file name based on job information
# -------------------------------------------------------------------
if [ -n "$JOB_ID" ] && [ -n "$TASK_ID" ]; then
  LOGFILE="resource_usage_${JOB_ID}_${TASK_ID}.log"
elif [ -n "$JOB_ID" ]; then
  LOGFILE="resource_usage_${JOB_ID}.log"
else
  LOGFILE="resource_usage.log"
fi

# -------------------------------------------------------------------
# Determine whether GPU monitoring should be enabled
# -------------------------------------------------------------------
GPU_MONITOR=false

# Check if GPU usage should be monitored in job script
if [ -n "$JOB_ID" ]; then
  GPU_JOB=$(qstat -j $JOB_ID | grep -i 'gpu=true')
  if [ -n "$GPU_JOB" ]; then
    GPU_MONITOR=true
  fi
fi

# For standalone runs, check if nvidia-smi is present and GPUs exist
if [ -z "$JOB_ID" ]; then
  if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
      GPU_MONITOR=true
    fi
  fi
fi

# -------------------------------------------------------------------
# Log the hostname and start time
# -------------------------------------------------------------------

# Log the hostname and start date/time
HOSTNAME=$(hostname | cut -d'.' -f1)
echo "Monitoring started on: $HOSTNAME" > "$LOGFILE"

echo "--------------------------------------------------------" >> "$LOGFILE"

if [ "$GPU_MONITOR" = true ]; then
  echo -e "Timestamp           | %CPU    | Memory (GB)  | GPU Usage" >> "$LOGFILE"
else
  echo -e "Timestamp           | %CPU    | Memory (GB)" >> "$LOGFILE"
fi
echo "--------------------------------------------------------" >> "$LOGFILE"
#echo "DEBUG USER: $USER" >> "$LOGFILE"
#echo "DEBUG JOB_ID: $JOB_ID" >> "$LOGFILE"
#echo "DEBUG TASK_ID: $TASK_ID" >> "$LOGFILE"

# -------------------------------------------------------------------
# Function: get_job_pids
#
# For a given SGE job id, task id find the sge_shepherd process and then
# list all descendant PIDs.
# -------------------------------------------------------------------
get_job_pids() {
  local jobid="$1"
  local taskid="$2"
  local shep_pid=""

  if [ -n "$taskid" ]; then
    if [ -f "/opt/gridengine/default/spool/$HOSTNAME/active_jobs/${jobid}.${taskid}/pid" ]; then
      shep_pid=$(cat "/opt/gridengine/default/spool/$HOSTNAME/active_jobs/${jobid}.${taskid}/pid")
    fi
  else
    shep_pid=$(pgrep -f "sge_shepherd-${jobid}")
  fi

  if [ -z "$shep_pid" ]; then
    return
  fi

  # Non-recursive version to get all descendant PIDs
  declare -a pids_to_check=("$shep_pid")
  declare -a all_pids=()

  while [ ${#pids_to_check[@]} -gt 0 ]; do
    current_pid=${pids_to_check[0]}  # Get the first PID in the array
    pids_to_check=("${pids_to_check[@]:1}")  # Remove the first element

    children=$(pgrep -P "$current_pid")

    for child in $children; do
      # Check if the child is an actual process (not just a thread)
      if [ -d "/proc/$child" ] && ps -o pid,comm --no-headers -p "$child" | grep -v '\]$' >/dev/null; then
        all_pids+=("$child")
        pids_to_check+=("$child")  # Add the child PID to the list of PIDs to check
      fi
    done
  done

  # Print shepherd PID and all descendant PIDs
  echo "$shep_pid"
  printf "%s\n" "${all_pids[@]}"
}

#-------------------------------------------------------------------
# Function: get_gpu_usage
# ------------------------------------------------------------------
get_gpu_usage() {
  current_user="$USER"
  declare -A gpu_mem_usage
  declare -A gpu_utilization_total
  declare -A gpu_process_count
  declare -A gpu_map
  gpu_usage_array=()

  # Map GPU UUIDs to indices
  index=0
  while IFS= read -r line; do
    uuid=$(echo "$line" | awk -F 'UUID: ' '{print $2}' | awk '{print $1}' | tr -d ')')
    gpu_map["$uuid"]=$index
    ((index++))
  done < <(nvidia-smi --list-gpus)

  # Query total GPU utilization per slot
  readarray -t gpu_utilization < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

  # Get the list of GPU processes (one-time nvidia-smi query)
  gpu_processes=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader)

  # Filter job-specific PIDs if provided
  if [ -n "$1" ]; then
    job_pids="$1"
  else
    job_pids=$(ps -u "$USER" -o pid=)
  fi

  # Create an associative array for faster PID lookup
  declare -A pid_lookup
  for pid in $job_pids; do
    pid_lookup["$pid"]=1
  done

  # Process each GPU process
  while IFS=',' read -r pid gpu_uuid used_memory; do
    pid=$(echo "$pid" | xargs)
    gpu_uuid=$(echo "$gpu_uuid" | xargs)
    used_memory=$(echo "$used_memory" | sed 's/MiB//' | xargs)

    # Convert memory from MiB to GB
    if [[ "$used_memory" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    used_memory_gb=$(awk "BEGIN {printf \"%.2f\", $used_memory/1024}")
    else
      used_memory_gb=""
    fi

    # Skip invalid PIDs or those not in the job-specific PIDs list
    if [[ -z "$pid" || ! "${pid_lookup[$pid]}" ]]; then
      continue
    fi

    process_user=$(ps -o user= -p "$pid" | tr -d ' ')

    if [ "$process_user" == "$current_user" ]; then
      gpu_index=${gpu_map["$gpu_uuid"]}

      # Initialize only if not set
      gpu_mem_usage["$gpu_index"]=${gpu_mem_usage["$gpu_index"]:-0}
      gpu_process_count["$gpu_index"]=${gpu_process_count["$gpu_index"]:-0}
      gpu_utilization_total["$gpu_index"]=${gpu_utilization_total["$gpu_index"]:-0}

      if [ -n "$gpu_index" ]; then
          # Accumulate memory usage and count processes per GPU
          gpu_mem_usage["$gpu_index"]=$(awk "BEGIN {print ${gpu_mem_usage["$gpu_index"]} + $used_memory_gb}")
          gpu_process_count["$gpu_index"]=$((gpu_process_count["$gpu_index"] + 1))
          gpu_utilization_total["$gpu_index"]=${gpu_utilization["$gpu_index"]}
      fi
    fi
  done <<< "$gpu_processes"

  # Format the output with proper alignment
  for gpu_index in "${!gpu_mem_usage[@]}"; do
      total_utilization=${gpu_utilization_total["$gpu_index"]}
      total_mem=${gpu_mem_usage["$gpu_index"]}
      proc_count=${gpu_process_count["$gpu_index"]}

      # Ensure proper alignment using printf
      gpu_usage_array+=("GPU Slot: ${gpu_index} | Mem: $(printf '%6.2f' "$total_mem") GB | GPU Utilization: $(printf '%3d' "$total_utilization")% | Processes: $(printf '%2d' "$proc_count")")
  done

  # Output all GPU usage info with aligned columns
  echo "${gpu_usage_array[@]}"
}

# -------------------------------------------------------------------
# Function: monitor_usage
# -------------------------------------------------------------------
monitor_usage() {
  while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    if [ -n "$JOB_ID" ]; then
      # Refresh job_pids every loop iteration
      job_pids=$(get_job_pids "$JOB_ID" "$TASK_ID")

      CPU_USAGE=0
      MEM_USAGE=0

      if [ -n "$job_pids" ]; then
        # Aggregate CPU and Memory usage for all job PIDs
        while read -r pid; do
          CPU=$(ps -p "$pid" -o %cpu --no-headers 2>/dev/null | awk '{sum+=$1} END {print sum}')
          MEM=$(ps -p "$pid" -o rss --no-headers 2>/dev/null | awk '{sum+=$1} END {print sum}')
          CPU_USAGE=$(awk -v c1="$CPU_USAGE" -v c2="$CPU" 'BEGIN {print c1 + c2}')
          MEM_USAGE=$(awk -v m1="$MEM_USAGE" -v m2="$MEM" 'BEGIN {print m1 + m2}')
        done <<< "$job_pids"
      fi

      # Convert KB to GB
      MEM_USAGE=$(awk -v mem="$MEM_USAGE" 'BEGIN {printf "%.2f", mem/1024/1024}')
      
      # Get GPU usage specific to the job
      if [ "$GPU_MONITOR" = true ]; then
        GPU_USAGE=$(get_gpu_usage "$job_pids")
        # Remove extra spaces from GPU_USAGE
        # GPU_USAGE=$(echo "$GPU_USAGE" | xargs)
        if [ -z "$GPU_USAGE" ]; then
          printf "%-19s | %-7s | %-12s\n" "$TIMESTAMP" "$CPU_USAGE" "$MEM_USAGE" >> "$LOGFILE"
        else
          printf "%-19s | %-7s | %-12s | %-40s\n" "$TIMESTAMP" "$CPU_USAGE" "$MEM_USAGE" "$GPU_USAGE" >> "$LOGFILE"
        fi
      else
        printf "%-19s | %-7s | %-12s\n" "$TIMESTAMP" "$CPU_USAGE" "$MEM_USAGE" >> "$LOGFILE"
      fi
    else
      # Aggregate usage for all user processes
      CPU_USAGE=$(ps -u "$USER" -o %cpu --no-headers 2>/dev/null | awk '{sum+=$1} END {print sum}')
      MEM_USAGE=$(ps -u "$USER" -o rss --no-headers 2>/dev/null | awk '{sum+=$1} END {printf "%.2f", sum/1024/1024}')
      
      if [ "$GPU_MONITOR" = true ]; then
        GPU_USAGE=$(get_gpu_usage)
        # Remove extra spaces from GPU_USAGE
        # GPU_USAGE=$(echo "$GPU_USAGE" | xargs)
        if [ -z "$GPU_USAGE" ]; then
          printf "%-19s | %-7s | %-12s\n" "$TIMESTAMP" "$CPU_USAGE" "$MEM_USAGE" >> "$LOGFILE"
        else
          printf "%-19s | %-7s | %-12s | %-40s\n" "$TIMESTAMP" "$CPU_USAGE" "$MEM_USAGE" "$GPU_USAGE" >> "$LOGFILE"
        fi
      else
        printf "%-19s | %-7s | %-12s\n" "$TIMESTAMP" "$CPU_USAGE" "$MEM_USAGE" >> "$LOGFILE"
      fi
    fi

    sleep "$SLEEP_DURATION_SECONDS"
  done
}

# -------------------------------------------------------------------
# Run the monitor_usage function in the background
# -------------------------------------------------------------------
monitor_usage &
MONITOR_PID=$!

# Ensure that monitoring stops when the script is terminated
trap "kill $MONITOR_PID" EXIT

# Keep the script running
wait
