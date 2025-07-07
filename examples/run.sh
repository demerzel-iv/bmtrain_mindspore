export MS_ENABLE_RUNTIME_PROFILER=1

# if number of arguments is less than 2, print usage
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <n_works> <exe_file>"
    exit 1
fi

n_works=$1
exe_file=$2

msrun --bind_core True \
     --master_port 8129\
     --worker_num ${n_works} --local_worker_num ${n_works} ${exe_file} 
