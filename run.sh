export MS_ENABLE_RUNTIME_PROFILER=1

if [[ -z $1 ]]; then 
    n_works=1
else
    n_works=$1
fi


if [[ -z $2 ]]; then 
    exe_file=test.py
else
    exe_file=$2
fi

#taskset -c 72-95 mpirun --allow-run-as-root -np ${n_works} python3 ${exe_file} 
#mpirun --bind-to-x --report-bindings \
#    --allow-run-as-root -np ${n_works} python3 ${exe_file} 

msrun --bind_core True \
     --master_port 8129\
     --worker_num ${n_works} --local_worker_num ${n_works} ${exe_file} 

#mpirun --allow-run-as-root -np ${n_works} python3 ${exe_file}
#msrun --worker_num ${n_works} --local_worker_num ${n_works} --log_dir=msrun_log test.py
