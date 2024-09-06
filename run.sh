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

taskset -c 0-23 mpirun --allow-run-as-root -np ${n_works} python3 ${exe_file}
#msrun --worker_num ${n_works} --local_worker_num ${n_works} --log_dir=msrun_log test.py
