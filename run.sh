if [[ -z $1 ]]; then 
    n_works=1
else
    n_works=$1
fi

mpirun --allow-run-as-root -np ${n_works} python3 test.py
#msrun --worker_num ${n_works} --local_worker_num ${n_works} --log_dir=msrun_log test.py
