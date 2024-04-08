if [[ -z $1 ]]; then 
    n_works=1
else
    n_works=$1
fi

mpirun --allow-run-as-root -np ${n_works} python3 test.py
