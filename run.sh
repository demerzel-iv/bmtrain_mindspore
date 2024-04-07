n_works=$1

mpirun --allow-run-as-root -np ${n_works} python3 test.py
