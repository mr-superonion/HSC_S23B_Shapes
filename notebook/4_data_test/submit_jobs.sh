#!/bin/bash

xsubLarge mpirun -np 400 ./test_1_bin.py --start 0 --end 1000
sleep 1.0
xsubLarge mpirun -np 400 ./test_2_tancross1.py --start 0 --end 1000
sleep 1.0
xsubLarge mpirun -np 400 ./test_2_tancross2_gaia.py --start 0 --end 1000
sleep 1.0
xsubLarge mpirun -np 400 ./test_2_tancross3_gglens.py --start 0 --end 1000
sleep 1.0
xsubLarge mpirun -np 400 ./test_3_galstar.py --start 0 --end 1000
