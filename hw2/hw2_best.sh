#!/bin/bash
wget -O model/reg_NN_weights.h5 "https://www.dropbox.com/s/t2ammpebqdlgifl/reg_NN_weights.h5?dl=0"
wget -O model/clf_NN_weights.h5 "https://www.dropbox.com/s/lqlhkvqte7twle3/clf_NN_weights.h5?dl=0"
python3 hw2_best_test.py $5 $6 $3 $4