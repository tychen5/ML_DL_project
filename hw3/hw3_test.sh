#!/bin/bash
wget -O models/norm "https://www.dropbox.com/s/mfd0pray9tdao56/norm?dl=0"
wget -O models/cnn_normal_0.7_3.h5_weights.h5 "https://www.dropbox.com/s/fitrb31b5hxseau/cnn_normal_0.7_3.h5_weights.h5?dl=0"
wget -O models/cnn_normal_0.7.h5_weights.h5 "https://www.dropbox.com/s/jdb2bpc2upqts2b/cnn_normal_0.7.h5_weights.h5?dl=0"
python3 test.py $1 $2