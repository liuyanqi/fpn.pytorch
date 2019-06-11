#!/bin/bash
python trainval_net.py --dataset pascal_voc --net res101 --bs 1 --cuda --epochs 10 --use_tfboard True