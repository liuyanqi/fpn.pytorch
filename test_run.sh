#!/bin/bash
python test_net.py --dataset ycb_voc --net res101 --cfg cfgs/res101.yml --cuda --checksession 1 --checkepoch 9 --checkpoint 5197 --vis