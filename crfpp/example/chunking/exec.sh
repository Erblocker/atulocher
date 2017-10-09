#!/bin/sh
../../crf_learn -c 4.0 template train.data model
../../crf_test -m model test.data > 1.txt

../../crf_learn -a MIRA template train.data model
../../crf_test -m model test.data > 2.txt

#../../crf_learn -a CRF-L1 template train.data model
#../../crf_test -m model test.data

rm -f model
