#!/bin/sh
../../crf_learn -f 3 -c 4.0 template train.data model
../../crf_test -m model test.data >1.txt

../../crf_learn -a MIRA -f 3 template train.data model
../../crf_test -m model test.data
rm -f model
