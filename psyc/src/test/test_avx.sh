#!/bin/bash

cwd=$(pwd)
cwd=$(basename "$cwd")

if [ "$cwd" = "test" ]; then
    cd ../../
fi

if [ -e /tmp/avx.nn.data ]; then
    rm /tmp/avx.nn.data
fi

if [ -e /tmp/avx.cnn.data ]; then
    rm /tmp/avx.cnn.data
fi

if [ -e /tmp/no_avx.nn.data ]; then
    rm /tmp/no_avx.nn.data
fi

if [ -e /tmp/no_avx.cnn.data ]; then
    rm /tmp/no_avx.cnn.data
fi

make clean && make

bin/psycl --enable-colors --name "AVX NN" --load resources/pretrained.mnist.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --save /tmp/avx.nn.data

bin/psycl --enable-colors --name "AVX CNN" --load resources/pretrained.cnn.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --save /tmp/avx.cnn.data

bin/psycl --enable-colors --name "AVX L2 NN" --load resources/pretrained.mnist.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --l2-decay 2.5 --save /tmp/avx.l2_nn.data

bin/psycl --enable-colors --name "AVX L2 CNN" --load resources/pretrained.cnn.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --l2-decay 2.5 --save /tmp/avx.l2_cnn.data

make clean && make AVX=off

bin/psycl --enable-colors --name "NO AVX NN" --load resources/pretrained.mnist.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --save /tmp/no_avx.nn.data

bin/psycl --enable-colors --name "NO AVX CNN" --load resources/pretrained.cnn.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --save /tmp/no_avx.cnn.data

bin/psycl --enable-colors --name "NO AVX L2 NN" --load resources/pretrained.mnist.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --l2-decay 2.5 --save /tmp/no_avx.l2_nn.data

bin/psycl --enable-colors --name "NO AVX L2 CNN" --load resources/pretrained.cnn.data --training-no-shuffle --train --mnist --epochs 1 --training-datalen 1 --validation-datalen 0 --batch-size 10 --l2-decay 2.5 --save /tmp/no_avx.l2_cnn.data

OBJS=(psyc utils convolutional recurrent lstm)
COBJS=""
for OBJ in ${OBJS[@]}; do
    echo "gcc -o /tmp/$OBJ.o -c src/$OBJ.c"
    gcc -o /tmp/$OBJ.o -c src/$OBJ.c
    COBJS="$COBJS /tmp/$OBJ.o"
done
if [ -e /tmp/compare_avx.o ]; then
    rm /tmp/compare_avx.o
fi
if [ -e /tmp/compare_avx ]; then
    rm /tmp/compare_avx
fi
gcc -o /tmp/compare_avx.o -c "src/test/compare_avx.c"
gcc -o /tmp/compare_avx $COBJS /tmp/compare_avx.o -lz -lm

/tmp/compare_avx

make clean > /dev/null
