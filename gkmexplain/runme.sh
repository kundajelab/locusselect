#!/usr/bin/env bash

#clone and install github.com/kundajelab/lsgkm in present dir for gkmexplain

for idx in 0 1 2 3 4 5
do
    rm -r coordinates_$idx
    mkdir coordinates_$idx
    cd coordinates_$idx
    cp ../../coordinates/coordinates_$idx.fa.gz .
    echo `pwd`
    zcat coordinates_$idx.fa.gz > coordinates_$idx.fa
    cd ..
done


for idx in 0 1 2 3 4 5
do
    ./rungkmexplainforcoord.sh $idx &
done
             
