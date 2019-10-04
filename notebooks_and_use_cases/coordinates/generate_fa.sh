#!/usr/bin/env bash

for i in 0 1 2 3 4 5
do
    bedtools getfasta -fi /data/refs/hg19/male.hg19.fa -fo coordinates_$i.fa -bed coordinates_$i.bed
done

gzip *.fa
