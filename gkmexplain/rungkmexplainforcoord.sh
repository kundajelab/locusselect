#!/usr/bin/env bash
idx=$1
echo "Running for $idx"
cd coordinates_$idx
for neg in r1 r2 r3 r4 r5
do
    echo "Neg set $neg"
    for dhs in m1 m2 m3 m4
    do
        echo "DHS $dhs"
        ../lsgkm/bin/gkmexplain coordinates_$idx.fa /data/locusselect/gkmsvmmodels/"K562_"$dhs"_"$neg".model.txt" "K562_"$dhs"_"$neg".explanation.txt"  2> "K562_"$dhs"_"$neg".error.log" > "K562_"$dhs"_"$neg".output.log"
    done
done
