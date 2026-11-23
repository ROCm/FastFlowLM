#!/bin/bash
# run.sh <build dir> <backing...>   e.g. run.sh build bo user4k user2m
b=$1; shift
exe=${PROBE:-$(dirname $0)/probe}
echo "mode,R,P,backing,median_us,min_us,GBps,Mrows_s,check,pages"
while IFS=$'\t' read -r name mode R P ib ob per nch pitch cb; do
  for bk in "$@"; do
    if ! out=$(timeout 60 $exe $b/$name/final.xclbin $b/$name/insts.bin $mode $R $P $ib $ob $per $nch $bk ${ITERS:-20} ${pitch:-0} 2>&1); then
      out="$name,$bk,FAILED,$(echo "$out" | tail -2 | tr '\n' ' ')"
    fi
    echo "$out"
  done
done < $b/cases.tsv
