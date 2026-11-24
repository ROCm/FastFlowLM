#!/bin/bash
# Copyright © Advanced Micro Devices, Inc., or its affiliates. All rights reserved.



# some constant variable
# Set variables
ENABLE_CHESSCC=1
M=4096
K=4096
N=8192
m=128
k=512
n=64
IS_B_ROW_MAJOR=1
row_num=4
col_num=8
r=8
s=8
t=8
ENABLE_AXI4=1
IS_B_COL_MAJOR_S_T_ROW_MAJOR=0
IS_BFLOAT16_OUT_TY=1

rm -f *.log
run_all_modes() {
    local m_val=$1
    local k_val=$2
    local n_val=$3

    # xchess with cache for row major order
    rm -r ../build
    make clean
 
    IS_B_ROW_MAJOR=1
    ENABLE_AXI4=1
    ENABLE_CHESSCC=1
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY kernel
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > xchess_row_major_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

    # xchess with cache for col major order
    IS_B_ROW_MAJOR=0
    ENABLE_AXI4=1
    ENABLE_CHESSCC=1
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean    
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > xchess_col_major_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

    # xchess without cache for row major order
    IS_B_ROW_MAJOR=1
    ENABLE_AXI4=0
    ENABLE_CHESSCC=1
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean    
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > xchess_row_major_no_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

    # xchess without cache for col major order
    IS_B_ROW_MAJOR=0
    ENABLE_AXI4=0
    ENABLE_CHESSCC=1
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean    
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > xchess_col_major_no_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

    # peano for same process
    rm -r ../build
    make clean
    # peano for row major order with cache
    IS_B_ROW_MAJOR=1
    ENABLE_AXI4=1
    ENABLE_CHESSCC=0
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY clean
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY kernel
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > peano_row_major_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log
    # peano for col major order with cache
    IS_B_ROW_MAJOR=0
    ENABLE_AXI4=1
    ENABLE_CHESSCC=0
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean    
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > peano_col_major_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

    # peano for row major order without cache
    IS_B_ROW_MAJOR=1
    ENABLE_AXI4=0
    ENABLE_CHESSCC=0
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean    
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > peano_row_major_no_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

    # peano for col major order without cache
    IS_B_ROW_MAJOR=0
    ENABLE_AXI4=0
    ENABLE_CHESSCC=0
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY  clean    
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY bitstream
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY instruction
    make ENABLE_CHESSCC=$ENABLE_CHESSCC M=$M K=$K N=$N m=$m_val k=$k_val n=$n_val IS_B_ROW_MAJOR=$IS_B_ROW_MAJOR row_num=$row_num col_num=$col_num r=$r s=$s t=$t ENABLE_AXI4=$ENABLE_AXI4 IS_B_COL_MAJOR_S_T_ROW_MAJOR=$IS_B_COL_MAJOR_S_T_ROW_MAJOR IS_BFLOAT16_OUT_TY=$IS_BFLOAT16_OUT_TY run > peano_col_major_no_cache_${m_val}_${k_val}_${n_val}_bfloat${IS_BFLOAT16_OUT_TY}_colmaj${IS_B_COL_MAJOR_S_T_ROW_MAJOR}.log

}




run_all_modes 64 256 128
run_all_modes 64 512 64
run_all_modes 64 256 64
run_all_modes 128 512 64


