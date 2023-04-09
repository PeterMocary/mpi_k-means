#!/bin/bash

# get numbers count
if [ $# -lt 1 ];then 
    numbers=32;
else
    numbers=$1;
fi;

# compile
mpic++ --prefix /usr/local/share/OpenMPI -o parkmeans parkmeans.cc

# create input file
dd if=/dev/random bs=1 count=$numbers of=numbers

# run
mpirun --prefix /usr/local/share/OpenMPI -oversubscribe -np $numbers parkmeans

# clean up
rm -f oems numbers
