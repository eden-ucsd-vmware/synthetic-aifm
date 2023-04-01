#!/bin/bash

source ../../shared.sh

sudo pkill -9 main

make clean
make -j

for i in {1..10}
do
  rerun_local_iokerneld_args simple 1,2,3,4,5,6,7,8,9,11,12,13,14,15
  rerun_mem_server
  run_program ./main | grep "=" 1>log.local.$i 2>&1
done

kill_local_iokerneld
