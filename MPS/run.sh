#!/bin/bash
sudo nsys profile --force-overwrite true -o mps_profile bash -c "
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50
  ./program1 & 
  ./program2 & 
  wait
"
