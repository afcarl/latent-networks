#!/bin/bash

conda create -n my_root --clone="/opt/conda"
source activate my_root
conda install -y fuel pytables
