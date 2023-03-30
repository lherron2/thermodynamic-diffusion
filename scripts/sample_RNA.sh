#!/bin/bash
PDB=$1
EPOCH=$2
NSAMP=100000
for T in {290..420..10}; do sbatch zrt_sample.sh $PDB $EPOCH $T $NSAMP; done
