#!/bin/bash
declare -a pdbids=(
"1zih" 
#"1anr" 
#"5udz"
)
declare -a pred_types=(
"x0" 
#"noise"
)
declare -a SC_options=(
"True" 
#"False"
)

EPOCH=30
NUMSAMPLES=1000
SYSCONFIG="/home/lherron/scratch/repos/thermodynamic-diffusion/systems/config/zrt_config.yaml"
EXPCONFIG="/home/lherron/scratch/repos/thermodynamic-diffusion/systems/config/RNA_config.yaml"

for PDBID in "${pdbids[@]}"; do
    for PREDTYPE in "${pred_types[@]}"; do
        for SC in "${SC_options[@]}"; do
            if [ "$SC" = "True" ]; then
                EXPID=${PREDTYPE}_mid_attn_sc
            elif [ "$SC" = "False" ]; then
                EXPID=${PREDTYPE}_mid_attn_no_sc
            fi
            sed s+OUTFILE+../outfiles/${PDBID}_${EXPID}.out+g zrt_sample_template.sh > zrt_sample.sh
                sbatch zrt_sample.sh $PDBID $EXPID $PREDTYPE $EPOCH $NUMSAMPLES $SC $SYSCONFIG $EXPCONFIG
        done
    done
done


