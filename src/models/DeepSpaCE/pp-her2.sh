#!/bin/bash

# Define the list of symbols
SYMBOLS=(A1 A2 A3 A4 A5 A6 B1 B2 B3 B4 B5 B6 C1 C2 C3 C4 C5 C6 D1 D2 D3 D4 D5 D6 E1 E2 E3 F1 F2 F3 G1 G2 G3 H1 H2 H3)

IDX=$1

# Loop over the list of symbols
LEFT_OUT=${SYMBOLS[$IDX]}

# Create an array of the remaining symbols
OTHER=("${SYMBOLS[@]}")
unset OTHER[$IDX]

# Join the remaining symbols with commas
OTHER_JOINED=$(IFS=, ; echo "${OTHER[*]}")

# Do something with the left-out and remaining symbols
echo "Sample: $LEFT_OUT"

python CropImage.py \
        --dataDir ../data/her2_formatted \
        --sampleName "$LEFT_OUT" \
        --transposeType 0 \
        --radiusPixel 75 \
        --extraSize 150 \
        --quantileRGB 80 \
        --threads 8

Rscript --verbose NormalizeUMI-her2.R \
        --dataDir ../data/her2_formatted \
        --sampleName "$LEFT_OUT" \
        --threshold_count 0 \
        --threshold_gene 0
        
# DO NOT THRESHOLD GENES AS THIS WILL REMOVE SOME TO PREDICT
