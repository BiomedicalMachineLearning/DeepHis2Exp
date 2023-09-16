#!/bin/bash

# Define the list of symbols
SYMBOLS=(1142243F 1160920F CID4465 CID44971 CID4535 CID4290)

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
# echo "Other: ${OTHER[@]}"
# echo "Other joined: $OTHER_JOINED"

python CropImage.py \
        --dataDir ../data/9BC_combined \
        --sampleName "$LEFT_OUT" \
        --transposeType 0 \
        --radiusPixel 75 \
        --extraSize 150 \
        --quantileRGB 80 \
        --threads 8

Rscript --verbose NormalizeUMI-alex.R \
        --dataDir ../9BC_combined \
        --sampleName "$LEFT_OUT" \
        --threshold_count 1000 \
        --threshold_gene 1000
