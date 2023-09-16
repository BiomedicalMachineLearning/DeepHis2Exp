#!/bin/bash

export CV_IO_MAX_IMAGE_PIXELS=185000000000
export OPENCV_IO_MAX_IMAGE_PIXELS=185000000000

gene_symbols=$(python gene_list_her2.py)

PARENT_DIR="../results/her2_ds/deepspace"

# Define the list of symbols
SYMBOLS=(A1 A2 A3 A4 A5 A6 B1 B2 B3 B4 B5 B6 C1 C2 C3 C4 C5 C6 D1 D2 D3 D4 D5 D6 E1 E2 E3 F1 F2 F3 G1 G2 G3 H1 H2 H3)

IDX=$1

# Get the symbol to leave out
LEFT_OUT=${SYMBOLS[$IDX]}

# Create an array of the remaining symbols
OTHER=("${SYMBOLS[@]}")
unset OTHER[$IDX]

# Join the remaining symbols with commas
OTHER_JOINED=$(IFS=, ; echo "${OTHER[*]}")

# Do something with the left-out and remaining symbols
echo "Left out: $LEFT_OUT"
echo "Other: ${OTHER[@]}"
echo "Other joined: $OTHER_JOINED"

mkdir -p "$PARENT_DIR/$LEFT_OUT"

python DeepSpaCE.py \
    --dataDir ../data/her2_formatted \
    --outDir "$PARENT_DIR/$LEFT_OUT" \
    --sampleNames_train "$OTHER_JOINED" \
    --sampleNames_test "$LEFT_OUT" \
    --sampleNames_semi None \
    --semi_option normal \
    --seed 0 \
    --threads 8 \
    --GPUs 1 \
    --cuda \
    --transfer \
    --model VGG16 \
    --batch_size 128 \
    --num_epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --clusteringMethod graphclust \
    --extraSize 150 \
    --quantileRGB 80 \
    --augmentation flip,crop,color,random \
    --early_stop_max 5 \
    --cross_index 0 \
    --geneSymbols "$gene_symbols"
