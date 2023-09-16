#!/bin/bash

export CV_IO_MAX_IMAGE_PIXELS=185000000000
export OPENCV_IO_MAX_IMAGE_PIXELS=185000000000

gene_symbols=$(python gene_list_markers.py)

PARENT_DIR="../results/visium_generalization/deepspace"

python DeepSpaCE.py \
    --dataDir ../data/9BC_combined \
    --outDir "$PARENT_DIR" \
    --sampleNames_train 1142243F,CID4465,CID44971,CID4535 \
    --sampleNames_test 1142243F,CID4290,CID4465,CID44971,CID4535,1160920F,block1,block2,FFPE \
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
