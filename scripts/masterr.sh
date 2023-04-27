#!/bin/bash

sbatch deepspace-pf.sh
sbatch Hist2ST_pf.sh
sbatch histogene_pf-cv.sh
sbatch stimage_pf_cv.sh
sbatch stnet-pf-cv-224.sh