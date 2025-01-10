#!/bin/bash

rm Figures/training/*.png
python test_models.py -m 0 -cuda train models -nmax 3 -e 5000 -s 100 -os 900
mkdir Figures/training_3 || true 
mv Figures/training/*.png Figures/training_3 || true

#rm Figures/training/*.png
#python test_models.py -m 0 -cuda train models -nmax 4 -e 5000 -s 100 -os 900
#mkdir Figures/training_4 || true
#mv Figures/training/*.png Figures/training_4 || true

#rm Figures/training/*.png
#python test_models.py -m 0 -cuda train models -nmax 5 -e 5000 -s 100 -os 900
#mkdir Figures/training_5 || true 
#mv Figures/training/*.png Figures/training_5 || true

