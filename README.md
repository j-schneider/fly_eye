# fly-eye

Code used in the PLOS ONE publication:
"Can Drosophila melanogaster tell who's who?"
(https://doi.org/10.1371/journal.pone.0205043)

Data can be downloaded from: https://doi.org/10.5683/SP2/JP4WDF

Requires Keras with Tensorflow backend

Example usage:
python fly-eye.py --model fly-eye --data flies --week 1 --pic_dir /media/fly/ssd_scratch/Data/Flies/ --pixels 29 --resize 33 --batch_size 128 --opt adam
