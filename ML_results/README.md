# sif
SIF Paper workflow

Right now, I've three different datasets:

- Dataset 1: Original RN Data (~500)
- Dataset 2: Data generated from equations (~140k)
- Dataset 3: Frac3D dataset is either sampled (used for GPSR) or not
    - 3a: Franc3D FULL (~140k)
    - 3b: Franc3D PHI SAMPLED (~19k)
 

For different test scenarios I'll be working on the following:

- Train all ML models on Dataset 1, Dataset 2, Dataset 3a, and Dataset 3b, using all values of a/c (not splitting here) and compare the results (accuracy and time).
- Split a/c to make two cases, a/c <= 1 and a/c > 1, and for each case do the following:
    - Solve at phi = pi/2: Doing this will ensure that the labels are M (g gets separated)
        - Train ML model (using a/c and a/t) to predict M.
        - For both Dataset 3a and Dataset 3b, divide Mg by M (predicted from model trained in above part) to get g and then train different ML model which learns to predict g. 
    - Solve at all phi values (here no splitting between mg) for all 4 datasets.
