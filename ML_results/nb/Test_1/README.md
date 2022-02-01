# Test 1

## Dataset Description
Different datasets:
- Dataset 1: Original RN Data (~500)
- Dataset 2: Data generated from equations (~140k)
- Dataset 3: Frac3D dataset is either sampled (used for GPSR) or not
    - 3a: Franc3D FULL (~140k)
    - 3b: Franc3D PHI SAMPLED (~19k)
 
# Test Description
- SVM with polynomial kernel (different degrees) tested and compared for all three datasets using all values of a/c (not splitting here).
- SVM with rbf kernel (different degrees) tested and compared for all three datasets using all values of a/c (not splitting here).
- SVM with linear kernel (different degrees) tested and compared for all three datasets using all values of a/c (not splitting here).
- SVM with Neural Network tested and compared for all three datasets using all values of a/c (not splitting here).
- All methods compared.