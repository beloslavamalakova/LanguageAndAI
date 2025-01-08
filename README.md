# JBC090 (2024-2) Language and AI
Beloslava Malakova , Alicja Gwiazda, Dani Stoyanova, Boris Stoyanov

## Data Preprocessing

## Logistic Regression
This component applies Logistic Regression to our data in multiple configurations. Specifically, we evaluate:

1. **Simple Logistic Regression**  
   - Uses no explicit regularization (`penalty=None`).
   - Serves as a baseline model.

2. **Regularized Logistic Regression**  
   - **Lasso (L1)**: Encourages sparsity in model coefficients.
   - **Ridge (L2)**: Shrinks coefficients to reduce overfitting without forcing them to zero.
   - **Elastic Net**: A combination of L1 and L2 penalties, balancing feature sparsity and coefficient shrinkage.

### TF-IDF Representations
We compare three TF-IDF matrices derived from:
- **Cleaned (1)**: Dataset with standard text preprocessing.
- **Cleaned (2)**: Another variant or additional preprocessing steps. (@Boris @Dani, add what is different:))
- **Raw**: Dataset without any preprocessing before TF-IDF.

Each of the four Logistic Regression configurations (Simple, L1, L2, and Elastic Net) is trained on a TF-IDF matrix.  

## SVM

## Results and Discussion
