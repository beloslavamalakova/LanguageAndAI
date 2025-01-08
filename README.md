# Impact of Gender-Neutral Data Cleaning on SVM and Logistic Regression for Gender Classification
Repository for a research paper "Impact of Gender-Neutral Data Cleaning on SVM and Logistic Regression for Gender Classification", conducted as part of the Language&AI (JBM090) course at Eindhoven Technical University of Technology lectured by dr. Chris Emmery. The dataset comes from a corpus created by and decribed in Emmery, C., Miotto, M., Kramp, S., & Kleinberg, B. (2024). *SOBR: A Corpus for Stylometry, Obfuscation, and Bias on Reddit*. Proceedings of the 2024 Conference on Language Resources and Evaluation (LREC). Retrieved from [https://aclanthology.org/2024.lrec-main.1302.pdf](https://aclanthology.org/2024.lrec-main.1302.pdf). Code is released under the MIT license.
```bibtex
@inproceedings{
  authors = {Malakova, Beloslava and Stoyanova, Dani and Stoyanov, Boris and Gwiazda, Alicja},
  title = {Impact of Gender-Neutral Data Cleaning on SVM and Logistic Regression for Gender Classification},
  year = {2024}
}
```
## TL;DR
-  **Data Cleaning:** Removed gender-implying words and replaced them with gender-neutral alternatives.  
-  **Data Processing:** Preprocessed both the original contaminated CSV and the gender-neutral CSV.  
-  **Feature Engineering:** Created TF-IDF encodings for (1) contaminated data, (2) gender-neutral data, and (3) raw, unprocessed data.  
-  **Model Training:**  
   - Trained **SVM** on all three encodings, fine-tuning the **C** hyperparameter (0.1, 1, 10).  
   - Trained **Logistic Regression** on all three encodings, fine-tuning regularization methods (**L1, L2, Elastic Net**).  
-  **Comparison:** Compared classification results across different models and data encodings.  

##  Dependencies

The code was written using these libraries and versions:


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
