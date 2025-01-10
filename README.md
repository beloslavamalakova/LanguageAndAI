# Impact of Gender-Neutral Data Cleaning on SVM and Logistic Regression for Gender Classification
Repository for a research paper "Impact of Gender-Neutral Data Cleaning on SVM and Logistic Regression for Gender Classification", conducted as part of the Language&AI (JBM090) course at Eindhoven Technical University of Technology lectured by dr. Chris Emmery. The dataset comes from a corpus created by and decribed in Emmery, C., Miotto, M., Kramp, S., & Kleinberg, B. (2024). *SOBR: A Corpus for Stylometry, Obfuscation, and Bias on Reddit*. Proceedings of the 2024 Conference on Language Resources and Evaluation (LREC). Retrieved from [https://aclanthology.org/2024.lrec-main.1302.pdf](https://aclanthology.org/2024.lrec-main.1302.pdf). Code is released under the MIT license.
```bibtex
@inproceedings{
  authors = {Malakova, Beloslava and Stoyanova, Dani and Stoyanov, Boris and Gwiazda, Alicja},
  title = {Impact of Gender-Neutral Data Cleaning on SVM and Logistic Regression for Gender Classification},
  year = {2024}
}
```
## Overview 
- [Paper Details](#paper-details) 
  - [TL;DR](#tldr)
  - [Reproduction](#reproduction)
  - [Dependencies](#dependencies)
  - [Resources](#resources)
- [Project's Pipeline](#projects-pipeline) 
  - [Data Cleaning](#1-data-cleaning)
  - [Data Preprocessing](#2-data-preprocessing)
  - [Feature Engineering](#3-feature-engineering)
  - [Logistic Regression](#4-logistic-regression)
  - [SVM](#5-svm)
- [Configuration](#configuration)
- [Extensions](#extensions)
## Paper Details
### TL;DR
-  **Data Cleaning:** Removed gender-implying words and replaced them with gender-neutral alternatives.  
-  **Data Processing:** Preprocessed both the original contaminated CSV and the gender-neutral CSV.  
-  **Feature Engineering:** Created TF-IDF encodings for (1) contaminated data, (2) gender-neutral data, and (3) raw, unprocessed data.  
-  **Model Training:**  
   - Trained **SVM** on all three encodings, fine-tuning the **C** hyperparameter (0.1, 1, 10, 20, 50, 100).  
   - Trained **Logistic Regression** on all three encodings, fine-tuning regularization methods (**L1, L2, Elastic Net**).  
-  **Comparison:** Compared classification results across different models and data encodings.  
### Reproduction
Order of running the code:
> **Important Note:** The code processes a modified gender CSV file where all information is joint into a single column, with values separated by commas (manually adjusted in Excel). Make sure to always update the file path in the code to match your CSV file location.
1. To clean the initial data by removing the gender-implying words run `preprocessing_separate.py`. It contains the replacement-words dictionary, which can be expanded. It preprocesses the data (may work with the original or the cleaned version) by adhering to standard preprocessing practices and creates `gender_shuffled.csv`, `gender_preprocessed.csv`, `gender_preprocessed_neutral_gender.csv`.
3. To compute the TF-IDF scores of every term across all documents run `tf_idf_new(without_preprocessing).py`. It creates for each of the three newly created files 5 new files, used later for both SVM and Logistic Regression:
   - f`{base_name}_tf_idf_sparse.csv`
   - f`{base_name}_tf_idf_sparse_matrix.npz`
   - f`{base_name}_vocabulary.csv`
   - f`{base_name}_document_labels.csv`
   - f`{base_name}_tfidf_vectorizer.pkl`
4. To perform Logistic Regression run `logistic_regression.py` with:
   - tf_idf_data = pd.read_csv("NAME_tf_idf_sparse.csv")
   - y = pd.read_csv("NAME_document_labels.csv")["female"]
   - word_mapping = pd.read_csv("NAME_vocabulary.csv")
5. To perform SVM run `svm_boris.py` with:
   - input_file = "gender_preprocessed_neutral_gender.csv"
6. To see the top words for each gender run `svm_top_features.py`
7. To test a given sentence run `test_single_sentance.py`

###  Dependencies

The code was written using these libraries and versions:
| Tool        | Version |
|----------------|---------|
| Python         | 3.10    |
| scikit-learn   | 1.3.0   |
| pandas         | 2.1.1   |
| numpy          | 1.24.3  |
| scipy          | 1.11.4  |
| tqdm           | 4.66.1  |
| joblib         | 1.3.2   |
### Resources
The experiments were conducted on multiple laptops with varying hardware configurations.  
Typical specifications included:  
- CPU: Intel Core i5, i7 and Apple M1  
- RAM: 8GB to 32GB  
- OS: macOS Sonoma, Windows 11, Ubuntu 20.04 LTS
- No dedicated GPUs were used for this analysis.
Specific running times were included in details in the paper.
## Project's Pipeline
### 1. Data Cleaning
This project's data cleaning process appends an unusual technique to the standard data cleaning practices. Since our primary focus was on handling a contaminated dataset, data cleaning here specifically refers to identifying gender-implying words(e.g., "he," "she," "man," "woman") and replacing them with gender-neutral alternatives(e.g., "they," "person").
### 2. Data Preprocessing
A single text post is preprocessed using the following steps:
1. Convert to lowercase
2. Remove punctuation
3. Split into words
4. Discard words containing characters outside of `[a-z0-9]`
5. Naively normalize words to a base form:  
   - Remove `'ing'` ending if present (e.g., `"playing"` → `"play"`)  
   - Remove `'ed'` ending if present (e.g., `"painted"` → `"paint"`)  
   - Remove trailing `'s'` for plural forms (e.g., `"kings"` → `"king"`)  
6. Remove words longer than 25 characters
7. Reduce occurrences of 3+ consecutive identical letters to 2:  
   - `"Haaaaappy"` → `"haappy"`.  
8. Return a cleaned list of words

**Limitation:**
Words ending with `'s'` like `"boss"` may be simplified to `"bos"`. However, the word `"bos"` is still connected to `"boss"` in the dataset, minimizing the negative impact.
### 3. Feature Engineering
The usage of the TF-IDF approach allowed to encode textual data into sparse matrices, which are suitable for the models.
Key steps in this approach:
1. **TF-IDF Calculation** 
2. **Encodings Creation**
    - **Contaminated**: Dataset with standard text preprocessing.
    - **Cleaned**: Dataset with gender-neutral words and standard text preprocessing.
    - **Raw**: Dataset without any preprocessing before TF-IDF.

### 4. Logistic Regression
This component applies Logistic Regression to our data in multiple configurations. Specifically, we evaluate:
1. **Simple Logistic Regression**  
   - Uses no explicit regularization (`penalty=None`).
   - Serves as a baseline model.
2. **Regularized Logistic Regression**  
   - **Lasso (L1)**: Encourages sparsity in model coefficients.
   - **Ridge (L2)**: Shrinks coefficients to reduce overfitting without forcing them to zero.
   - **Elastic Net**: A combination of L1 and L2 penalties, balancing feature sparsity and coefficient shrinkage.

We compare the three TF-IDF matrices derived from the encodings(Contaminated, Cleaned, and Raw).

Each of the four Logistic Regression configurations (Simple, L1, L2, and Elastic Net) is trained on a TF-IDF matrix.  

Order of running the code:
1. preprocessing_separate.py -> creates "gender_shuffled.csv", "gender_preprocessed.csv", "gender_preprocessed_neutral_gender.csv"
2. tf_idf_new(without_preprocessing).py -> creates for each of the three newly created files 5 new files, used later for both SVM and Logistic Regression:
   - f"{base_name}_tf_idf_sparse.csv"
   - f"{base_name}_tf_idf_sparse_matrix.npz"
   - f"{base_name}_vocabulary.csv"
   - f"{base_name}_document_labels.csv"
   - f"{base_name}_tfidf_vectorizer.pkl"
3. run logistic_regression.py with:
   - tf_idf_data = pd.read_csv("NAME_tf_idf_sparse.csv")
   - y = pd.read_csv("NAME_document_labels.csv")["female"]
   - word_mapping = pd.read_csv("NAME_vocabulary.csv")


### 5. SVM
This component applies SVM to our data in multiple configurations. Specifically, we evaluate SVM using a linear kernel with different values of C hyperparameter. Tested values were: 0.1, 1, 10, 20, 50, 100; leading to six different versions of SVM.
We compare the three TF-IDF matrices derived from the encodings(Contaminated, Cleaned, and Raw).

Each of the six SVM configurations is trained on a TF-IDF matrix and calculates the result as the average of the threefold cross valudation. 

Order of running the code:
1. preprocessing_separate.py -> creates "gender_shuffled.csv", "gender_preprocessed.csv", "gender_preprocessed_neutral_gender.csv"
2. tf_idf_new(without_preprocessing).py -> creates for each of the three newly created files 5 new files, used later for both SVM and Logistic Regression:
   - f"{base_name}_tf_idf_sparse.csv"
   - f"{base_name}_tf_idf_sparse_matrix.npz"
   - f"{base_name}_vocabulary.csv"
   - f"{base_name}_document_labels.csv"
   - f"{base_name}_tfidf_vectorizer.pkl"
3. run svm_boris.py with:
   - input_file = "gender_preprocessed_neutral_gender.csv"

## Configuration
This section outlines the elements that can be adjusted to modify the experiment and explore alternative results.

1. **Dataset Variation**
   - The experiment can be repeated using a different gender-labeled dataset to assess the robustness of the models across datasets.  
   - To modify the dataset, changes need to be made in the file `"x"`.  

2. **Hyperparameter Values**
   - The fine-tuning of hyperparameter C in SVM can be adjusted to explore model performance further.  
   - Current C hyperparameter in SVM include values (0.1, 1, 10, 20, 50, 100)  
   - To modify the hyperparameters' values, changes need to be made in the file `"x"`.

3. **Model Hyperparameter Modification**
   - Beyond fine-tuning, the core hyperparameters themselves can be modified.  
   - For example, the **SVM kernel** can be changed:
     - Current: Linear  
     - Alternatives: Polynomial, RBF (Radial Basis Function)
   - To modify the hyperparameters or add more, changes need to be made in the file `"x"`.
  
## Extensions
Future improvements to this work could include testing the models on a different corpus to evaluate their robustness and generalizability. The range of models used can also be expanded, for example, by training the SVM with a non-linear kernel to explore more complex classifying boundaries. Additionally, in file `logistic_regression.py`, we present the top 10 words identified by the logistic regression model as most influential in classifying texts under male and female labels. In our current work, these word lists did not contain any gender-related terms. However, when applied to a different corpus, this aspect could be further investigated, and steps could be taken to gender-neutralize influential words in the dataset if necessary.
