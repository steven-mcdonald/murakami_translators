# Murakami Translators

![Murakami Reader Header](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/alva-pratt-a5ToDH34m0I-unsplash-crop.jpg)

Photo by [Alva Pratt](https://unsplash.com/@alvapratt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/japanese-reading?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Key Stages](#key-stages)
- [Results](#results)
- [Next Steps](#next-steps)

## Overview

As an avid reader of fiction, I have often wondered about the impact of the translator's style on a translated novel. In particular, I'd been curious about the works of Haruki Murakami. He writes in Japanese and has three main english language translators: Alfred Birnbaum, Jay Rubin and Philip Gabriel.  How much of what I was reading was Murakami and how much was the translator? Did the different translators have different takes on the original work? This seemed quite possible to me given the linguistic gulf between Japanese and English.

Could Machine Learning models trained to predict the translator of an unseen text help answer these questions? That is the aim of this project.

## Objectives

1. Create a machine learning model to investigate whether it is possible to predict which translator translated an unseen sample of text from one of Murakami's books.
2. If the model is successful, gain insights into the differences in the translator's styles through the model features.

## Key Stages

#### Data Loading

Import ebooks of Murakami's works from various formats (epub, docx, pdf) into standardised text files for analysis with Python.

Key libraries used were ebooklib, BeautifulSoup and textract.

- **Notebook:** [01_read_ebooks_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/01_read_ebooks_v01.ipynb)

#### Data Cleaning 

Clean and organise the texts. Split the book texts into small samples as datapoints. Sample set at ~ 1000 characters length. Text splitting on full stops to avoid partial sentences

- **Notebook:** [02_text_prep_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/02_text_prep_v01.ipynb)

#### Feature Engineering and EDA

Engineer features related to translation style while avoiding features related to the overall content or themes of the books. Explore the data and features.

Key libraries used were textacy, vaderSentiment, sklearn.

- **Notebooks:** [03_df_generation_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/03_df_generation_v01.ipynb), [04_df_EDA_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/04_df_EDA_v01.ipynb),  [05_df_additional_features_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/05_df_additional_features_v01.ipynb)

#### Modelling with Combinations of Features

Evaluate the effectiveness of logistic regression models generated using different features

using accuracy and confusion matrices as the main evaluation metrics to measure the success of the models.

- **Notebooks:** [06a_modelling_lreg_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06a_modelling_lreg_v01.ipynb)

#### Comparing Machine Learning Algorithms

Evaulate the effectiveness of alternative model algorithms to accurately predict the translator while providing insights into the importance of the features used in the predictions

- **Notebooks:** [06b_modelling_knn_dtree_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06b_modelling_knn_dtree_v01.ipynb), [06c_modelling_ensembles_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06c_modelling_ensembles_v01.ipynb), [06d_modelling_SVM_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06d_modelling_SVM_v01.ipynb), [06e_modelling_nnet_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06e_modelling_nnet_v01.ipynb), [07_score_analysis_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/07_score_analysis_v01.ipynb)

#### Investigating the most Confident Predictions

Checking the most confidently predicted chunks of text for each translator together with the most important features from the model to understand the differences in translation style

- **Notebooks:**[08_lreg_top_pred_analysis_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/08_lreg_top_pred_analysis_v01.ipynb)

#### Re-modelling with Dropped Features



#### Confirmation with an Alternative Test Set 

Re-run the most suitable model, de-selecting features that may relate to the 

- **Notebooks:**[09_lreg_same_text_check.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/09_lreg_same_text_check.ipynb)

## Results

- All models tested were able to predict the translator of a test dataset with accuracy above the baseline score of 0.40 (i.e. better than choosing the most common translator by default)

- Logistic Regression, SVM and XGBoost models all had 5-fold cross-validation accuracy above 0.67. Although XGBoost was the most accurate model, further analysis was continued using Logistic Regression due to its short run time and more easily interpretable features.

  ![model scores](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/model_cv_acc_comparison_01.png)

- Feature importance from the best Logistic Regression model indicate the following features to be the most important when differentiating each translator from the other two.

  **Alfred Birnbaum**:

  - more monosyllable words
  - shorter sentences
  - fewer verbs

  **Jay Rubin**:

  - longer sentences
  - more pronouns
  - frequency of the word 'had'

  **Philip Gabriel**:

  - fewer pronouns
  - more verbs
  - frequency of the word he

  The figure below helps summarise the values of some of these key features for the most confidently predicted chunks of text for each translator

  ![model scores](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/top_pred_key_coeffs_01.png)

- Some features were dropped before modelling as they were potentially related to the content  or page formatting such as punctuation.

- Two remaining features which were important in the predictions, namely frequency of the words 'had' and 'he' may also be more related to the tense and the narrative perspective. The modelling was therefore rerun without these features to ensure that the baseline could still be beaten.

- As the initial train/test split was performed at random there is the risk that the model is learning something of the style of the books themselves rather than the translation style. In order to rule this out, the model is re-run with the test set being only text where there is the same source material but different translators. 

## Next Steps

**Modelling**

- Further features e.g. ngrams, word order, sentence structure etc.
- Other models e.g. different neural network architectures 

**Topic Expansion**

- Clustering e.g. set 3 clusters - does data cluster on translator?
- Another author especially one book with several translators
- Network map of characters per book 
- Analyze web scrape of book reviews for opinions of translators
