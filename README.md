![Murakami Title Image](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/Murakami_Title_Image.png)

# Murakami Translators Project

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Key Stages](#key-stages)
- [Summary and Conclusions](#summary-and-conclusions)
- [Possible Extensions](#possible-extensions)

------

## Overview

As an avid reader of fiction, I have often wondered about the impact of the translator's style on a translated novel. In particular, I've been curious about the works of Haruki Murakami. He writes in Japanese and has three main english language translators: Alfred Birnbaum, Jay Rubin and Philip Gabriel.  How much of what I was reading was Murakami and how much was the translator? Did the different translators have different takes on the original work? This seemed quite possible to me given the linguistic gulf between Japanese and English.

Could Machine Learning models trained to predict the translator of an unseen text help answer these questions? That is the aim of this project.

------

## Objectives

1. Create a machine learning model to investigate whether it is possible to predict which translator translated an unseen sample of text from one of Murakami's books.
2. If the model is successful, gain insights into the differences in the translator's styles through the model features.

------

## Key Stages

### Data Loading

Ebooks of Murakami's works in various formates (epub, docx, pdf) were imported into standardised text files for analysis with Python. The key libraries used were ebooklib, BeautifulSoup and textract.

[01_read_ebooks_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/01_read_ebooks_v01.ipynb)

### Data Cleaning 

As the original ebooks were in many different formats, the resulting text files were also in several forms from lists of lists to continuous strings. Each text was read in and split into chapters while extracting chapter numbers, chapter numbers and translators when suitable. 

Book chapters can vary significantly in length and with a dataset comprising of 7 books I only had just over two hundred chapters in total. We typically need a few thousand data points when performing machine learning on any dataset with more than a handful of features. Sentences and paragraphs can also vary significantly in length and therefore prove challenging to compare, while even pages can vary in length depending on the book formatting. I therefore decided to split the book texts into chunks set at ~1000 characters in length. The actual chunk lengths vary in size as they are split on full stops to avoid chunks containing partial sentences.

[02_text_prep_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/02_text_prep_v01.ipynb)

### Feature Engineering and EDA

Engineering suitable features was crucial to the project. Features needed to be related to translation style rather than the overall content or themes of the books. Features such as characters, locations or topics can be very useful when predicting a specific text. However, they are likely to be related to the underlying story and not to any decision made during the translation process.

- ##### Textacy Basic Counts

  The textacy feature, basic_counts, provides some statistics on a supplied text which could be related to the translation style such as the number of unique words and the numder of sentences in a given text.

- ##### POS Counts

  POS (Parts Of Speech) labels, again from textacy, were used to generate further features. Words were given POS labels such as adverbs, pronouns, adjectives etc. and then the counts of each POS type for a chunk of text were added as features.

- ##### Bag-of-Words

  Bag-of-Words analysis using sklearn's CountVectorizer was used to generate counts of each word in the dataset. The count per chunk was then divided by the total number of words used by each translator to get a measure of how common that word is for the translator. Counts of some general use words that showed variation between the different translators were added as features

- ##### Selected Adverbs and Adjectives

  It is possible that a given translator may favour a given adjective or adverb. By selecting only those words with POS labels for adjectives and adverbs further features were added based on counts for adverbs such as 'very' and 'really' as well as for adjectives such as 'small' and 'good'

- ##### Vader Sentiment Scores

  Vader sentiment scores which give values for positive, negative and neutral for a text indicating it's sentiment. It was considered likely that these features would potentially be more closely linked to the underlying themes of a book rather than a choice to change the sentiment by the translator. In any case, these features were tested initially to try to understand if they could play a role in the modelling

- ##### Count Normalisation

  One final important point for the feature engineering is that, as chunk length varied somewhat, counts per chunk were normalised to the equivalent of a 1000 character chunk. This was to reduce the length of the chunk itself having an impact. 

Once the features had been generated, some further EDA (Exploratory Data Analysis) was carried out to investigate the features. 

[03_df_generation_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/03_df_generation_v01.ipynb)

[04_df_EDA_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/04_df_EDA_v01.ipynb)

[05_df_additional_features_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/05_df_additional_features_v01.ipynb)

### Modelling with Combinations of Features

As several types of features were engineered, an initial set of regularised logistic regression models was generated. Each model having an additional feature type added to help understand their contribution to the predictions. 

Model hyperparameters were gridsearched and cross-validated accuracy was used to measure the predictive success of each model together with confusion matrices.

Logistic Regression scored well above the baseline accuracy of 0.40 (i.e. better than choosing the most common translator by default). 

![lreg scores](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/lreg_sel_feature_scores.png) 

We can see that the basic textacy counts and counts of POS types contribute significantly to the accuracy. Individual general word counts were particularly beneficial for predicting Philip Gabriel translations as we see the ROC-AUC score improve. Counts of some specific adverbs and adjectives also improved the predictive power slightly. On the other hand Vader sentiment scores did not significantly improve the accuracy and in fact had lower accuracy on the test set

This was a very positive outcome and meant that I could continue on to the second objective of gaining insights into the translator's styles from the model features. However, before that, other machine learning algorithms that could potentially generate even more accurate predictions were tested in the next step.

[06a_modelling_lreg_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06a_modelling_lreg_v01.ipynb)

### Comparing Machine Learning Algorithms

Several alternative models were tested and compared, from relatively simple algorithms such as KNN through to more complex ensembles methods and neural networks. 

The project objectives from the outset were to have a model that is both accurate and interpretable. Boosting, SVM and Multi-Layer Perceptron Neural Networks all generated greater accuracy in their predictions than Logistic Regression. However, they are less open to interpretation and have longer run times, in general. Therefore further analysis and interpretation was continued with the Logistic Regression model

![model scores](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/model_cv_acc_comparison_01.png)

[06b_modelling_knn_dtree_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06b_modelling_knn_dtree_v01.ipynb)

[06c_modelling_ensembles_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06c_modelling_ensembles_v01.ipynb)

[06d_modelling_SVM_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06d_modelling_SVM_v01.ipynb)

[06e_modelling_nnet_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/06e_modelling_nnet_v01.ipynb)

[07_score_analysis_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/07_score_analysis_v01.ipynb)

### Investigating the most Confident Predictions

Perhaps the most interesting stage of the project was reviewing the the most confidently predicted chunk of text for each translator together with the model features. From this I could gain some insights into the differences in the translator's styles  

![](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/top_lreg_coeffs_01.png)

The above figure indicates the most important features from the logistic regression model for determining each translator compared to the others. From this we can say that in general:

**Birnbaum:** More monosyllabic words | Shorter sentences | Fewer verbs

**Rubin:** Longer sentences | More pronouns | More use of the word 'had'

**Gabriel:** Fewer pronouns | More verbs | more use of the word 'he'

The figure below helps summarise the values of some of these key features for the most confidently predicted chunks of text for each translator

![model scores](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/top_pred_key_coeffs_01.png) 

[08_lreg_top_pred_analysis_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/08_lreg_top_pred_analysis_v01.ipynb)

### Re-modelling with Dropped Features

The top prediction analysis highlighted the risk of using seemingly very general words. From the initial modelling, **'had'** and **'he'** were strong predictors for Jay Rubin and Philip Gabriel respectively. When analysing the most confidently predicted chunks we can see that these features are more linked to the novels themselves than any significant translation style and we are overfitting to the training data. 

The novel 'Kafka on the Shore' translated by Philip Gabriel is written in the third person whereas many of Murakami's other novels are written in first person. The model is therefore more likely to be using the frequency of the word 'he' to predict the novel rather than the translator. The same goes for 'The Wind-Up Bird Chronicle', translated by Jay Rubin which contains a significant amount of text in the past tense and so the model is likely to be using the frequency of the word 'had' to predict the novel rather than the translator once again.

The logistic regression modelling was re-run without the features related to the frequency of the words 'had' and 'he'.

![](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/top_lreg_coeffs_02.png)

**Birnbaum:** Shorter sentences | Fewer verbs | More unique words

**Rubin:** Longer sentences | More pronouns | Fewer adverbs

**Gabriel:** Less use of the word 'said' | Fewer pronouns | More verbs 

This issue arises due to the initial randomised train/test split which takes sections of each book to train and other sections when testing the model. In the following section we can avoid this issue by setting aside complete books as test sets and training on the remaining books.

[09a_modelling_lreg_feature_drop_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/09a_modelling_lreg_feature_drop_v01.ipynb)

[09b_lreg_top_pred_analysis_v01.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/09b_lreg_top_pred_analysis_v01.ipynb)

### Confirmation with an Alternative Test Set 

As a final confirmation of the modelling approach I re-ran the Logistic Regression ( dropping the 'had' and 'he' count features) with a non-random train/test split.

The test set consisted of the text for which I had versions by two different translators, Jay Rubin and Alfred Birnbaum. The dataset contained several chapters of Norwegian Wood and one chapter of  'A Wind-Up Bird Chronicle' which had been translated twice, once by each translator. This provided a good control set as the  original Japanese text was the same and the only difference was the translator.

Two versions of this test ran. The first with all three translators in the training set but only the texts from Birnbaum and Rubin outlined above in the test set.

Fortunately, this model was still able to beat the baseline and confirmed that it was possible to predict the translator using the available features.

![](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/top_lreg_coeffs_rerun_01.png)

**Birnbaum:** Shorter sentences | Fewer verbs | More unique words

**Rubin:** Longer sentences | More pronouns | Fewer adverbs

**Gabriel:** Less use of the word 'said' | Fewer pronouns | More verbs 

As the test set consisted of only two translators, Birnbaum and Rubin, I ran a second version with only these two translators in the training set as well. The test accuracy was improved.

![](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/top_lreg_coeffs_rerun_01.png)

As there were only 2 target categories there was only one feature set in the model - those predicting Rubin:

**Rubin:** Longer sentences | More auxiliary words | Fewer characters | Fewer unique words



[09c_lreg_same_text_check.ipynb](https://github.com/steven-mcdonald/murakami_translators/blob/master/notebooks/09c_lreg_same_text_check.ipynb)

------

## Summary and Conclusions

### Objective 1: Model Generation

- The 7 available books were all successfully loaded and cleaned. A data sample was set as a chunk of text ~1000 characters long.
- Features that could be expected to relate to the translator style were extracted and tested with Logistic Regression. At this point, Vader sentiment scores were dropped from the modelling as they did not improve the accuracy scores.
- Further modelling algorithms were tested. Although some such as SVM and XGBoost performed slightly better on 5-fold cross-validation accuracy, they was more challenging to extract information on feature importance and so the Logistic Regression model was used for further analysis.
- **All models tested were able to predict the translator of a test dataset with accuracy above the baseline score of 0.40 (i.e. better than choosing the most common translator by default). the chosen Logistic Regression model had an accuracy score of 0.68**.

### Objective 2: Feature Interpretation

- A number of features which are not immediately apparent such as sentence, adverb and pronoun counts were shown to help distinguish the translators of these texts 
- Although the feature importance changed somewhat from model to model, some key features appeared regularly. In particular we can be quite confident on the features differentiating between Birnbaum and Rubin due to the final test set which had identical souce material for each of the two translators
- 
- The texts predicted to belong to each translator with the highest probabilities were assessed. From this, features potentially related to the underlying stories rather than translation style became apparent. These features were dropped and the modelling re-ran. The accuracy dropped slightly but was still well above baseline with an accuracy of 0.649 compared to 0.677 before they were dropped.
- As a final confirmation of the modelling approach a Logistic Regression model was generated with a non-random train/test split. Text which had been translated twice by different translators was used as the test set. In this way the potential for the model to fit to the story rather than the translator was avoided. The resulting model was still able to predict above baseline

------

## Possible Extensions

### **Modelling**

- Engineer further features e.g. based on ngrams, word order, sentence structure etc.
- Adjust the sample size e.g. try ~500 characters or ~2000 characters per chunk
- Test other models e.g. different neural network architectures 

### **Topic Expansion**

- Try unsupervised clustering e.g. set 3 clusters - does data cluster on translator?
- Try the same approach on another author, especially one book with several translators e.g. a 19th century classic that has been re-translated several times.
- Explore the content of the novels e.g. generate a network map of characters per book 
- Analyze web scraped book reviews for opinions of translators. Have reviewers noticed aspects of the translator's styles?
