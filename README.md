# Murakami Translators

![Murakami Reader Header](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/alva-pratt-a5ToDH34m0I-unsplash-crop.jpg)

Photo by [Alva Pratt](https://unsplash.com/@alvapratt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/japanese-reading?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## Overview

- Personal Project
- Capstone for General Assembly

## Objective

- Create a machine learning model to investigate whether it is possible to predict which translator translated an unseen sample of text from one of Murakami's books.
- If the model is successful, gain insights into the differences in the translator's styles through the model features.

## Key Steps

- Read ebooks from various formats (epub, docx, pdf) into standardised text files for analysis in python using ebooklib, BeautifulSoup and textract.
- Split the book texts into small samples for training and testing the model. Sample set at ~ 1000 characters length. Text splitting on full stops to avoid partial sentences
- Engineer features related to translation style while avoiding features related to the overall content or themes of the books
- Evaluate the effectiveness of models generated using different features
- Evaulate the effectiveness of different model algorithms to accurately predict the translator while providing insights into the importance of the features used in the predictions

## Results

- All models tested were able to predict the translator of a test dataset with accuracy above baseline (i.e. better than choosing the most common translator by default)

- Logistic Regression, SVM and XGBoost models all had cross validation accuracy above ...

- Feature importance from ... indicate the following

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

- Some features were dropped as they were potentially related to the content 

## Next steps

- Further features e.g. ngrams, word order, sentence structure etc.
- Other models e.g. different neural network architectures 
- Clustering e.g. set 3 clusters - does data cluster on translator?
- Another author especially one book with several translators
- Network map of characters per book
- Analyze web scrape of book reviews for opinions of translators