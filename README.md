# Murakami Translators

![Murakami Reader Header](https://github.com/steven-mcdonald/murakami_translators/blob/master/images/alva-pratt-a5ToDH34m0I-unsplash-crop.jpg)

Photo by [Alva Pratt](https://unsplash.com/@alvapratt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/japanese-reading?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

### Overview

- Personal Project
- Capstone for General Assembly

### Objective

- Create a machine learning model to investigate whether it is possible to predict which translator translated an unseen sample of text from one of Murakami's books
- If the model is successful, gain insights into the differences in the translator's styles through the model features

### Key Steps

- Read ebooks from various formats into standardised text files
- Split the book texts into small samples for training and testing the model
- Engineer features related to translation style while avoiding features related to the overall content or themes of the books
- Evaluate the effectiveness of models generated using different features
- Evaulate the effectiveness of different model algorithms to accurately predict the translator while providing insights into the importance of the features used in the predictions

### Next Steps

- All models tested were able to predict the translator of a test dataset with accuracy above baseline (i.e. better than choosing the most common translator by default)
- Logistic Regression, SVM and XGBoost models all had cross validation accuracy above ...
- Feature importance from ... indicate that
- Some features were dropped as they were potentially related to the content 