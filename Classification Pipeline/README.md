# Classification Pipeline

This repository describes the foundry pipeline for applying text classification.

## Text Classification
Using our classification library, we apply our classification model on a cleaned subset for training to get predicted results for an entire dataset.

#### The pipeline is as follows:

1. Data cleaning
2. Data sub setting
3. Model Training
4. Model Application
5. Evaluation

#### Custom Classification Model:
Now we will detail our custom classification model in spaCy and how it works.  SpaCy is an advanced library for performing NLP tasks like classification - you can easily customize it to your specific use case.

Rather than using the base classification tool that spaCy offers, we will firstly train it on a relevant labeled dataset so that it will perform well for our purposes - making it a supervised machine learning application. 

TextCategorizer is the optional and trainable pipeline component for spaCy that we use. A convolutional neural network architecture lies behind spaCy's Text Categorizer.  A [CNN](https://medium.com/voice-tech-podcast/text-classification-using-cnn-9ade8155dfb9) architecture performs better than a [Naive Bayes](https://iq.opengenus.org/text-classification-naive-bayes/) or a [K Nearest Neighbors](https://iq.opengenus.org/text-classification-using-k-nearest-neighbors/) model because it considers the similarity of words rather than just the term frequency. (Note: In the future, a model architecture that can perform better than CNN is [Hierarchical Attention Networks](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf). It is able to take into account further context. This would need to be applied independently from spaCy.)

When we train the model with our prelabeled dataset, we shuffle the prelabeled set and separate it into a training (80%) and testing (20%) set. 

For testing purposes, we create our own evaluation function as well. As far as metrics, the function will calculate the values of true positive, true negative, false positive, and false negative. It will use those values to calculate precision, recall and the f1 score. 

Precision tells us the accuracy of the model (percentage of positive predications are true), recall tells us the sensitivity of the model (percentage of actual positives are predicted true). We want a better precision score when we want to lower the existence of false positives. We want a better recall score when we want to lower the existence of false negatives. For our particular use case, we want a better recall/sensitivity score. The F1 score combines precision and recall and helps to seek a balance between the two. It is particularly useful when there is a large number of actual negatives -- an uneven class distribution, which is the case for us.  For all three of these metrics, the closer the score to 1, the better. 

Now that we have an evaluation function, we can use it while training the model to improve its performance. We will loop over the training examples and partition them into batches using spaCy's minibatch and computing helpers. Through each iteration, the model is evaluated and then updated through the nlp.update() command. This method allows us to improve the Precision, Recall, and F1 scores as we train.

Once we have the trained model, we can apply it to new data to get classification scores.


## Foundry Application
Foundry has specific tools that can be used to deploy this model. I will detail the steps needed to do so.

### Library Creation

The first thing you will need to create is a custom classification library in foundry. Create a new repository in a shared folder - be sure to select 'Python Library' for repository type upon initiliazation. You can follow the steps detailed [here](https://hope.palantirfoundry.com/workspace/documentation/product/foundry-ml/tutorial-text-spacy) and follow all steps (including optional ones) detailed [here](https://hope.palantirfoundry.com/workspace/documentation/product/transforms/share-python-libraries) for proper configuration. 

Be sure to use `classification.py` as your module. (Note: Name your plugin `classification_plugin`.)

Once you have tagged the new library, you can now use it in other repositories and workbooks.

### Data Prep Workbook

The first workbook you will create is the data preparation pipeline. You do not need to customize this environment. Import your dataset. We will keep everything as a pyspark dataset in this workbook. In this code workbook you will need to clean your data (remove punctuation, conver to lower case, etc.) and pre-label the trianing dataset. I removed my code here for confidentiality. 

### Model Training Workbook

The next workbook we will create is the model training pipeline. You will need to customize your environment with the following packages:

- `classification-custom-library` or whatever you named your custom library
- `spacy == 2.3.5`
- `spacy-model-en_core_web_sm == 2.3.1`

You will also need to add `global_code.py` to the Global Code section. 

Now you can import the prelabeled datset and full dataset that has been cleaned. We will create the `classification_model` transform (importing the prelabeled dataset as a pandas dataframe) and paste the corresponding code. Run this transform and save as a dataset. You should see it return an object and model stages. 

Next, you can create the `classification_model_inference` transform (importing the full dataset that has been cleaned as a pandas dataframe and the `classification_model` as an object) and paste the corresponding code. This script will take a while to run on a large dataset. You can test with a subset of the dataset.

### Modeling Objective

Now that we have a working model, we can submit it to a Modeling Objective which is essentially a way to document this work in Foundry. Create a Modeling Objective, update the README.md, submit your model that should be found in the workbook-output folder of your Model Training workbook.  You can follow the steps detailed [here](https://hope.palantirfoundry.com/workspace/documentation/product/foundry-ml/overview) to implement any further documentation.



