# NLP Applications
Natural Language Processing Applications and Architectural Solutions

## Background
My work with Human Rights Issues has opened up a world of text problems, many of which can be explored using Natural Language Processsing. I will detail multiple techniques I have explored along with the enablement solutions I architected within [Palantir Foundry](https://www.palantir.com/platforms/foundry/). 

## Natural Language Processing
NLP is concerned with giving computers the ability to understand text and spoken words in much the same way human beings can. Text data can be looked at as a pool of words and phrases. NLP attempts to structure this unstructured data.

### NLP Named Entity Recognition (NER) 

Named Entity Recognition is a methodology that extracts entities (significant words and phrases) from the text data and can categorize them based off similar words. This methodology can be applied with many different algorithms, many of which are already managed solutions. These include [SpaCy NER](https://stackoverflow.com/questions/60381170/which-deep-learning-algorithm-does-spacy-uses-when-we-train-custom-model), [NLTK](https://www.nltk.org/book/ch07.html), [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.shtml), and more. 

I apply the SpaCy NER solution which is a neural network state prediction model that consists of two or three subnetworks: tok2vec, lower, upper. You can also customize this model to extract word groups you are particularly interested in. I explain the implementation of the generic model and the customized model in my `NER Pipeline` documentation.

### NLP Classification
NLP's classification technique categorizes whole texts into different categories. This is a supervised learning technique that takes a training subset to determine what category a text might fall under. This methodology can be built off of many different algorithms such as Naive Bayes, KNN, Convolutional Neural Networks, and Hierarchical Attention Networks. I chose a CNN based managed solution within spaCy called `TextCategorizer`. You can learn more of how I applied this in my `Classification Pipeline` documentation.

### NLP Topic Modeling
Topic Modeling can be considered the reverse engineering approach to Classification. In classification, you know what topics you are looking for and trying to categorize them. In topic modeling, you are discovering all the different topics mentioned in the text you may have never known. This methodology can be built off of the following [algorithms](https://iq.opengenus.org/topic-modelling-techniques/):
  - Latent Dirichlet Allocation (LDA)
  - Non Negative Matrix Factorization (NMF)
  - Latent Semantic Analysis (LSA)
  - Parallel Latent Dirichlet Allocation (PLDA)
  - Pachinko Allocation Model (PAM)

I have applied this using [Gensim](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#6.-What-is-the-Dominant-topic-and-its-percentage-contribution-in-each-document) which is a managed solution with an LDA approach. Because I use this as an exploratory solution for now, I cannot provide a developed pipeline in documentation. Please see the Gensim link if you'd like to apply this technique yourself.
 
### Fuzzy Classification or Deduplication
Typos and different spellings are human errors that can negatively effect the usefulness of your text data. In a situation where you have a database of names, you may have duplicate records without knowing it because of these human errors. The solution here is fuzzy matching and more advanced deduplication techniques.
 
Now the most simple solution is [fuzzywuzzy](https://pypi.org/project/fuzzywuzzy/), but this solution's time increases exponentially based on data size. It compares every single row based off mathematical differences. This works well for small solutions; however, for large databases, this is impossible to implement. 
 
To account for this, you can use machine learning to reduce this time significantly. In fact, I have implemented this solution with a previously developed ensemble algorithm called [dedupe](https://dedupe.io/documentation/how-it-works.html) that consists of regularized logistic regression to get pair probabilities and hierarchical clustering with centroid linkage to group sets of pairs. This is a supervised model where you provide example pairs and it returns predicted pairs on the dataset as a whole in record time. 

You can apply this to your own solution by following these [examples](https://github.com/dedupeio/dedupe-examples).

While dedupe was available in [pypi](https://pypi.org/project/dedupe/), it was not available in conda-forge until I uploaded and now [manage it](https://anaconda.org/conda-forge/dedupe) and its dependencies. This was necessary to enable this solution in Palantir Foundry.
 
 ### Language Detection
 Another scenario you come across with text data is different languages and needing to identify them. I created a pipeline that can easily and accurately identify langauges being used in text based off a previously trained model called FastText. There are many [pre-trained models](https://modelpredict.com/language-identification-survey) that can be used to identify different languages in text. We choose to use pre-trained models for this solution because we do not need it to be customized further (unlike our classification or NER models).
 
 We were able to test two solutions in Foundry: LangDetect and FastText. We have the documentation for both, but in the end FastText is the better option. The reasons for choosing  FastText include the speed at which it runs in Foundry and its proven accuracy compared to other models.
FastText is built off an ensemble of supervised clusterization models. You can learn more about these models [here](https://fasttext.cc/docs/en/language-identification.html).

You can also learn more about how I implemented this solution in my `Language Detection Pipeline` documentation.

## Conclusion
We have discussed a vast amount of NLP solutions that anyone can begin to apply to their text solutions. In the end, there are many NLP solutions to explore even further, and additional techniques to improve what you've already implemented. Keep in mind the priorities of your projects to choose the best solution for you. 
