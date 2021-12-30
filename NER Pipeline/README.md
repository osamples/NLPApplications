# NER Pipeline

We will define a customized NER pipeline that can be used for future use cases in Foundry.

#### Summary

Named Entity Recognition is a NLP method to find similar words or entities. Entities are the most important chunks of a particular sentence such as noun phrases, verb phrases, or both. Example entities can be Dates, Locations, Organizations, etc. Spacy has a pretrained NER model that works well for a set of entities that are predefined. We apply their pretrained model in our pipeline, but we also apply a customized model based on a training subset we define.

##### List of Entities Types:

General: Date, Person, Organization, Geopolitical Entity (GPE), Cardinal, Ordinal, Time, Nationalities or Religous or Political Greoups (NORP), FAC (Buildings, airports, highways, bridges, etc.), Quantity, Location, Money, Product, Law, Event, Work of Art, Language

Custom: Transportation (Uber, Lyft, Greyhound, etc.)

##### NER Model

Generally, Entity Detection algorithms are ensemble models of : Rule-based Parsing, Dictionary lookups, Part of Speech (POS) Tagging, and Dependency Parsing.

Specifically, Spacy's NER model can be detailed by the following algorithm definitions: 

Transition-based parsing is an approach to structured prediction where the task of predicting the structure is mapped to a series of state transitions. You might find this [tutorial](https://explosion.ai/blog/parsing-english-in-python) helpful for background information. The neural network state prediction model consists of either two or three subnetworks:​

- tok2vec: Map each token into a vector representation. This subnetwork is run once for each batch.

- lower: Construct a feature-specific vector for each (token, feature) pair. This is also run once for each batch. Constructing the state representation is then simply a matter of summing the component features and applying the non-linearity.​

- upper (optional): A feed-forward network that predicts scores from the state representation. If not present, the output from the lower model is used as action scores directly.

This model is similar to Classification and uses the same evaluation metrics such as Precision, Recall, and F1 Score.

##### Evaluation

The performance of these models are okay. They are essentially the base model, so there are ways to improve the model performance. Also, we can take the results steps further to better define classification models, alerting techniques, and further extraction.


## Palentir Foundry Application
Foundry has specific tools that can be used to deploy this model. I will detail the steps needed to do so.

### Library Creation

The first thing you will need to create is a library in foundry. We will do this for the general NER model and our custom model. Create a new repository in a shared folder - be sure to select 'Python Library' for repository type upon initiliazation. You can follow the steps detailed [here](https://hope.palantirfoundry.com/workspace/documentation/product/foundry-ml/tutorial-text-spacy) and follow all steps (including optional ones) detailed [here](https://hope.palantirfoundry.com/workspace/documentation/product/transforms/share-python-libraries) for proper configuration. 

Be sure to use `ner.py` as your general module and `custom_ner.py` as your custom model. (Note: Name your plugins `ner_plugin` and `custom_ner_plugin`.)

Once you have tagged the new libraries, you can now use them in other repositories and workbooks.

### Data Prep Workbook

The first workbook we will create is the data preparation pipeline. You do not need to customize this environment. Import your dataset. We will keep everything as a pyspark dataset in this workbook. Create a transform called `pyspark_cleaned` and use the corresponding code. Create a transform called `pyspark_subset` and use the corresponding code.  Save the `pyspark_cleaned` and `pyspark_subset` transforms as datasets. 

### Model Training Workbook

The next workbook we will create is the model training pipeline. You will need to customize your environment with the following packages:

- `ner-library` and `custom-ner-library` or whatever you named your custom libraries
- `spacy == 2.3.5`
- `spacy-model-en_core_web_sm == 2.3.1`

We will create the `ner_model` transform and paste the corresponding code. Run this transform and save as a dataset. You should see it return an object and model stages. Perform the same steps for `custom_ner_model`.

Now you can import the `pyspark_subset` dataset. Next, you can create the `ner_output` transform (importing the `pyspark_subset` as a pandas dataframe and the `ner_model` as an object) and paste the corresponding code. This script will take a while to run on a large dataset. You can test with a subset of the dataset. Perform the same steps for `custom_ner_model`.

Finally, you can explore the results for the ner model by creating a `metrics` transform and analyzing the log outputs. You can do the same for the custom ner model with a `custom_metrics` transform.

### Modeling Objective

Now that we have a working model, we can submit it to a Modeling Objective which is essentially a way to document this work in Foundry. Create a Modeling Objective, update the README.md, submit your model that should be found in the workbook-output folder of your Model Training workbook.  You can follow the steps detailed [here](https://hope.palantirfoundry.com/workspace/documentation/product/foundry-ml/overview) to implement any further documentation.



