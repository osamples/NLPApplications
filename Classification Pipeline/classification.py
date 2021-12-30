import spacy
import random
from spacy.util import minibatch, compounding
import pandas as pd
import dill
from foundry_ml.stage.flexible import stage_transform, register_stage_transform_for_class
from foundry_ml.stage.serialization import deserializer, serializer, register_serializer_for_class
from foundry_object.utils import safe_write_data, load_data


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NotSuspected":
                continue
            if score >= 0.7 and gold[label] >= 0.7:
                tp += 1.0
            elif score >= 0.7 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.7:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def custom_classifier(data, nlp):
    textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"})
    nlp.add_pipe(textcat, last=True)
    #add the labels associated with your classifier (should be 'LabelRow' in your prelabeled dataset)
    textcat.add_label("True")
    textcat.add_label("False")
    # Converting the dataframe into a list of tuples
    data['tuples'] = data.apply(lambda row: (row['msg_lower'], row['LabelRow']), axis=1)
    train = data['tuples'].tolist()

    def load_data(limit=0, split=0.8):
        train_data = train
        # Shuffle the data
        random.shuffle(train_data)
        texts, labels = zip(*train_data)
        # get the categories for each review
        cats = [{"True": bool(y), "False": not bool(y)} for y in labels]

        # Splitting the training and evaluation data
        split = int(len(train_data) * split)
        return (texts[:split], cats[:split]), (texts[split:], cats[split:])

    n_texts = 100000

    # Calling the load_data() function 
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(limit=n_texts)

    # Processing the final format of training data
    train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

    # ("Number of training iterations", "n", int))
    n_iter = 10

    # Disabling other components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()

        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))

        # Performing training
        for i in range(n_iter):
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                        losses = losses)

        # Calling the evaluate() function and printing the scores
            with textcat.model.use_params(optimizer.averages):
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  
                .format(losses['textcat'], scores['textcat_p'],
                        scores['textcat_r'], scores['textcat_f']))
    return nlp


# define wrapper class for foundry_ml registry
class ClassificationModel():

    # takes in loaded language model as input and custom tokenizer
    def __init__(self, model_name, train_data):
        self.spacy = spacy.load(model_name)
        self.spacy = custom_classifier(train_data, self.spacy)

    def predict(self, text):
        doc = self.spacy(text)
        results = doc.cats
        return results

    def predict_df(self, df):
        df["results"] = df["msg_lower"].apply(self.predict)
        results = pd.io.json.json_normalize(df['results']).reset_index()
        output = df.reset_index().merge(results, on='index')
        # now label as true or false past a certain threshold
        output['LabelTrue'] = output['True'].ge(.9)
        return output


# Annotate a function that will wrap the model and data passed between stages.
@stage_transform()
def _transform(model, df):
    return model.predict_df(df)


# Call this to send to foundry_ml Stage Registry, force=True to override any existing registered transform
register_stage_transform_for_class(ClassificationModel, _transform, force=True)


# Deserializer decorator
@deserializer("spacy_ner_model.dill", force=True)
def _deserializer(filesystem, path):
    # Loading pickled file
    return dill.loads(load_data(filesystem, path, True), encoding='latin1')


# Serializer decorator
@serializer(_deserializer)
def _serializer(filesystem, value):
    path = 'spacy_ner_model.dill'
    safe_write_data(filesystem, path, dill.dumps(value), base64_encode=True)
    return path


register_serializer_for_class(ClassificationModel, _serializer, force=True)
