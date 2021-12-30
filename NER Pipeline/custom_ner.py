import random
import spacy
from tqdm import tqdm
import pandas as pd
import dill
from foundry_ml.stage.flexible import stage_transform, register_stage_transform_for_class
from foundry_ml.stage.serialization import deserializer, serializer, register_serializer_for_class
from foundry_object.utils import safe_write_data, load_data
### removing train_data for confidentiality but provided example syntax. provide as much train data as possible
TRAIN_DATA = [
    ('this is uber', {
        'entities': [(9, 12, 'TRANSPORT')]
    })
]


def custom_ner(nlp, train_data = TRAIN_DATA, n_iter = 100):
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            for text, annotations in tqdm(train_data):
                nlp.update(
                    [text],
                    [annotations],
                    drop=0.15,
                    sgd=optimizer,
                    losses=losses)
            print(losses)

    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    return nlp


# define wrapper class for foundry_ml registry
class CustomNerModel():

    # takes in loaded language model as input and custom tokenizer
    def __init__(self, model_name):
        self.spacy = spacy.blank(model_name)
        self.spacy = custom_ner(self.spacy)

    def predict(self, text):
        result = {}
        doc = self.spacy(text)
        for ent in doc.ents:
            result[ent.label_] = ent.text
        return result

    def predict_df(self, df):
        df["Matches"] = df["msg_lower"].apply(self.predict)
        matches = pd.io.json.json_normalize(df.Matches).astype(str)
        final = df.reset_index().merge(matches.reset_index(), on='index', how='left')
        return final


# Annotate a function that will wrap the model and data passed between stages.
@stage_transform()
def _transform(model, df):
    return model.predict_df(df)


# Call this to send to foundry_ml Stage Registry, force=True to override any existing registered transform
register_stage_transform_for_class(CustomNerModel, _transform, force=True)


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


register_serializer_for_class(CustomNerModel, _serializer, force=True)
