import spacy
import pandas as pd
import dill
from foundry_ml.stage.flexible import stage_transform, register_stage_transform_for_class
from foundry_ml.stage.serialization import deserializer, serializer, register_serializer_for_class
from foundry_object.utils import safe_write_data, load_data

def ner(nlp):
    test = 'this a test to get ner entities like 14 years old and miami and december'
    doc = nlp(test)
    for ent in doc.ents:
        print(ent.text)
    return nlp

# define wrapper class for foundry_ml registry
class NerModel():

    # takes in loaded language model as input and custom tokenizer
    def __init__(self, model_name):
        self.spacy = spacy.load(model_name)
        self.spacy = ner(self.spacy)

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
register_stage_transform_for_class(NerModel, _transform, force=True)


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


register_serializer_for_class(NerModel, _serializer, force=True)
