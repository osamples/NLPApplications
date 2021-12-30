def ner_model():
    from ner_plugin import NerModel
    from foundry_ml import Model, Stage

    # pass in a spacy model with vectors
    model = NerModel('en_core_web_sm')

    return Model(Stage(model))
