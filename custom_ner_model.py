def custom_ner_model():
    from custom_ner_plugin import CustomNerModel
    from foundry_ml import Model, Stage

    # pass in a spacy model with vectors
    model = CustomNerModel('en')

    return Model(Stage(model))
