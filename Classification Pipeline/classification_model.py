def classification_model(pyspark_subset):
    from classification_plugin import ClassificationModel
    from foundry_ml import Model, Stage

    # pass in a spacy model with vectors
    model = ClassificationModel('en_core_web_sm', pyspark_subset)

    return Model(Stage(model))
