def classfication_model_inference(classification_model, pyspark_cleaned):
    from classification_plugin import ClassificationModel
    from foundry_ml import Model, Stage
    
    output = classification_model.transform(pyspark_cleaned)
    return output
