def custom_output(custom_ner_model, pyspark_subset):
    from custom_ner_plugin import CustomNerModel
    from foundry_ml import Model, Stage
    
    output = custom_ner_model.transform(pyspark_subset[['id', 'msg_lower']])
    return output
