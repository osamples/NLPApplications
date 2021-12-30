def output(ner_model, pyspark_subset):
    from ner_plugin import NerModel
    from foundry_ml import Model, Stage
    
    output = ner_model.transform(pyspark_subset[['id', 'msg_lower']])
    return output
