def metrics(ner_output):
    labels = ner_output.columns[4:]
    print(labels)
    for label in labels:
        df=ner_output[['msg_lower', label]]
        df=df.dropna(subset=[label])
        print(df)
        print(df[label].value_counts())
