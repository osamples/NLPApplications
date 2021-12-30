# Language Detection Pipeline

As a data scientist, you need a pipeline set in place that can easily and accurately identify languages being used in text. I will define the methodology and tools applied for our solution below.

## Base Model: FastText

There are many [pre-trained models](https://modelpredict.com/language-identification-survey) that can be used to identify different languages in text. We choose to use pre-trained models for this solution because we do not need it to be customized further (unlike our classification or NER models).

We were able to test two solutions in Foundry: LangDetect and FastText. We have the documentation for both, but in the end FastText is the better option. The reasons for choosing  FastText include the speed at which it runs in Foundry and its proven accuracy compared to other models.

FastText is built off an ensemple of supervized clusterization models. You can learn more about these models [here](https://fasttext.cc/docs/en/language-identification.html).

## Foundry Implementation

To implement FastText, you must first download this bin file: [`lid.176.bin`](https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin). Follow the next steps when uploading this file to Foundry. In order to access files from a transform, they need to be part of a dataset, not a blob upload. You can upload it correctly by either creating an empty dataset -> upload file, or upload directly into a folder and selecting “upload as dataset.” Following the instructions [here](https://hope.palantirfoundry.com/workspace/documentation/product/code-workbook/python-raw-file-access), from Code Workbooks you can import the file as a dataset. Next, create a transform that imports the file as a `Transform input`. You'll also want to import the text data you have available (as a pandas dataframe). Paste the code from `fasttext_app.py` and run the transform. You will very quickly get the results for language detection.

```py
#fasttext_app.py code snippet
def fasttext_app(lid_176_bin, text):
    import tempfile
    import shutil

    fs = lid_176_bin.filesystem()
    with fs.open("lid.176.bin", "rb") as f:
        with tempfile.TemporaryDirectory() as local_dir:  # create a temporary local directory
            tmp_filename = local_dir + "/lid.176.bin"
            with open(tmp_filename, "wb") as tmp_file:  # copy the dataset file into the local directory
                shutil.copyfileobj(f, tmp_file)
            model = fasttext.load_model(tmp_filename)
            print('it worked')

    text['langue']= text['text'].astype(str).apply(model.predict)
    text['lang'] = text['langue'].str[0].str[0].str[9:]

    return text
```


From there, you can look at some metrics if you'd like by running the `update_country_code.py` transform. 


You can also look at the results from LangDetect, but it does not perform well on the large data provided. In fact, you have to test on a subset in Foundry. That is why FastText is our optimal solution.

## Next Steps
This pipeline can be applied to all future language detection needs. Please let me know if you have any questions.

