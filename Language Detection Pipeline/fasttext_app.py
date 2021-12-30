#Be sure to import text as a pandas dataframe.
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

    text['langue']= text['text'].astype(str).apply(model.predict)
    text['lang'] = text['langue'].str[0].str[0].str[9:]

    return text
