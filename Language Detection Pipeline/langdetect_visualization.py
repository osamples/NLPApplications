#Import langdetect as a pyspark dataframe. Use this to visualize the langdetect results.
def langdetect_visualization(langdetect):
    import pandas as pd
    import matplotlib.pyplot as plt
    from iso639 import languages
    langdetect['lang'] = langdetect['langue'].apply(lambda x: languages.get(alpha2=x).name)
    langdetect['lang'].value_counts().plot(kind='barh', figsize=(8, 6))
    plt.show()
    return langdetect
    
