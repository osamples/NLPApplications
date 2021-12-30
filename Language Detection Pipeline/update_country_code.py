def update_country_code(fasttext_app):
    import pandas as pd
    import matplotlib.pyplot as plt
    from iso639 import languages
    
    def get_lang(x):
        try:
            lang = languages.get(alpha2=x).name
        except:
            lang = x
        return lang

    fasttext_app['language'] = fasttext_app['lang'].apply(lambda x: get_lang(x))

    fasttext_app['language'].value_counts()[:10].plot(kind='barh', figsize=(8, 6))
    plt.show()  
    return fasttext_app
