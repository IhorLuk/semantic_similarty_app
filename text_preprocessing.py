import re
from cleantext import clean

full_regex = r'((?<=[^a-zA-Z0-9])(?:https?\:\/\/|[a-zA-Z0-9]{1,}\.{1}|\b)(?:\w{1,}\.{1}){1,5}(?:com|org|edu|gov|uk|net|ca|de|jp|fr|au|us|ru|ch|it|nl|se|no|es|mil|iq|io|ac|ly|sm){1}(?:\/[a-zA-Z0-9]{1,})*)'

# full clean text function
def clean_text(document):

    final_document = ""

    # replace links
    document = re.sub(r'http\S+', '', document)
    document = re.sub(r'https\S+', '', document)
    document = re.sub(full_regex, '', document)
    document = re.sub(r'www\S+', '', document)
    document = re.sub(r'(@[A-Za-z0–9_]+)', '', document)

    # remove emojis
    document = clean(document, fix_unicode=False, to_ascii=False,
                     lower=False, no_emoji=True, no_punct=False)

    # remove new lines
    document = re.sub(r'\n', '', document)

    document = document.split()
    useless_words = [':=:', ' More:']
    text_filtered = [word for word in document if not word in useless_words]

    final_document = ' '.join(text_filtered)

    final_document = final_document.strip()
    return final_document

