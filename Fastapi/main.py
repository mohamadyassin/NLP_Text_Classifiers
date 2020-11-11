from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

# Load Spacy model
import spacy
nlp = spacy.load('en_core_web_md')

# Disable the compenents
nlp.disable_pipes('tagger', 'parser', 'ner')


# Initialize the matcher
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# Define pattern and add to matcher
pattern = nlp("table of contents")
matcher.add('TOC', None, pattern)

# Load the featurizer
from joblib import load
featurizer_rf = load('tfidf_rf.joblib')


# Load the classifier
from joblib import load
rf_clf = load('rf_clf.joblib')


class Item(BaseModel):
    text: list



app = FastAPI()


@app.post("/items")
async def function_name(input_texts: Item, limit: str = 10000):
    new_texts = find_pages(input_texts.text)
    response_item = Item(text=new_texts)
    return response_item





def find_pages(texts):
        
    # Use lower case
    texts = [doc.lower() for doc in texts]
    
    # Find the index
    toc_index = []
    for index, text in enumerate(texts):
        if matcher(nlp(text)) != []:
            toc_index.append(index)
    
    # Delete the toc
    texts = [text for index, text in enumerate(texts) if index not in toc_index]
        
    # Create features out of raw texts
    import pandas as pd
    X = featurizer_rf.fit_transform(texts)
    X = X.toarray()
    X = pd.DataFrame(X, columns=featurizer_rf.get_feature_names())
    
        
    # Define predicted labels
    y_pred_rf = list(rf_clf.predict(X))
    
    # Find the index
    conf_index = []
    for index, prediction in enumerate(y_pred_rf):
            if prediction == 2:
                conf_index.append(index)
    
    # Delete conf pages
    texts = [text for index, text in enumerate(texts) if index not in conf_index]
    
    return texts
