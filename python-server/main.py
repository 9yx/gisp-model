import json
from string import punctuation
import gensim
import requests
from gensim.matutils import softcossim
from gensim import corpora
from gensim.models import WordEmbeddingSimilarityIndex
from nltk.corpus import stopwords
from pymystem3 import Mystem
from flask import Flask, jsonify, request

app = Flask(__name__)
mystem = Mystem()
stop_words = stopwords.words('russian')

def preprocess_text(text):
    tokens = mystem.lemmatize(text.lower()
                              .replace("#", " ")
                              .replace("!", " ")
                              .replace(".", " ")
                              .replace(".", " ")
                              .replace(",", " ")
                              .replace("  ", " "))
    tokens = [token for token in tokens if token not in stop_words
              and token != " "
              and token.strip() not in punctuation]

    text = " ".join(tokens)

    return text


# Получение меры сходства doc1 и doc2 используя обученную модель
@app.route("/softcossim", methods=['POST'])
def softcossimCall():
    data = request.json

    doc_1 = preprocess_text(data.get('doc1', '')).split()
    doc_2 = preprocess_text(data.get('doc2', '')).split()

    documents = [doc_1, doc_2]
    dictionary = corpora.Dictionary(documents)
    model = gensim.models.KeyedVectors.load('/home/runx/gisp-model/python-server/models/gisp.model')
    similarity_matrix = model.wv.similarity_matrix(dictionary)
    bow_1 = dictionary.doc2bow(doc_1)
    bow_2 = dictionary.doc2bow(doc_2)
    return jsonify(softcossim(bow_1, bow_2, similarity_matrix))


if __name__ == "__main__":
    app.run(host='0.0.0.0')
