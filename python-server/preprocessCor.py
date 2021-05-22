from string import punctuation
from nltk.corpus import stopwords
from pymystem3 import Mystem

stop_words = stopwords.words('russian')
mystem = Mystem()

#  Предобработка файлов для тренировки модели

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


filepath = '/home/runx/gisp-model/python-server/files/gisp.cor'
f = open('/home/runx/gisp-model/python-server/files/gispProcessed.cor', 'w')

with open(filepath) as fp:
    line = fp.readline()
    cnt = 1
    while line:
        print("Line {}: {}".format(cnt, line.strip()))
        f.write(preprocess_text(line) + "\n")
        line = fp.readline()
        cnt += 1
f.close()