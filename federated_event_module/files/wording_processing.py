import nltk
import string
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import numpy as np
import word_2_vec

stop_words = set(stopwords.words('english') + list(string.punctuation) + ['citation'])

target_words = ['compromised', 'malware', ' ransomware', 'phishing', 'backdoor', 'iot', 'malicious', 'threat']


def word_vectorization(allsentences, epochs, learning_rate, dim):
    doc_vectors = []
    corpus_filiter = []
    for words in allsentences:
        #words = re.sub('http://\S+|https://\S+', '', words)
        #words = re.sub('http[s]?://\S+', '', words)
        #words = re.sub(r"http\S+", "", words)
        if isinstance(words, str):
            words_fil = [i for i in word_tokenize(words.lower()) if i not in stop_words]
        sentence_fil = ' '.join(map(str, words_fil))
        corpus_filiter.append(sentence_fil)

    word_to_index,index_to_word,corpus,vocab_size,length_of_corpus = word_2_vec.generate_dictinoary_data(corpus_filiter)
    window_size = 1
    training_data,training_sample_words = word_2_vec.generate_training_data(corpus,
                                            window_size,vocab_size,word_to_index,
                                            length_of_corpus,'yes')
    epochs = epochs
    learning_rate = learning_rate
    dim = dim
    epoch_loss,weights_1,weights_2 = word_2_vec.train(dim,window_size,epochs,
                                    vocab_size, training_data,learning_rate)

    for doc in corpus_filiter:
        words = doc.split()
        sum = np.zeros(dim)
        for word in words:
            index = word_to_index[word]
            word_vector = weights_1[epochs-1][index]
            sum = sum + word_vector
        doc_vector = sum/len(words)

        doc_vectors.append(doc_vector)

    df = pd.DataFrame(doc_vectors)

    return df


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
        words = sorted(list(set(words)))

    return words

def word_extraction(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text

def generate_bow(allsentences):
    vocab = tokenize(allsentences)
    print("Word List for Document \n{0} \n".format(vocab))

    for sentence in allsentences:
        words = word_extraction(sentence)
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i,word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1
                    #print("{0}\n{1}\n".format(sentence,np.array(bag_vector)))
        print(np.sum(bag_vector))


def word_token_count(sentences):
    words = [i for i in word_tokenize(sentences.lower()) if i not in stop_words]
    num_keywords = 0
    for word in words:
        if word in target_words:
            num_keywords = num_keywords + 1

    return num_keywords

def tfidf_cal(corpus):
    corpus_filiter = []
    for words in corpus:
        words_fil = [i for i in word_tokenize(words.lower()) if i not in stop_words]
        sentence_fil = ' '.join(map(str, words_fil))
        corpus_filiter.append(sentence_fil)

    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(corpus_filiter)
    df = pd.DataFrame(tfIdf[0].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
    df = df.sort_values('TF-IDF', ascending=False)
    #print (df.head(50))
