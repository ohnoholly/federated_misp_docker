import pandas as pd
import time
import statistics
import wording_processing
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
import math
import numpy as np
import pickle
import argparse
import coloredlogs, logging


def Label_Encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    labels_class = le.classes_
    code = le.transform(labels)
    df = pd.DataFrame(code)
    return df, labels_class


def categorical_process(feature, pca_flag=True, index='', n_components=10):
    mlb = MultiLabelBinarizer()
    code = mlb.fit_transform(feature)
    list_class = mlb.classes_
    sum_freq = np.sum(code,axis=0)
    if pca_flag == True:
        pca = PCA(n_components=n_components)
        code = pca.fit_transform(code)
    if index:
        df = pd.DataFrame(code, index=index)
    else:
        df = pd.DataFrame(code)
    return df, list_class, sum_freq


def one_hot_encoder(feature, pca_flag=False, index='', n_components=10):
    feature = np.asarray(feature).reshape(-1, 1)
    enc = OneHotEncoder()
    code = enc.fit_transform(feature).toarray()
    list_class = list(enc.categories_)
    print(list_class)
    sum_freq = np.sum(code, axis=0)
    if pca_flag == True:
        pca = PCA(n_components=n_components)
        code = pca.fit_transform(code)
    df = pd.DataFrame(code)
    return df, list_class, sum_freq

def date_timestamp_convert(date_string):
    datetime_object = datetime.strptime(date_string, '%d %B %Y')
    timestamp = time.mktime(datetime_object.timetuple())
    return timestamp



def sample_generator(data_input):

    inputfile = pd.read_excel(data_input)

    # Vectorize the value
    corpus_value = inputfile['Atr_Value'].values.tolist()
    Instance_value_vectors = wording_processing.word_vectorization(corpus_value, 2, 0.1, 20)

    logging.info("The attribute categories extracted from the data:")
    # Encode the category of the attributes
    atr_category = inputfile['Category'].tolist()
    category_code, _, _ = one_hot_encoder(atr_category, pca_flag=True)

    logging.info("The attribute types extracted from the data:")
    # Encode the attribute type
    atr_type = inputfile['Atr_type'].tolist()
    type_code, _, _ = one_hot_encoder(atr_type, pca_flag=True)

    comments = []
    for idx, row in inputfile.iterrows():

        comment = str(row['Comment']) + ',' + row['Event_Name']
        if int(row['Is_Object']) == 1:
            comment = comment + row['Object_Name'] + ',' + row['Object_Description'] + ',' + row['Object_relation']

        comments.append(comment)

    logging.info("Starting vectorizing the text in the comments and other text data")
    Instance_comment_vectors = wording_processing.word_vectorization(comments, 2, 0.1, 20)

    data_extracted = []
    for idx, row in inputfile.iterrows():

        #Extract timestamps
        timestamp_created = row['Created_time']
        timestamp_modified = row['Last_modified']
        timestamp_peroid = row['Peroid']

        is_ids = 0
        deleted = 0
        disable_cor = 0
        if row['Is_IDS'] == 'TRUE':
            is_ids = 1
        if row['Deleted'] == 'TRUE':
            deleted = 1
        if row['Disable_Correlation'] == 'TRUE':
            disable_cor = 1


        vector_value = Instance_value_vectors.iloc[idx].to_list()
        vector_comment = Instance_comment_vectors.iloc[idx].to_list()

        code_category = category_code.iloc[idx]
        code_category = code_category.tolist()

        code_type = type_code.iloc[idx]
        code_type = code_type.tolist()

        event_category = row['Event_Category'] # Row information
        org_name = row['Organization'] # Row information
        event_name = row['Event_Name'] # Row information
        threat_level = int(row['Threat_level'])


        examples = code_category + code_type + [timestamp_created, timestamp_modified, timestamp_peroid, is_ids, deleted, disable_cor] + vector_value + vector_comment + [event_category, org_name, event_name, threat_level]
        data_extracted.append(examples)


    extract_df = pd.DataFrame(data_extracted)
    logging.info(str(extract_df.shape[0])+" data samples are completely generated...")
    logging.info("Each data sample consists of "+ str(extract_df.shape[1]-4)+" features.")

    return extract_df, inputfile
