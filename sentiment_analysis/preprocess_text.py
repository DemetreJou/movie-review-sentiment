import pandas as pd
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
import numpy as np

import html

import os
import re
import spacy


nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')

pool = Pool(cpu_count())  # TODO: does default Pool() use all cpu cores, is cpu_count() redundant?


def process_sentence(sentence):
    modComm = sentence

    # remove whitespace
    modComm = re.sub(r"\n{1,}", " ", modComm)
    modComm = re.sub(r"\t{1,}", " ", modComm)
    modComm = re.sub(r"\r{1,}", " ", modComm)
    modComm = re.sub(r"\r\n{1,}", " ", modComm)

    # unescape html
    modComm = html.unescape(modComm)

    # remove URLs
    modComm = re.sub(r"(http|www)\S+", "", modComm)

    # remove duplicate spaces
    modComm = re.sub(r" +", " ", modComm)

    # add Parts-of-Speech tags
    utt = nlp(modComm)
    modComm = ""
    for sentence in utt.sents:
        for token in sentence:
            if token.lemma_.startswith("-") and not token.text.startswith("-"):
                if token.text.isupper():
                    modComm += f"{token.text}/{token.tag_} "
                else:
                    modComm += f"{token.text.lower()}/{token.tag_} "

            elif token.text.isupper():
                modComm += f"{token.lemma_.upper()}/{token.tag_} "
            else:
                modComm += f"{token.lemma_.lower()}/{token.tag_} "

        # remove space at end of sentence
        if modComm != "":
            modComm = modComm[:-1]
        modComm += "\n"

    return modComm


def process_dataframe(df):
    df['Phrase'] = df['Phrase'].apply(lambda str: process_sentence(str))
    return df


def parallelize_dataframe(df, function, num_cores):
    df_shards = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(function, df_shards))
    pool.close()
    pool.join()
    return df


if __name__ == "__main__":
    raw_data = pd.read_csv(os.path.join("data", "data.csv"))
    print(raw_data.shape)
    # raw_data = raw_data[:1000]
    processed_data = parallelize_dataframe(raw_data, process_dataframe, 4)
    # TODO: remove index before saving
    processed_data.to_csv(os.path.join("data", "processed_data.csv"))
