"""----------------------------------------------- IMPORTS ----------------------------------------------------------"""

import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sentence_transformers import SentenceTransformer

""" --------------------------------------------- DESCRIPTION -------------------------------------------------------"""

"""

"""

"""---------------------------------------------- CONSTANTS ---------------------------------------------------------"""

CSV_FILE = 'transEmail1.csv'  # todo: replace to your own .csv file
NUMBER_OF_MAILS = 20  # todo: choose how many e-mails you want to show
EMBEDDER = 'paraphrase-mpnet-base-v2'  # todo: select an embedder: https://www.sbert.net/docs/pretrained_models.html

"""------------------------------------------- IMPLEMENTATION -------------------------------------------------------"""


def preprossess_emails_1():
    """

    :return:
    """
    # Select features from original dataset to form a new dataframe
    df1 = df[['Date', 'From Address', 'to Address', 'Subject', 'Body']]

    # For each row, combine all the columns into one column
    df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)

    # Store them in the pandas dataframe
    df_clean = pd.DataFrame({'clean': df2})

    # Create the list of list format of the custom corpus for gensim modeling
    return [row.split(',') for row in df_clean['clean']][:NUMBER_OF_MAILS]


def preprossess_emails_2(sent):
    """

    :param sent:
    :return:
    """
    final_sentences = []

    for lst in sent:
        final_sentences.append(" ".join(lst))

    return final_sentences


def get_address(address):
    """

    :param address:
    :return:
    """
    flag = False

    new_address = []
    for lt in address:

        if lt == ">":
            flag = False

        if flag:
            new_address.append(lt)

        if lt == "<":
            flag = True

    if len(new_address) == 0:
        new_address = address

    return "".join(new_address)


def perform_linkade(sent, link):
    """

    :param sent:
    :param link:
    :return:
    """
    labels = []
    for lst in sent:
        labels.append((lst[1], lst[3]))

    plt.figure(figsize=(8, 4))
    p = len(labels)

    r = shc.dendrogram(
                    link,
                    truncate_mode='lastp',  # show only the last p merged clusters
                    p=p,  # show only the last p merged clusters
                    no_plot=True,
                    )

    # create a label dictionary
    temp = {r["leaves"][ii]: labels[ii] for ii in range(len(r["leaves"]))}

    def llf(xx):
        return "{}".format(temp[xx])

    shc.dendrogram(
                link,
                truncate_mode='lastp',  # show only the last p merged clusters
                p=p,  # show only the last p merged clusters
                leaf_label_func=llf,
                leaf_rotation=90,
                leaf_font_size=8,
                show_contracted=True,  # to get a distribution impression in truncated branches
                )
    plt.show()


if __name__ == '__main__':

    # create an embedder
    embedder = SentenceTransformer(EMBEDDER)

    # read e-mails from file
    df = pd.read_csv(CSV_FILE)

    # prepossess emails
    sentensses = preprossess_emails_1()
    corpus = preprossess_emails_2(sentensses)

    # encode e-mails
    corpus_embeddings = embedder.encode(corpus)

    # create a linkage model
    linkage = shc.linkage(corpus_embeddings, 'ward', optimal_ordering=True)

    # Perform linkage
    perform_linkade(sentensses, linkage)
