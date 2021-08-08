"""----------------------------------------------- IMPORTS ----------------------------------------------------------"""

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

""" --------------------------------------------- DESCRIPTION -------------------------------------------------------"""

"""

"""

"""---------------------------------------------- CONSTANTS ---------------------------------------------------------"""

DIMENSIONS = 2

CSV_FILE = 'transEmail1.csv'  # todo: replace to your own .csv file
NUMBER_OF_MAILS = 20  # todo: choose how many e-mails you want to show
EMBEDDER = 'paraphrase-mpnet-base-v2'  # todo: select an embedder: https://www.sbert.net/docs/pretrained_models.html

# todo: choose how many clusters you want to perform
# todo: to choose an optimal k - uncomment optimal_k in the main function
# todo: notice: number of colors in COLORS list need to fit to K_MEAN_CLUSTERS
K_MEAN_CLUSTERS = 8
COLORS = ["gray", "blue", "purple", "orange", "black", "green", "pink", "yellow", "brown", "red"]


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


def k_mean(embeddings):
    """
    Perform k-mean clustering
    :return: k-mean labels
    """
    clustering_model = KMeans(n_clusters=K_MEAN_CLUSTERS)
    clustering_model.fit(embeddings)
    return clustering_model.labels_


def print_clusters(clusters, mails):
    """

    :param clusters:
    :param mails:
    :return:
    """
    clustered_sentences = [[] for j in range(K_MEAN_CLUSTERS)]
    for sentence_id, cluster_id in enumerate(clusters):
        clustered_sentences[cluster_id].append(mails[sentence_id])

    for i, cluster in enumerate(clustered_sentences):
        print("Cluster ", i, "len: ", len(cluster))
        print(cluster)
        print("")


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


def show_pca(plot_points, clusters, sent):
    """

    :param plot_points:
    :param clusters:
    :param sent:
    :return:
    """
    x_axis = [o[0] for o in plot_points]
    y_axis = [o[1] for o in plot_points]
    fig, ax = plt.subplots(figsize=(20, 10))

    ax.scatter(x_axis, y_axis, c=[COLORS[d] for d in clusters])

    from_address = []
    for lst in sent:
        from_address.append(get_address(lst[1]))

    for i, txt in enumerate(from_address):
        ax.annotate(txt, (x_axis[i], y_axis[i]))

    plt.show()


def optimal_k(plot_points):
    """

    :param plot_points:
    :return:
    """
    sse = []
    list_k = list(range(3, NUMBER_OF_MAILS))

    for k in list_k:
        km = KMeans(n_clusters=k, random_state=22)
        km.fit(plot_points)

        sse.append(km.inertia_)

    # Plot sse against k
    plt.figure(figsize=(6, 6))
    plt.plot(list_k, sse)
    plt.xlabel(r'Number of clusters *k*')
    plt.ylabel('Sum of squared distance')
    plt.show()


if __name__ == '__main__':

    # create an embedder and a pca model
    embedder = SentenceTransformer(EMBEDDER)
    pca = PCA(n_components=DIMENSIONS)

    # read e-mails from file
    df = pd.read_csv(CSV_FILE)

    # prepossess emails
    sentensses = preprossess_emails_1()
    corpus = preprossess_emails_2(sentensses)

    # encode e-mails
    corpus_embeddings = embedder.encode(corpus)

    # Perform k-mean clustering
    cluster_assignment = k_mean(corpus_embeddings)

    # todo: uncomment if you want to print clusters
    # print_clusters(cluster_assignment, corpus)

    # Perform pca process
    scatter_plot_points = pca.fit_transform(corpus_embeddings)

    # todo: uncomment if you want to see optimal k
    # optimal_k(scatter_plot_points)

    # show pca
    show_pca(scatter_plot_points, cluster_assignment, sentensses)
