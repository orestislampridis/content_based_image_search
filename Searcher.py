# import the necessary packages

import urllib.parse as urlparse

import numpy as np
import pandas.io.sql as sqlio
import psycopg2 as pq
from scipy.spatial.distance import cdist


def get_euclidean_distance(vectorA, vectorB):
    # Compute jaccard similarity and then jaccard distance
    vectorA = np.array(vectorA)
    vectorB = np.array(vectorB)
    distance = cdist(vectorA.reshape(1, -1), vectorB.reshape(1, -1), metric='euclidean').ravel()
    return distance


def get_canberra_distance(vectorA, vectorB):
    # Compute jaccard similarity and then jaccard distance
    vectorA = np.array(vectorA)
    vectorB = np.array(vectorB)
    distance = cdist(vectorA.reshape(1, -1), vectorB.reshape(1, -1), metric='canberra').ravel()
    return distance


def get_cosine_distance(vectorA, vectorB):
    # Compute jaccard similarity and then jaccard distance
    vectorA = np.array(vectorA)
    vectorB = np.array(vectorB)
    distance = cdist(vectorA.reshape(1, -1), vectorB.reshape(1, -1), metric='cosine').ravel()
    return distance


def get_jaccard_distance(vectorA, vectorB):
    # Compute jaccard similarity and then jaccard distance
    vectorA = np.array(vectorA)
    vectorB = np.array(vectorB)
    distance = cdist(vectorA.reshape(1, -1), vectorB.reshape(1, -1), metric='jaccard').ravel()
    return distance


def get_dice_distance(vectorA, vectorB):
    # Compute jaccard similarity and then jaccard distance
    vectorA = np.array(vectorA)
    vectorB = np.array(vectorB)
    distance = cdist(vectorA.reshape(1, -1), vectorB.reshape(1, -1), metric='dice').ravel()
    return distance


def get_chi2_distance(vectorA, vectorB, eps=1e-10):
    # compute the chi-squared distance
    distance = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(vectorA, vectorB)])
    return distance


def get_jensenshannon_distance(vectorA, vectorB):
    # Compute jaccard similarity and then jaccard distance
    vectorA = np.array(vectorA)
    vectorB = np.array(vectorB)
    distance = cdist(vectorA.reshape(1, -1), vectorB.reshape(1, -1), metric='jensenshannon').ravel()
    return distance


class Searcher:
    def __init__(self, query_features, method, distance, limit, database_url):
        # store the index of images
        self.query_features = query_features
        self.method = method
        self.distance = distance
        self.limit = limit
        self.database_url = database_url

        url = urlparse.urlparse(self.database_url)
        dbname = url.path[1:]
        user = url.username
        password = url.password
        host = url.hostname
        port = url.port

        try:
            cn = pq.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )

        except (Exception, pq.Error) as error:
            print("Error while connecting to PostgreSQL", error)

        cr = cn.cursor()
        sql = 'SELECT * FROM files;'
        cr.execute(sql)
        tmp = cr.fetchall()

        self.df = sqlio.read_sql_query(sql, cn)

    def search(self):
        # initialize our dictionary of results
        df = self.df

        results = {}

        id_list = df["orig_filename"].tolist()

        if self.method == "color":
            vector_list = df["color_descriptor"].tolist()
        elif self.method == "sift":
            vector_list = df["sift"].tolist()
        elif self.method == "surf":
            vector_list = df["surf"].tolist()
        elif self.method == "kaze":
            vector_list = df["kaze"].tolist()
        elif self.method == "orb":
            vector_list = df["orb"].tolist()

        vector_list_cleaned = [list(float(item) for item in t) for t in vector_list]
        dictionary = dict(zip(id_list, vector_list_cleaned))

        if self.distance == "euclidean":
            for (k, features) in dictionary.items():
                d = get_euclidean_distance(features, self.query_features)
                results[k] = d
        elif self.distance == "canberra":
            for (k, features) in dictionary.items():
                d = get_canberra_distance(features, self.query_features)
                results[k] = d
        elif self.distance == "cosine":
            for (k, features) in dictionary.items():
                d = get_cosine_distance(features, self.query_features)
                results[k] = d
        elif self.distance == "jaccard":
            for (k, features) in dictionary.items():
                d = get_jaccard_distance(features, self.query_features)
                results[k] = d
        elif self.distance == "dice":
            for (k, features) in dictionary.items():
                d = get_dice_distance(features, self.query_features)
                results[k] = d
        elif self.distance == "chi_squared":
            for (k, features) in dictionary.items():
                d = get_chi2_distance(features, self.query_features)
                results[k] = d
        elif self.distance == "jensenshannon":
            for (k, features) in dictionary.items():
                d = get_jensenshannon_distance(features, self.query_features)
                results[k] = d

        # sort our results, so that the smaller distances (i.e. the
        # more relevant images are at the front of the list)
        results = sorted([(v, k) for (k, v) in results.items()])

        # return our results
        print(results)
        return results[:self.limit]
