#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program analyses data about film preferences 

@author: Laoise Beatty
"""

from os import listdir
from os.path import isfile, join
import sys
import datetime as dtt
import pathlib
import logging

import numpy as np
from numpy import linalg as LA  # norm
import matplotlib.pyplot as plt

# analysis stuff
import pandas as pd
from functools import wraps

from scipy.sparse.linalg import svds
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm

n_epoch = 300
np.random.seed(1234)


DATA_DIR = "./ml-100k/"
INFO_FILE = "u.info"
GENRE_FILE = "u.genre"
USER_FILE = "u.user"
ITEM_FILE = "u.item"
OCCUPATION_FILE = "u.occupation"
DATA_FILE = "u.data"
SMALL_DATA_FILE = "u5.test"


def readinfo():
    with open(DATA_DIR + INFO_FILE, "r") as f:
        # first line is number of users
        line = f.readline()
        # extract the number of users ie number before the space
        num_user = int(line.split(" ")[0])
        # next line is items
        line = f.readline()
        num_items = int(line.split(" ")[0])
        line = f.readline()
        num_ratings = int(line.split(" ")[0])
        return (num_user, num_items, num_ratings)
# test code
#num_user, num_items, num_ratings = readinfo()
#print(num_user, num_items, num_ratings)


def readusers():
    header_names = ['user id', 'age', 'gender', 'occupation', 'zip code']
    # the user file is not a tab separated list the items are seperted with a |
    df = pd.read_csv(DATA_DIR+USER_FILE, sep="|", error_bad_lines=False, encoding='ISO-8859-1', names=header_names)
    #print(df.head())
    return df


def readitems():
    header_names = ["movie id", "movie title", "release date", "video release date", "IMDb URL", "unknown", "Action", "Adventure", "Animation", "Children's",
                    "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    # the user file is not a tab separated list the items are seperted with a |
    # I have had to delete the last blank line from the dataset
    df = pd.read_csv(DATA_DIR+ITEM_FILE, sep="|", index_col=False, error_bad_lines=False, encoding='ISO-8859-1', names=header_names)
    #print(df.head())
    return df


def readdatageneric(filename):
    header_names = ['user id', 'item id',  'rating', 'timestamp']
    # the user file is not a tab separated list the items are seperted with a |
    # I have had to delete the last blank line from the dataset
    df = (pd.read_csv(filename, sep="\t", error_bad_lines=False, encoding='ISO-8859-1', names=header_names).astype({'user id': "float64", 'item id': "float64", 'rating': "float64", }))
    print(df.head())
    return df


def readdata():
    return readdatageneric(DATA_DIR+DATA_FILE)


def readsmall():
    return readdatageneric(DATA_DIR+SMALL_DATA_FILE)



########## fully taken from online so ok to keep
def matrix_factorization(R, P, Q, K, steps=20, alpha=0.0002, beta=0.02):
	'''
	R: ratings
	P: user features
	Q: items
	K: latent features
	steps: the number of steps 
	alpha: learning rate
	beta: regularization
    '''
	Q = Q.T
	RMSD = np.zeros([steps])

	for step in range(steps):
		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					# calculate error
					eij = R[i][j] - np.dot(P[i, :], Q[:, j])

					for k in range(K):
						# calculate gradient with a and beta parameter
						P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
						Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])

		eR = np.dot(P, Q)
		e = 0

		for i in range(len(R)):
			for j in range(len(R[i])):
				if R[i][j] > 0:
					e = e + pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
					for k in range(K):
						e = e + (beta/2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
		RMSD[step] = np.sqrt(e/len(R[R != 0]))
		# 0.001: local minimum
		if e < 0.001:
			break

	return P, Q.T, RMSD


def ratings_matrix(input_array):
	arr = np.array(input_array)
	len_1 = int(np.amax(np.array(input_array)[:, 1]))
	len_0 = int(np.amax(arr[:, 0]))
	#rows = len_0
	#cols = len_1
	ratings = np.zeros((len_0, len_1))

	for rows in range(len(arr[:, 0])):
		row = arr[rows]
		i = int(row[0])-1
		j = int(row[1])-1
		ratings[i, j] = int(row[2])

	return ratings


def similarity(Q, birthday, n):
	similarity_matrix = cosine_similarity(Q)
	similarity_row = similarity_matrix[birthday, :]
	movie_id = np.linspace(1, len(Q), num=len(Q))
	similarity_df = pd.DataFrame(list(zip(movie_id, similarity_row)), columns=["movie id", "Similarity"])

	return similarity_df.nlargest(n, 'Similarity').reset_index(drop=True)


def get_movie_title(factors_dataframe, item_dataframe):
	# matchs movie title with "movie id"/"item id"

	df = pd.merge(factors_dataframe, item_dataframe, on="movie id")[
            ['movie id', 'FACTOR_1', 'FACTOR_2', 'movie title']]

	return df


def movie(Q, dataframe, colum_name):

	movie_df = pd.DataFrame(columns=["FACTOR_1", "FACTOR_2"])

	factor_1 = []
	factor_2 = []
	movie_id = []

	for i in dataframe[str(colum_name)]:
		i = int(i)
		factor_1.append(Q[i-1, 0])
		factor_2.append(Q[i-1, 1])
		movie_id.append(i)

	movie_df = pd.DataFrame(list(zip(movie_id, factor_1, factor_2)), columns=["movie id", "factor 1", "factor 2"])

	return movie_df


test_df = readdata()

user_df = readusers()

item_df = readitems()

small_df = readsmall()

#small_test_data = load_test_data(BASE_FILEPATH + 'u6.test')


# Part  5
# ratting matrix

R = ratings_matrix(small_df)
#rating_matrix(small_test_data,user_df,item_data)
#DATA_NON_ZERO = len(data_set)
# SOLVING #
# just to be clear
N = len(R)  # rows : Number_Of_Users
print(N)
M = len(R[0])  # columns  : Number_Of_Movies
print(M)
K = 3  # Number of Latent Features

# Initialise P and Q
P = np.random.rand(N, K)
Q = np.random.rand(M, K)

# Calculate
nP, nQ, RMSE = matrix_factorization(R, P, Q, K, steps=500, alpha=0.0002)
print(nP, nQ)
# Get predicted matrix
nR = np.dot(nP, nQ.T)  # gives us our predictions for our rattings


# Part  6
Q1, sigma, Qtransp = svds(R, k=3)
nP_svd, nQ_svd, RMSE_svd = matrix_factorization(
	R, Q1, Qtransp.T, K, steps=500, alpha=0.0002)


plt.plot(RMSE_svd)
plt.plot(RMSE)



# Part  7

similarity_df = similarity(Q, 4, 20)
movies = movie(Q, similarity_df, 'movie id')
merged = pd.merge(movies, item_df, on="movie id")[
    ['movie id', 'factor 1', 'factor 2', 'movie title']]

# graph
def Plot_1_top_20(dataframe_Factors):

	fig, ax = plt.subplots(figsize=(10, 10), facecolor='w')

	for key, row in dataframe_Factors.iterrows():
		ax.scatter(row['factor 1'], row['factor 2'], color="orange", s=100)
		ax.annotate(row['movie title'], xy=(row['factor 1'], row['factor 2']))

	plt.xlabel('factor 1', size=30)
	plt.ylabel('factor 2', size=30)
	plt.xlim((-4, 4))
	plt.ylim((-2, 5))
	plt.savefig("Plot 1 top 20.png", dpi=800)
	plt.show()

	return 0


Plot_1_top_20(merged)


plt.show()
