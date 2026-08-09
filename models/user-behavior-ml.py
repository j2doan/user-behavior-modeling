#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import math
import scipy.optimize
import numpy as np
import string
from sklearn import linear_model
import random
import gzip
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import implicit
from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix


random.seed(42)
np.random.seed(42)

def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0





##################################################
# Rating prediction                              #
##################################################

# ---------- IGNORE, NOT HIGH ENOUGH ACC ----------
# Idea: gradient descent
# θ = θ + learning_rate ⋅ (error ⋅ grad − reg ⋅ θ)
# theta being either the about to be new gamma_u or gamma_i
# repeat steps until convergence (epoch)

# I AM NOT REBUILDING THE DICTIONARY EACH TIME
# def preprocess_interactions(train_interactions):
#     users = {}
#     items = {}

#     user_index = 0
#     item_index = 0

#     indexed_data = []

#     for (u, b, r) in train_interactions:
#         if u not in users:
#             users[u] = user_index
#             user_index += 1
#         if b not in items:
#             items[b] = item_index
#             item_index += 1

#         indexed_data.append((users[u], items[b], r))

#     return users, items, indexed_data

# def rating_grad_descent(indexedTrain, indexedValid, num_users, num_items, num_latent_factors=10, learning_rate=0.01, regularizer=0.1, epochs=10):
#     # NOW WITH LATENT FACTORS
#     alpha = np.mean([r for (_, _, r) in indexedTrain])
#     betaU = np.zeros(num_users)
#     betaI = np.zeros(num_items)

#     gammaU = 0.01 * np.random.randn(num_users, num_latent_factors)
#     gammaI = 0.01 * np.random.randn(num_items, num_latent_factors)

#     # TIME FOR GRADIENT DESCENT
#     for step in range(epochs):

#         random.shuffle(indexedTrain)

#         for (u, b, r) in indexedTrain:
#             pred = alpha + betaU[u] + betaI[b] + np.dot(gammaU[u], gammaI[b])
#             error = r - pred

#             alpha += learning_rate * error
#             betaU[u] += learning_rate * (error - regularizer * betaU[u])
#             betaI[b] += learning_rate * (error - regularizer * betaI[b])

#             gu = gammaU[u].copy()
#             gi = gammaI[b].copy()

#             gammaU[u] += learning_rate * (error * gi - regularizer * gu)
#             gammaI[b] += learning_rate * (error * gu - regularizer * gi)

#         # VALIDATE
#         if len(indexedValid) > 0:
#             mse = 0
#             for (u, b, r) in indexedValid:
#                 pred = alpha + betaU[u] + betaI[b] + np.dot(gammaU[u], gammaI[b])
#                 mse += (r - pred) ** 2
#             mse /= len(indexedValid)

#             print(f"Epoch {step+1}/{epochs}. Validation MSE = {mse:.4f}")

#     return alpha, betaU, betaI, gammaU, gammaI

# def writePredictionsRating(alpha, betaU, betaI, gammaU, gammaI):
#     # Write your predictions to a file that you can submit
#     predictions = open("predictions_Rating.csv", 'w')
#     for l in open("pairs_Rating.csv"):
#         if l.startswith("userID"):
#             predictions.write(l)
#             continue
#         u,b = l.strip().split(',')
#         u_idx = users.get(u, None)
#         b_idx = items.get(b, None)

#         if u_idx is None or b_idx is None:
#             pred = alpha
#         else:
#             pred = alpha + betaU[u_idx] + betaI[b_idx] + np.dot(gammaU[u_idx], gammaI[b_idx])

#         _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

#     predictions.close()
# ---------- IGNORE, NOT HIGH ENOUGH ACC ----------

# IDEA: HW3 SOLUTION, but higher regularizer lambda and more epoch steps

def getGlobalAverage(trainRatings):
    # Return the average rating in the training set
    total_ratings = 0
    count = 0
    for x in trainRatings:
        rating = x[2] if isinstance(x, (list,tuple)) else x # could be a tuple or raw rating
        total_ratings += rating
        count += 1
    if count > 0:
        return total_ratings / count
    else:
        return 0



def trivialValidMSE(ratingsValid, globalAverage):
    # Compute and return the MSE of a trivial model that always returns the global mean computed above
    total_squared_error = 0.0
    count = 0
    for x in ratingsValid:
        rating = x[2] if isinstance(x, (list,tuple)) else x
        total_squared_error += (rating - globalAverage) ** 2
        count += 1
    if count > 0:
        return total_squared_error / count
    else:
        return 0


def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    # Update equation for alpha
    residual_sum = 0
    count = 0
    for user, item, rate in ratingsTrain:
        b_u = 0
        b_i = 0
        if user in betaU:
            b_u = betaU[user]
        if item in betaI:
            b_i = betaI[item]
        residual_sum += rate - b_u - b_i
        count += 1
    newAlpha = 0
    if count > 0:
        newAlpha = residual_sum / count
    return newAlpha



def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    # Update equation for betaU
    newBetaU = {}
    for user in ratingsPerUser:
        numer = 0
        denom = lamb + len(ratingsPerUser[user])
        for (item, rate) in ratingsPerUser[user]:
            numer += rate - alpha - betaI.get(item, 0)
        if denom > 0:
            newBetaU[user] = numer / denom
        else:
            newBetaU[user] = 0
    return newBetaU



def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    # Update equation for betaI
    newBetaI = {}
    for item in ratingsPerItem:
        numer = 0
        denom = lamb + len(ratingsPerItem[item])
        for (user, rate) in ratingsPerItem[item]:
            numer += rate - alpha - betaU.get(user, 0)
        if denom > 0:
            newBetaI[item] = numer / denom
        else:
            newBetaI[item] = 0
    return newBetaI



def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    # Compute the MSE and the mse+regularization term
    total_squared_error = 0
    count = 0
    for user, item, rate in ratingsTrain:
        b_u = 0
        b_i = 0
        if user in betaU:
            b_u = betaU[user]
        if item in betaI:
            b_i = betaI[item]
        pred = alpha + b_u + b_i
        err = rate - pred
        total_squared_error += err * err
        count += 1
    if count == 0:
        return 0, 0
    mse = total_squared_error / count
    regularizer = 0
    for u in betaU:
        v = betaU[u]
        regularizer += v * v
    for i in betaI:
        v = betaI[i]
        regularizer += v * v
    return mse, mse + lamb*regularizer



def validMSE(ratingsValid, alpha, betaU, betaI):
    # Compute the MSE on the validation set
    total_squared_error = 0
    count = 0
    for user, item, rate in ratingsValid:
        pred = alpha + betaU.get(user, 0) + betaI.get(item, 0)
        total_squared_error += (rate - pred) ** 2
        count += 1
    if count > 0:
        validMSE = total_squared_error / count
    else:
        return 0
    return validMSE



def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    # Improve upon your model from the previous question (e.g. by running multiple iterations)
    lamb = 4
    alpha = alpha
    betaU = dict(betaU)
    betaI = dict(betaI)
    for _ in range(18):
        alpha  = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
        mse, mse_reg = msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb)
        print(f"Iteration {_ + 1}: Training MSE = {mse:.4f}, MSE + Reg = {mse_reg:.4f}")
    return alpha, betaU, betaI



def writePredictionsRating(alpha, betaU, betaI):
    # Write your predictions to a file that you can submit
    predictions = open("predictions_Rating.csv", 'w')
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        bu = 0
        bi = 0
        if u in betaU:
            bu = betaU[u]
        if b in betaI:
            bi = betaI[b]
        _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

    predictions.close()







##################################################
# Read prediction                                #
##################################################

# ---------- IGNORE THIS, FAILED ATTEMPT ----------
# IDEA: bayesian personalized ranking
# r_ui = observed preference
# 1 if the user read the book, 0 otherwise/we dont know
# 0 NOT a real negative, its just unknown
# c_ui = confidence weight
# assign HIGH weight (1) to positive
# # assign LOW weight (0.01) to negative, because
# we can't guarantee rule out they won't like it

# def build_global_index_maps(interactions):
#     users = {}
#     items = {}

#     for (u, b, r) in interactions:
#         if u not in users:
#             users[u] = len(users)
#         if b not in items:
#             items[b] = len(items)

#     return users, items

# def build_matrix(interactions, users, items):
#     row_idx, col_idx, data = [], [], []

#     for (u, b, r) in interactions:
#         if u in users and b in items:
#             row_idx.append(users[u])
#             col_idx.append(items[b])
#             data.append(1)

#     return csr_matrix((data, (row_idx, col_idx)), shape=(len(users), len(items)))

# def find_best_threshold(model, X_val_pairs, users, items):
#     scores = []
#     labels = []

#     for (u, b, label) in X_val_pairs:
#         if u not in users or b not in items:
#             continue

#         u_idx = users[u]
#         b_idx = items[b]
#         score = np.dot(model.user_factors[u_idx], model.item_factors[b_idx])

#         scores.append(score)
#         labels.append(label)

#     if len(scores) == 0:
#         print("NO VALIDATION DATA AFTER FILTERING → threshold=0")
#         return 0

#     scores = np.array(scores)
#     labels = np.array(labels)

#     thresholds = np.linspace(scores.min(), scores.max(), 50)

#     best_t = 0
#     best_acc = -1

#     for t in thresholds:
#         preds = (scores > t).astype(int)
#         acc = (preds == labels).mean()
#         if acc > best_acc:
#             best_acc = acc
#             best_t = t

#     print(f"Selected threshold = {best_t:.4f}, acc = {best_acc:.4f}")
#     return best_t

# def filter_validation_by_model(model, validation_pairs, users, items):
#     # model factor sizes
#     max_user_idx = model.user_factors.shape[0]
#     max_item_idx = model.item_factors.shape[0]

#     filtered = []
#     skipped = 0

#     for (u, b, label) in validation_pairs:
#         if u not in users or b not in items:
#             continue

#         u_idx = users[u]
#         b_idx = items[b]

#         if u_idx < max_user_idx and b_idx < max_item_idx:
#             filtered.append((u, b, label))
#         else:
#             skipped += 1

#     print(f"Filtered validation pairs: {len(filtered)} (skipped {skipped})")
#     return filtered

# def writePredictionsRead(model, users, items, threshold):
#     predictions = open("predictions_Read.csv", 'w')
#     for l in open("pairs_Read.csv"):
#         if l.startswith("userID"):
#             predictions.write(l)
#             continue
#         u,b = l.strip().split(',')
#         if u in users and b in items:
#             score = model.predict(users[u], users[b]) # np.dot(model.user_factors[users[u]], model.item_factors[items[b]])
#         else:
#             score = 0
#         pred = 1 if score > threshold else 0
#         _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

#     predictions.close()
# ---------- IGNORE THIS, FAILED ATTEMPT ----------



# IDEA: HW3 solution, jaccard similarity + popularity threshold

def generateValidation(allRatings, ratingsValid):
    # Using ratingsValid, generate two sets:
    # readValid: set of (u,b) pairs in the validation set
    # notRead: set of (u,b') pairs, containing one negative (not read) for each row (u) in readValid  
    # Both should have the same size as ratingsValid
    readValid = {(u, b) for (u, b, rate) in ratingsValid}
    itemsPerUser = defaultdict(set)
    allItems = set()
    for u, b, rate in allRatings:
        itemsPerUser[u].add(b)
        allItems.add(b)

    rng = random.Random(42)
    notRead = set()
    for u, _ in readValid:
        candidates = list(allItems - itemsPerUser[u])
        b_prime = rng.choice(candidates)
        notRead.add((u, b_prime))

    return readValid, notRead



def baseLineStrategy(mostPopular, totalRead):
    return1 = set()
    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break

    return return1



def improvedStrategy(mostPopular, totalRead):
    return1 = set()
    # Same as above function, just find an item set that'll have higher accuracy
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/1.5: break
    return return1



def evaluateStrategy(return1, readValid, notRead):
    # Compute the accuracy of a strategy which just returns "true" for a set of items (those in return1)
    # readValid: instances with positive label
    # notRead: instances with negative label
    tp = sum(1 for u, b in readValid if b in return1)
    tn = sum(1 for u, b in notRead if b not in return1)
    if (len(readValid) + len(notRead)) > 0:
        acc = (tp + tn) / (len(readValid) + len(notRead))
    else:
        return 0
    return acc



def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):
    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular
    max_similar = 0
    u_b = set(ratingsPerItem.get(b, []))
    if u in ratingsPerUser:
        for b_prime in ratingsPerUser[u]:
            if isinstance(b_prime, tuple):
                b_prime_val = b_prime[0]
            else:
                b_prime_val = b_prime

            if b_prime_val in ratingsPerItem:
                similar = Jaccard(u_b, set(ratingsPerItem[b_prime_val]))
                if similar > max_similar:
                    max_similar = similar

    if max_similar > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0



def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        pred = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')
    predictions.close()



##################################################
# Category prediction                            #
##################################################

# IDEA: tf-idf
# tfidf(w, d) = tf(w, d) * log(num_docs / df(w))
# use TfidfVectorizer and svd dim reduction
# cap num features to get only top relevant words
# to make optinal dimensionality reduction, we can set svd to None
# if there is no input given svd_components to reduce to
# and only change if there is. save data 

def writePredictionsCategory(pred_test):
    predictions = open("predictions_Category.csv", 'w')
    pos = 0

    for l in open("pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
        pos += 1

    predictions.close()











##################################################
# Global Code (Generate pred files)              #
##################################################

if __name__ == "__main__":

    # ######################################
    # ---------- 0. Initial Setup ----------
    # ######################################

    DATA_DIR = "datasets/assignment1/"

    # Load rating data
    print("Loading rating data...")
    train_interactions = []
    for u, b, r in readCSV(DATA_DIR + "train_Interactions.csv.gz"):
        train_interactions.append((u, b, r))

    ratingsTrain = train_interactions[:190000]
    ratingsValid = train_interactions[190000:]

    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u,b,r in ratingsTrain:
        ratingsPerUser[u].append((b,r))
        ratingsPerItem[b].append((u,r))

    # # Preprocess traindata ONCE
    # users, items, indexed = preprocess_interactions(train_interactions)
    # num_users = len(users)
    # num_items = len(items)





    # ##########################################
    # ---------- 1. Rating prediction ----------
    # ##########################################

    print("Training rating model...")
    alpha0 = getGlobalAverage([r for (_,_,r) in ratingsTrain])
    betaU0 = {u:0 for u in ratingsPerUser}
    betaI0 = {b:0 for b in ratingsPerItem}
    alpha, betaU, betaI = goodModel(
        ratingsTrain, ratingsPerUser, ratingsPerItem, alpha0, betaU0, betaI0
    )

    # Write rating predictions
    print("Writing predictions_Rating.csv...")
    os.chdir(DATA_DIR)
    writePredictionsRating(alpha, betaU, betaI)
    os.chdir("../..")   # go back to top-level folder

    # ----- CUTOFF HERE FOR HYPERTUNING -----

    # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # best_mse = float('inf')
    # best_params = None

    # Iterate through hyperparameters
    # for nf in [10, 20]: # [10, 20]
    #     for lr in [0.005, 0.01]: # [0.005, 0.01]
    #         for reg in [0.01, 0.05, 0.1]: # [0.01, 0.05, 0.1]
    #             for ep in [10]: # [10]

    #                 mse_list = []

    #                 for train_idx, valid_idx in kf.split(indexed):

    #                     indexedTrain = [indexed[i] for i in train_idx]
    #                     indexedValid = [indexed[i] for i in valid_idx]

    #                     alpha, betaU, betaI, gammaU, gammaI = rating_grad_descent(
    #                         indexedTrain, indexedValid,
    #                         num_users, num_items,
    #                         num_latent_factors=nf,
    #                         learning_rate=lr,
    #                         regularizer=reg,
    #                         epochs=ep
    #                     )

    #                     # Same thing as printed validation mse, compute mse for this fold
    #                     fold_mse = 0
    #                     for (u,b,r) in indexedValid:
    #                         pred = alpha + betaU[u] + betaI[b] + np.dot(gammaU[u], gammaI[b])
    #                         fold_mse += (r - pred)**2
    #                     fold_mse /= len(indexedValid)

    #                     mse_list.append(fold_mse)

    #                 avg_mse = np.mean(mse_list)
    #                 print(f"[nf={nf}, lr={lr}, reg={reg}, ep={ep}] Avg MSE = {avg_mse:.4f}")

    #                 if avg_mse < best_mse:
    #                     best_mse = avg_mse
    #                     best_params = {
    #                         "nf": nf,
    #                         "lr": lr,
    #                         "reg": reg,
    #                         "ep": ep
    #                     }

    # print("Best params:", best_params)
    # print("Best CV MSE:", best_mse)
    # ----- CUTOFF HERE FOR HYPERTUNING -----

    # best_params = {"nf":20, 'lr':0.01, 'reg':0.05, 'ep':10}

    # # Train final model on full dataset with best hyperparameters
    # print("Training final model on full dataset...")
    # full_data = indexed
    # no_valid = []
    # alpha, betaU, betaI, gammaU, gammaI = rating_grad_descent(
    #     full_data,
    #     no_valid,
    #     num_users,
    #     num_items,
    #     num_latent_factors=best_params["nf"],
    #     learning_rate=best_params["lr"],
    #     regularizer=best_params["reg"],
    #     epochs=best_params["ep"]
    # )

    # # Write rating predictions
    # print("Writing predictions_Rating.csv...")
    # os.chdir(DATA_DIR)
    # writePredictionsRating(alpha, betaU, betaI, gammaU, gammaI)
    # os.chdir("../..")





    # ########################################
    # ---------- 2. Read prediction ----------
    # ########################################

    # ----- IGNORE THIS, FAILED ATTEMPT -----
    # print("Preparing read prediction model...")

    # # Make validation sets
    # # readValid = positive samples for validation
    # # notReadValid = negative/unknown samples for validation

    # # Build GLOBAL user/item maps
    # users_bpr, items_bpr = build_global_index_maps(train_interactions)

    # # ratingsTrain, ratingsValid = train_test_split(
    # #     train_interactions,
    # #     test_size=0.1,
    # #     random_state=42
    # # )

    
    # # Build interaction matrix
    # X_train = build_matrix(train_interactions, users_bpr, items_bpr)

    # # Build validation set (pos + neg)
    # # readValid = [(u, b) for (u, b, r) in ratingsValid if r > 0]

    # # all_items = set(items_bpr.keys())

    # # notReadValid = []
    # # for (u, b) in readValid:
    # #     user_read_items = {bb for (uu, bb) in readValid if uu == u}
    # #     neg_candidates = list(all_items - user_read_items)
    # #     if neg_candidates:
    # #         neg_item = random.choice(neg_candidates)
    # #         notReadValid.append((u, neg_item))

    # # validation_pairs = ([(u, b, 1) for (u, b) in readValid] + [(u, b, 0) for (u, b) in notReadValid])

    # # print(f"Validation pairs: {len(validation_pairs)}")

    # # Train BPR model
    # model = BayesianPersonalizedRanking(
    #     factors=50,
    #     learning_rate=0.05,
    #     regularization=0.01,
    #     iterations=15
    # )
    # model.fit(X_train.T.tocsr())

    # # Find best threshold
    # # threshold = find_best_threshold(model, validation_pairs, users_bpr, items_bpr)

    # # Write read predictions
    # print("Writing predictions_Read.csv...")
    # os.chdir(DATA_DIR)
    # writePredictionsRead(model, users_bpr, items_bpr, 0)
    # os.chdir("../..")
    # -----IGNORE THIS, FAILED ATTEMPT -----

    print("Preparing read prediction model...")
    # Compute item popularity
    bookCount = defaultdict(int)
    totalRead = 0
    for u,b,r in train_interactions:
        bookCount[b] += 1
        totalRead += 1
    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort(reverse=True)
    # ------------------------------
    # Write read predictions
    # ------------------------------
    print("Writing predictions_Read.csv...")
    os.chdir(DATA_DIR)
    writePredictionsRead(ratingsPerItem, ratingsPerUser)
    os.chdir("../..")





    # #############################################
    # ---------- 3. Category Predictions ----------
    # #############################################

    train_cat = list(readGz(DATA_DIR + "train_Category.json.gz"))
    X_train = [d["review_text"] for d in train_cat]
    y_train = [d["genreID"] for d in train_cat]

    test_cat = list(readGz(DATA_DIR + "test_Category.json.gz"))
    test_texts = [d["review_text"] for d in test_cat]

    # Split into train/validation for hyperparameter tuning
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # TF-IDF + Logistic Regression with ngrams and stopword removal
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
            ngram_range=(1,3),
            max_features=10000,
            stop_words='english',
            sublinear_tf=True,
            smooth_idf=True
        )),
        ('clf', LogisticRegression(max_iter=5000, multi_class='multinomial', random_state=42))
    ])

    # ----- CUTOFF HERE FOR HYPERTUNING -----
    # param_grid = {
    #     'clf__C': [1, 5, 10, 50]
    # }

    # grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    # grid.fit(X_train_split, y_train_split)

    # best_model = grid.best_estimator_
    # print("Best regularization C:", grid.best_params_['clf__C'])

    # y_valid_pred = best_model.predict(X_valid_split)
    # print(f"Validation Accuracy: {accuracy_score(y_valid_split, y_valid_pred):.4f}")
    # ----- CUTOFF HERE FOR HYPERTUNING -----

    pipeline.set_params(clf__C=1)
    pipeline.fit(X_train_split, y_train_split)
    best_model = pipeline
    print("Using best fixed regularization C = 1")

    # Predict on test data
    pred_test = best_model.predict(test_texts)

    # Write predictions
    print("Writing predictions_Category.csv...")
    os.chdir(DATA_DIR)
    writePredictionsCategory(pred_test)
    os.chdir("../..")

    print("All prediction files written successfully!")




