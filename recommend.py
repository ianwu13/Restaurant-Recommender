'''
Method Description:
In my method I use a feature combination type hybrid recommender.
The actual recommender model is an XGBoost regressor. Meanwhile, SVD, KNN, Co-clustering, and Item-Item Collaborative
Filtering are used as the contributing recommenders. The predictions from the contributing recommenders are combined
with user and business features to generate the final set of features that the XGBoost model takes in to generate a
final prediction. The hyperparameters for each model were initially found through grid-search cross validation, and
then fine-tuned by hand to get their final values.
I initially tried to implement a switching model with XGBoost, Item-Item CF, and SVD but found that it was too difficult
to select the best model for a given instance. A weighted sum of the models learned through linear regression also
proved difficult to achieve improvement with. I then tried adding the predictions for SVD and Item-Item CF to the input
features for XGBoost and found this to yield some improvement. By introducing the KNN and Co-Clustering models from
SciKit-Surprise, I was able to further improve the accuracy. Finally, I re-tuned the hyperparameters of the hybrid model
as a whole to get my final setup and RMSE score.
As previously mentioned, user and business features were also included as input to the XGBoost model. User features were
selected based on testing/selections from Assignment 3, and business features were selected based on the results of the
following 2 studies:
    - https://cseweb.ucsd.edu/classes/wi15/cse255-a/reports/fa15/010.pdf
    - https://scholars.unh.edu/cgi/viewcontent.cgi?article=1379&context=honors

Error Distribution:
>=0 and <1: 102353
>=1 and <2: 32706
>=2 and <3: 6153
>=3 and <4: 832
>=4: 0

RMSE:
0.9783791745615509

Execution Time:
372.51162099838257 s
'''

from pyspark import SparkContext
import sys
import time
import json
import math

import xgboost as xgb

import pandas as pd
import surprise


sc = SparkContext('local[*]', '553_Competition_Project')
sc.setLogLevel('ERROR')

folder_path = sys.argv[1].rstrip('/')  # Param: folder_path: the path of dataset folder.
in_file = sys.argv[2]  # Param: test_file_name: the name of the testing file (e.g., yelp_val.csv), including file path
out_file = sys.argv[3]  # Param: output_file_name: the name of the prediction result file, including the file pat

# Generate file paths for data
train_file = '/'.join([folder_path, 'yelp_train.csv'])
user_file = '/'.join([folder_path, 'user.json'])
#user.json: {"user_id":"lzlZwIpuSWXEnNS91wxjHw","name":"Susan","review_count":1,"yelping_since":"2015-09-28","friends":"None","useful":0,"funny":0,"cool":0,"fans":0,"elite":"None","average_stars":2.0,"compliment_hot":0,"compliment_more":0,"compliment_profile":0,"compliment_cute":0,"compliment_list":0,"compliment_note":0,"compliment_plain":0,"compliment_cool":0,"compliment_funny":0,"compliment_writer":0,"compliment_photos":0}
business_file = '/'.join([folder_path, 'business.json'])
#business.json: {"business_id":"Apn5Q_b6Nz61Tq4XzPdf9A","name":"Minhas Micro Brewery","neighborhood":"","address":"1314 44 Avenue NE","city":"Calgary","state":"AB","postal_code":"T2E 6L6","latitude":51.0918130155,"longitude":-114.031674872,"stars":4.0,"review_count":24,"is_open":1,"attributes":{"BikeParking":"False","BusinessAcceptsCreditCards":"True","BusinessParking":"{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}","GoodForKids":"True","HasTV":"True","NoiseLevel":"average","OutdoorSeating":"False","RestaurantsAttire":"casual","RestaurantsDelivery":"False","RestaurantsGoodForGroups":"True","RestaurantsPriceRange2":"2","RestaurantsReservations":"True","RestaurantsTakeOut":"True"},"categories":"Tours, Breweries, Pizza, Restaurants, Food, Hotels & Travel","hours":{"Monday":"8:30-17:0","Tuesday":"11:0-21:0","Wednesday":"11:0-21:0","Thursday":"11:0-21:0","Friday":"11:0-21:0","Saturday":"11:0-21:0"}}


# XGB Hyperparameters
xgb_params = {
    'n_estimators': 150,
    'alpha': 1,
    'max_depth': 6,
    'tree_method': 'hist',
    'gamma': 0.01,
    'seed': 0
}
# gamma = 0
# n_estimators
# reg_alpha
# reg_lambda

# TODO: Will change if features change
alt_uf = [2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
alt_bf = [2.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Collaborative Filtering Parameters
case_amplification_p_minus_1 = 2.5 - 1  # w --> w*(abs(w)^(p-1))
choose_n_neighbors = 20

corat_threshold = 35
'''
00 --> rmse: 1.1443680627990607
10 --> rmse: 1.13997378924834
20 --> rmse: 1.1016024762086296
25 --> rmse: 1.091699454932054
30 --> rmse: 1.0873005117067456
35 --> rmse: 1.083803226459816
'''

# Surprise Hyperparameters
svd_params = {
    'n_epochs': 25,
    'lr_all': 0.01,
    'reg_all': 0.2,
    'random_state': 0
}
svd_discrepancy = 0.00344
# n_epochs=25, lr_all=0.01, reg_all=0.2 --> 1.0017625763667906

knn_params = {
    'k': 15,
    'random_state': 0
}
knn_discrepancy = 0.01817
# k=15 --> 1.0865954761635528

cc_params = {
    'n_cltr_u': 5,
    'n_cltr_i': 5,
    'random_state': 0
}
cc_discrepancy = -0.00346
# n_cltr_u=5, n_cltr_i=5 --> 1.066489389156545

# Item-Item CF Functions

# Construct baskets - (business_id, (user_id, stars))
def basket_map(x):
    sploot = x.split(',')
    return sploot[1], (sploot[0], float(sploot[2]))


def usr_rat_map(x):
    sploot = x.split(',')
    return sploot[0], (sploot[1], float(sploot[2]))


def get_normalized_ratings(x):
    r_bar = 0
    count = 0
    for i in x:
        r_bar += i[1]
        count += 1
    r_bar /= count

    res = [(i[0], i[1] - r_bar) for i in x]

    return res


def pred_map(x):
    sploot = x.split(',')
    return sploot[1], sploot[0]


# Takes 2x list[(rater, norm_rating)] --> pierson corr
def get_pierson_corr(a, b):
    dict_b = {i[0]: i[1] for i in b}
    tot = 0
    ss_a = 0
    ss_b = 0
    corat_count = 0
    for i in a:
        if i[0] in dict_b:
            tot += i[1] * dict_b[i[0]]
            ss_a += i[1] ** 2
            ss_b += dict_b[i[0]] ** 2
            corat_count += 1

    if corat_count < corat_threshold:
        return 0

    denom = (math.sqrt(ss_a) * math.sqrt(ss_b))
    if denom > 0:
        weight = tot / denom
    else:
        # No user has rated both items - Weight is 0
        return 0

    # case/weight amplification - VERY HELPFUL
    weight = weight * (abs(weight) ** case_amplification_p_minus_1)

    return weight


# (biz_id, list[user_id]) --> list[("user_id,biz_id", prediction)]
def calc_predictions(x):
    def get_usr_avg(y):
        if y in user_ratings_list.keys():
            usr_rats = [i[1] for i in user_ratings_list[y]]
            return sum(usr_rats) / len(usr_rats)
        else:
            return 2.5

    if x[0] in items.keys():
        biz_ratings = items[x[0]]  # list[(rater/user_id, norm_rating)]
    else:
        # business has not been rated/does not occur in training set
        # Returning average rating for user
        return [(f'{usr},{x[0]}', get_usr_avg(usr)) for usr in x[1]]  # list[("user_id,biz_id", prediction)]

    needed_sims = set()
    users_to_rate = []
    users_to_default = []
    for usr in x[1]:
        if usr in user_ratings_list.keys():
            needed_sims = needed_sims.union(set(user_ratings_list[usr]))
            users_to_rate.append(usr)
        else:
            users_to_default.append(usr)

    weights = {}
    for biz in needed_sims:
        item_info = items[biz[0]]
        weights[biz[0]] = get_pierson_corr(biz_ratings, item_info)

    # Take in x: sorted iterable of users wanting prediction for key in x[0]
    # return set of predictions for user/biz pairs

    res = []
    for usr in users_to_rate:
        pred = 0
        weights_sum = 0

        sorted_usr_rat = sorted(list(user_ratings_list[usr]), key=lambda x: -weights[x[0]])

        for r in sorted_usr_rat[:choose_n_neighbors]:
            # r --> (biz_ids, rating)
            w = weights[r[0]]
            if w > 0:
                pred += r[1] * w
                weights_sum += abs(w)

        if weights_sum > 0:
            pred /= weights_sum
        else:
            # user has rated other items, but none of them are corated with the target biz for prediction
            # Returning average rating for user
            pred = get_usr_avg(usr)

        res.append((f'{usr},{x[0]}', pred))

    # User has not rated anything/does not occur in training set
    # Returning average rating for biz
    biz_rats = [i[1] for i in un_normed_items[x[0]]]
    pred = sum(biz_rats) / len(biz_rats)
    for usr in users_to_default:
        res.append((f'{usr},{x[0]}', pred))

    return res  # list[("user_id,biz_id", prediction)]


def force_to_range(x):
    if x > 5:
        return 5
    elif x < 1:
        return 1
    else:
        return x


# SVD Functions

def usr_rat_map_svd(x):
    sploot = x.split(',')
    return sploot[0], sploot[1], float(sploot[2])


# XGB Functions

# TODO: DECIDE ON FEATURES
def usr_features_map(x):
    dat = json.loads(x)
    usr_id = dat['user_id']

    avg_stars = dat['average_stars']

    rev_count = dat['review_count']

    useful = dat['useful']
    funny = dat['funny']
    cool = dat['cool']
    ufc_sum = useful + funny + cool
    if ufc_sum > 0:
        useful /= ufc_sum
        funny /= ufc_sum
        cool /= ufc_sum

    fans = dat['fans']

    comp_list = [
        dat['compliment_hot'],
        dat['compliment_more'],
        dat['compliment_profile'],
        dat['compliment_cute'],
        dat['compliment_list'],
        dat['compliment_note'],
        dat['compliment_plain'],
        dat['compliment_cool'],
        dat['compliment_funny'],
        dat['compliment_writer'],
        dat['compliment_photos']
    ]
    comp_total = sum(comp_list)
    if comp_total > 0:
        for i in range(len(comp_list)):
            comp_list[i] /= comp_total

    features = [
        avg_stars,
        rev_count,
        useful,
        funny,
        cool,
        fans,
        *comp_list
    ]

    return usr_id, features


# TODO: DECIDE ON FEATURES
def biz_features_map(x):
    dat = json.loads(x)
    biz_id = dat['business_id']

    avg_stars = dat['stars']

    rev_count = dat['review_count']

    attr = dat['attributes']
    if attr is None:
        credit = 0.5
        kids = 0.5
        reserve = 0.5
        groups = 0.5
        alc = 0.5
        takeout = 0.5
        out_sit = 0.5
        tv = 0.5
    else:
        if 'BusinessAcceptsCreditCards' in attr.keys():
            credit = 1 if attr['BusinessAcceptsCreditCards'] else 0
        else:
            credit = 0.5
        if 'GoodForKids' in attr.keys():
            kids = 1 if attr['GoodForKids'] else 0
        else:
            kids = 0.5
        if 'RestaurantsReservations' in attr.keys():
            reserve = 1 if attr['RestaurantsReservations'] else 0
        else:
            reserve = 0.5
        if 'RestaurantsGoodForGroups' in attr.keys():
            groups = 1 if attr['RestaurantsGoodForGroups'] else 0
        else:
            groups = 0.5
        if 'Alcohol' in attr.keys():
            alc = 0 if attr['Alcohol'] == 'none' else 1
        else:
            alc = 0.5
        if 'RestaurantsTakeOut' in attr.keys():
            takeout = 1 if attr['RestaurantsTakeOut'] else 0
        else:
            takeout = 0.5
        if 'OutdoorSeating' in attr.keys():
            out_sit = 1 if attr['OutdoorSeating'] else 0
        else:
            out_sit = 0.5
        if 'HasTV' in attr.keys():
            tv = 1 if attr['HasTV'] else 0
        else:
            tv = 0.5

    features = [
        avg_stars,
        rev_count,
        credit,
        kids,
        reserve,
        groups,
        alc,
        takeout,
        out_sit,
        tv
    ]

    return biz_id, features


def get_usr_features(usr):
    if usr in usr_features.keys():
        return usr_features[usr]
    else:
        return alt_uf


def get_biz_features(biz):
    if biz in biz_features.keys():
        return biz_features[biz]
    else:
        return alt_bf


# Start timing
start = time.time()

# Item-Item CF
print('TRAINING ITEM_ITEM CF')
# rdd has (biz_id, (user_id, rating))
rdd, rdd_xgb = sc.textFile(train_file).filter(lambda x: x != 'user_id,business_id,stars').randomSplit([0.5, 0.5], seed=0)

user_ratings_list = rdd.map(usr_rat_map).groupByKey().collectAsMap()  # {usr_id: list[(biz_ids, rating)]}

# items has (business_id, list[(user_id, stars)])
rdd_baskets = rdd.map(basket_map).groupByKey().cache()
items = rdd_baskets.mapValues(get_normalized_ratings).collectAsMap()
un_normed_items = rdd_baskets.collectAsMap()  # {biz_id (aka item): list[(rater, norm_rating)]}

# rdd has (biz_id, list[user_id])
rdd_infile = sc.textFile(in_file).filter(lambda x: x != 'user_id,business_id,stars').map(pred_map).groupByKey()
# results has ("user_id,biz_id", prediction)
item_item_results = rdd_infile.flatMap(calc_predictions).mapValues(force_to_range).collectAsMap()
item_item_train_results = rdd_xgb.map(pred_map).groupByKey().flatMap(calc_predictions).mapValues(force_to_range).collectAsMap()


# SVD
print('TRAINING SVD')
# rdd has (biz_id, (user_id, rating))
# ALREADY DONE: rdd = sc.textFile(train_file).filter(lambda x: x != 'user_id,business_id,stars').cache()

# TODO: Maybe try removing users/biz with low num ratings?
user_ratings = rdd.map(usr_rat_map_svd).collect()

dataset = surprise.Dataset.load_from_df(pd.DataFrame(user_ratings), surprise.Reader()).build_full_trainset()

svd = surprise.SVD(**svd_params)
svd.fit(dataset)

knn = surprise.prediction_algorithms.knns.KNNBasic(**knn_params)
knn.fit(dataset)

# s1 = surprise.prediction_algorithms.slope_one.SlopeOne()
# s1.fit(dataset)

cc = surprise.prediction_algorithms.co_clustering.CoClustering(**cc_params)
cc.fit(dataset)


# XGB
print('TRAINING XGB')

# Get data for users/businesses --> {usr_id/biz_id: list[features]}
usr_features = sc.textFile(user_file).map(usr_features_map).collectAsMap()
biz_features = sc.textFile(business_file).map(biz_features_map).collectAsMap()


def train_map(x):
    sploot = x.split(',')
    key = f'{sploot[0]},{sploot[1]}'

    if sploot[0] in usr_features.keys():
        u_f = usr_features[sploot[0]]
    else:
        u_f = alt_uf

    if sploot[1] in biz_features.keys():
        b_f = biz_features[sploot[1]]
    else:
        b_f = alt_bf

    item_cf_pred = item_item_train_results[key]
    svd_pred = svd.predict(sploot[0], sploot[1]).est - svd_discrepancy
    # knn_pred = knn.predict(sploot[0], sploot[1]).est - knn_discrepancy
    cc_pred = cc.predict(sploot[0], sploot[1]).est - cc_discrepancy
    if sploot[0] in user_ratings_list.keys():
        n_usr_reviews = len(user_ratings_list[sploot[0]])
    else:
        n_usr_reviews = 0
    if sploot[1] in un_normed_items.keys():
        n_biz_reviews = len(un_normed_items[sploot[1]])
    else:
        n_biz_reviews = 0

    features = u_f + b_f + [item_cf_pred, svd_pred, cc_pred, n_usr_reviews, n_biz_reviews] # , knn_pred]
    rating = float(sploot[2])

    return features, rating, sploot[0], sploot[1]


train_dat = rdd_xgb.map(train_map).collect()

of_train = [i[0] for i in train_dat]
# Must calculate knn predcitions outside of map because model too large to send to workers
knn_p_train = [knn.predict(i[2], i[3]).est - knn_discrepancy for i in train_dat]
x_train = [i[:-2] + [j] + i[-2:] for i, j in zip(of_train, knn_p_train)]
y_train = [i[1] for i in train_dat]

# Train model
clf = xgb.XGBRegressor(**xgb_params)
clf.fit(x_train, y_train)


# WRITE TO OUTPUT FILE
print('GENERATING FINAL PREDS')
with open(out_file, 'w') as f:
    f.write('user_id, business_id, prediction\n')
    # f.write('true,xgb_pred\n')

    f_tmp = open(in_file, 'r')
    line = f_tmp.readline()
    line = f_tmp.readline()
    while line != '':
        sploot = line[:-1].split(',')
        key = f'{sploot[0]},{sploot[1]}'

        item_cf_pred = item_item_results[key]
        svd_pred = svd.predict(sploot[0], sploot[1]).est - svd_discrepancy
        knn_pred = knn.predict(sploot[0], sploot[1]).est - knn_discrepancy
        cc_pred = cc.predict(sploot[0], sploot[1]).est - cc_discrepancy

        if sploot[0] in user_ratings_list.keys():
            n_usr_reviews = len(user_ratings_list[sploot[0]])
        else:
            n_usr_reviews = 0
        if sploot[1] in un_normed_items.keys():
            n_biz_reviews = len(un_normed_items[sploot[1]])
        else:
            n_biz_reviews = 0

        feats = get_usr_features(sploot[0]) + get_biz_features(sploot[1]) + [item_cf_pred, svd_pred, cc_pred, knn_pred, n_usr_reviews, n_biz_reviews]
        xgb_pred = float(clf.predict(feats)[0])
        xgb_pred = force_to_range(xgb_pred)

        f.write(f'{sploot[0]},{sploot[1]},{xgb_pred}\n')

        line = f_tmp.readline()
    f_tmp.close()

end = time.time()
exe_time = end - start
print(f'Duration: {exe_time}')

'''
# RUNNING CODE ON VOCAREUM
# FIRST CALL THESE IN TERMINAL:

export PYSPARK_PYTHON=python3.6
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit

# TO RUN SCRIPT IN TERMINAL:

/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit competition.py ../resource/asnlib/publicdata/ ../resource/asnlib/publicdata/yelp_val.csv ./test.txt
'''
