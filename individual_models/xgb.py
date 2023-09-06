from pyspark import SparkContext
import sys
import time
import json

import xgboost as xgb

sc = SparkContext('local[*]', 'hw_3_task_2_3')
sc.setLogLevel('ERROR')

folder_path = sys.argv[1].rstrip('/')  # Param: folder_path: the path of dataset folder.
in_file = sys.argv[2]  # Param: test_file_name: the name of the testing file (e.g., yelp_val.csv), including file path
out_file = sys.argv[3]  # Param: output_file_name: the name of the prediction result file, including the file pat

train_file = '/'.join([folder_path, 'yelp_train.csv'])
user_file = '/'.join([folder_path, 'user.json'])
#user.json: {"user_id":"lzlZwIpuSWXEnNS91wxjHw","name":"Susan","review_count":1,"yelping_since":"2015-09-28","friends":"None","useful":0,"funny":0,"cool":0,"fans":0,"elite":"None","average_stars":2.0,"compliment_hot":0,"compliment_more":0,"compliment_profile":0,"compliment_cute":0,"compliment_list":0,"compliment_note":0,"compliment_plain":0,"compliment_cool":0,"compliment_funny":0,"compliment_writer":0,"compliment_photos":0}
business_file = '/'.join([folder_path, 'business.json'])
#business.json: {"business_id":"Apn5Q_b6Nz61Tq4XzPdf9A","name":"Minhas Micro Brewery","neighborhood":"","address":"1314 44 Avenue NE","city":"Calgary","state":"AB","postal_code":"T2E 6L6","latitude":51.0918130155,"longitude":-114.031674872,"stars":4.0,"review_count":24,"is_open":1,"attributes":{"BikeParking":"False","BusinessAcceptsCreditCards":"True","BusinessParking":"{'garage': False, 'street': True, 'validated': False, 'lot': False, 'valet': False}","GoodForKids":"True","HasTV":"True","NoiseLevel":"average","OutdoorSeating":"False","RestaurantsAttire":"casual","RestaurantsDelivery":"False","RestaurantsGoodForGroups":"True","RestaurantsPriceRange2":"2","RestaurantsReservations":"True","RestaurantsTakeOut":"True"},"categories":"Tours, Breweries, Pizza, Restaurants, Food, Hotels & Travel","hours":{"Monday":"8:30-17:0","Tuesday":"11:0-21:0","Wednesday":"11:0-21:0","Thursday":"11:0-21:0","Friday":"11:0-21:0","Saturday":"11:0-21:0"}}


# XGB Hyperparameters
xgb_params = {
    'alpha': 1,
    'max_depth': 8,
    'tree_method': 'hist'
}

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

# Start timing
start = time.time()


# TODO: DECIDE ON FEATURES
def usr_features_map(x):
    dat = json.loads(x)
    usr_id = dat['user_id']

    avg_stars = dat['average_stars']

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

    features = [
        avg_stars,
        rev_count
    ]

    return biz_id, features


# Get data for users/businesses --> {usr_id/biz_id: list[features]}
usr_features = sc.textFile(user_file).map(usr_features_map).collectAsMap()
biz_features = sc.textFile(business_file).map(biz_features_map).collectAsMap()

# TODO: Will change if features change
alt_uf = [2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
alt_bf = [2.5, 1]


def train_map(x):
    sploot = x.split(',')

    if sploot[0] in usr_features.keys():
        u_f = usr_features[sploot[0]]
    else:
        u_f = alt_uf

    if sploot[1] in biz_features.keys():
        b_f = biz_features[sploot[1]]
    else:
        b_f = alt_bf

    features = u_f + b_f
    rating = float(sploot[2])

    return features, rating


train_dat = sc.textFile(train_file).filter(lambda x: x != 'user_id,business_id,stars').map(train_map).collect()

x_train = [i[0] for i in train_dat]
y_train = [i[1] for i in train_dat]

# Train model
clf = xgb.XGBRegressor(**xgb_params)
clf.fit(x_train, y_train)


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


# WRITE TO OUTPUT FILE
with open(out_file, 'w') as f:
    f.write('user_id, business_id, prediction\n')

    f_tmp = open(in_file, 'r')
    line = f_tmp.readline()
    line = f_tmp.readline()
    while line != '':
        sploot = line[:-1].split(',')
        key = f'{sploot[0]},{sploot[1]}'

        feats = get_usr_features(sploot[0]) + get_biz_features(sploot[1])
        xgb_pred = clf.predict(feats)[0]

        # Max business review is 7968
        xgb_coeff = 1
        pred = (xgb_coeff * xgb_pred)
        f.write(f'{key},{pred}\n')

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

/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G xgb.py ../resource/asnlib/publicdata/ ../resource/asnlib/publicdata/yelp_val.csv ./xgb_test.txt
'''
