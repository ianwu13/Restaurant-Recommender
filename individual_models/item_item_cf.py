from pyspark import SparkContext
import sys
import time
import math

sc = SparkContext('local[*]', 'hw_3_task_2_1')
sc.setLogLevel('ERROR')

train_file = sys.argv[1]
in_file = sys.argv[2]
out_file = sys.argv[3]

# Tuning parameters
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

# Start timingt
start = time.time()


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


# rdd has (biz_id, (user_id, rating))
rdd = sc.textFile(train_file).filter(lambda x: x != 'user_id,business_id,stars').cache()

user_ratings_list = rdd.map(usr_rat_map).groupByKey().collectAsMap()  # {usr_id: list[(biz_ids, rating)]}

# items has (business_id, list[(user_id, stars)])
rdd = rdd.map(basket_map).groupByKey().cache()
items = rdd.mapValues(get_normalized_ratings).collectAsMap()
un_normed_items = rdd.collectAsMap()  # {biz_id (aka item): list[(rater, norm_rating)]}


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
    elif x < 0:
        return 0
    else:
        return x


# rdd has (biz_id, list[user_id])
rdd = sc.textFile(in_file).filter(lambda x: x != 'user_id,business_id,stars').map(pred_map).groupByKey()
# results has ("user_id,biz_id", prediction)
results = rdd.flatMap(calc_predictions).mapValues(force_to_range).collectAsMap()

# WRITE TO OUTPUT FILE
with open(out_file, 'w') as f:
    f.write('user_id, business_id, prediction\n')

    f_tmp = open(in_file, 'r')
    line = f_tmp.readline()
    line = f_tmp.readline()
    while line != '':
        sploot = line[:-1].split(',')
        key = f'{sploot[0]},{sploot[1]}'
        f.write(f'{key},{results[key]}\n')
        line = f_tmp.readline()
    f_tmp.close()

end = time.time()
exe_time = end - start
print(f'Duration: {exe_time}')

'''
# RUNNING CODE ON VOCAREUM
# FIRST CALL THESE IN TERMINAL:
    - export PYSPARK_PYTHON=python3.6
    - export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64
    - /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit

# TO RUN SCRIPT IN TERMINAL:
    /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G item_item_cf.py ../resource/asnlib/publicdata/yelp_train.csv ../resource/asnlib/publicdata/yelp_val.csv ./item_cf_test.txt
'''
