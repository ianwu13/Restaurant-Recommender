from pyspark import SparkContext
import sys
import time
import json

import pandas as pd
import surprise

# https://colab.research.google.com/github/singhsidhukuldeep/Recommendation-System/blob/master/Building_Recommender_System_with_Surprise.ipynb#scrollTo=GuJyfbZc6miT

sc = SparkContext('local[*]', 'hw_3_task_2_3')
sc.setLogLevel('ERROR')

folder_path = sys.argv[1].rstrip('/')  # Param: folder_path: the path of dataset folder.
in_file = sys.argv[2]  # Param: test_file_name: the name of the testing file (e.g., yelp_val.csv), including file path
out_file = sys.argv[3]  # Param: output_file_name: the name of the prediction result file, including the file pat

train_file = '/'.join([folder_path, 'yelp_train.csv'])

# Start timing
start = time.time()

# rdd has (biz_id, (user_id, rating))
rdd = sc.textFile(train_file).filter(lambda x: x != 'user_id,business_id,stars').cache()

# Assign indices to users and businesses
'''
# Get data for users/businesses
{usr_id/biz_id: list[features]}
'''
train_usr_indices = rdd.map(lambda x: x.split(',')[0]).distinct().zipWithIndex().collectAsMap()
n_usr = len(train_usr_indices)
train_biz_indices = rdd.map(lambda x: x.split(',')[1]).distinct().zipWithIndex().collectAsMap()
n_biz = len(train_biz_indices)


def usr_rat_map(x):
    sploot = x.split(',')
    return sploot[0], sploot[1], float(sploot[2])


user_ratings = rdd.map(usr_rat_map).collect()

dataset = surprise.Dataset.load_from_df(pd.DataFrame(user_ratings), surprise.Reader()).build_full_trainset()

svd = surprise.SVD(n_epochs=25, lr_all=0.01, reg_all=0.2)
# n_epochs=25, lr_all=0.01, reg_all=0.2 --> 1.0017625763667906
svd.fit(dataset)
# SVDpp --> Duration: 792.9774758815765

# Save model
# surprise.dump.dump('svd_model.pickle', algo=svd)

# TO LOAD MODEL:
# _, svd = surprise.dump.load('svd_model.pickle')

# WRITE TO OUTPUT FILE
with open(out_file, 'w') as f:
    f.write('user_id, business_id, prediction\n')

    f_tmp = open(in_file, 'r')
    line = f_tmp.readline()
    line = f_tmp.readline()
    while line != '':
        sploot = line[:-1].split(',')
        res = svd.predict(sploot[0], sploot[1])

        f.write(f'{sploot[0]},{sploot[1]},{res.est}\n')

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

/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G svd.py ../resource/asnlib/publicdata/ ../resource/asnlib/publicdata/yelp_val.csv ./svd_test.txt
'''
