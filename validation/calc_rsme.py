import math

f_true = open('yelp_val.csv', 'r')
l_t = f_true.readline()
l_t = f_true.readline()
truth = {}
while l_t != '':
    splott = l_t.split(',')
    truth[f'{splott[0]},{splott[1]}'] = float(splott[2])

    l_t = f_true.readline()


f_pred = open('predictions.txt', 'r')
l_p = f_pred.readline()
l_p = f_pred.readline()

sse = 0
avg_dif = 0
avg_pos_dif = 0
avg_neg_dif = 0
pos_dif_count = 0
neg_dif_count = 0
count = 0
count_0_1 = 0
count_1_2 = 0
count_2_3 = 0
count_3_4 = 0
count_4_5 = 0
count_5_plus = 0

while l_p != '':
    # handle default placeholder
    splott = l_p.split(',')
    v_p = float(splott[2])
    v_t = truth[f'{splott[0]},{splott[1]}']

    if v_p == -999:
        l_p = f_pred.readline()
        continue

    dif = v_p - v_t

    if dif > 0:
        avg_pos_dif += dif
        pos_dif_count += 1
    elif dif < 0:
        avg_neg_dif += dif
        neg_dif_count += 1

    sse += dif ** 2
    avg_dif += dif

    dif = abs(dif)
    if dif >= 5:
        count_5_plus += 1
    elif dif >= 4:
        count_4_5 += 1
    elif dif >= 3:
        count_3_4 += 1
    elif dif >= 2:
        count_2_3 += 1
    elif dif >= 1:
        count_1_2 += 1
    else:
        count_0_1 += 1

    l_p = f_pred.readline()
    count += 1

avg_dif /= count
if pos_dif_count > 0:
    avg_pos_dif /= pos_dif_count
else:
    avg_pos_dif = -999
if neg_dif_count > 0:
    avg_neg_dif /= neg_dif_count
else:
    avg_neg_dif = -999
rmse = math.sqrt(sse/count)

print(f'rmse: {rmse}')
print(f'avg dif (v_p - v_t): {avg_dif}')

print(f'avg positive dif: {avg_pos_dif}')
print(f'avg negative dif: {avg_neg_dif}')

print(f'count_0_1: {count_0_1}')
print(f'count_1_2: {count_1_2}')
print(f'count_2_3: {count_2_3}')
print(f'count_3_4: {count_3_4}')
print(f'count_4_5: {count_4_5}')
print(f'count_5_plus: {count_5_plus}')
