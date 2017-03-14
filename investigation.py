import csv
import numpy as np
from utils import train_validation_split
import time

start_time = time.time()
print(time.strftime('%H:%M:%S'))


data_file = 'data/driving_log.csv'

with open(data_file, 'r', ) as f:
    reader = csv.reader(f)
    # data is a list of tuples (img path, steering angle)
    data = np.array([row for row in reader])

    # Provided data has form of:
    #   data[i][0] = centre image
    #   data[i][1] = left image
    #   data[i][2] = right image
    #   data[i][3] = steering angle
    #   data[i][4] = throttle
    #   data[i][5] = brake
    #   data[i][6] = speed

print('Records count : ', len(data))
print('First : ', data[0])
print('Second: ', data[1])
print()

# Extract the interesting data - centre image and labels

X_train = []
y_train = []
n_min_y = 0.0
n_max_y = 0.0
for i in range(len(data)):
    #print('Record : ', data[i])
    if data[i][0] != 'center':              # ignore the header row
        s = str(data[i][0]).replace('IMG/', '')
        s = 'data/IMG/' + s
        print(s)
        X_train.append(data[i][0])
        y_train.append(float(data[i][3]))
        if float(data[i][3]) < n_min_y:
            n_min_y = float(data[i][3])
            print('New minimum steering angle : ', n_min_y, ' (row ', i, ')')
        if float(data[i][3]) > n_max_y:
            n_max_y = float(data[i][3])
            print('New maximum steering angle ; ', n_max_y, ' (row ', i, ')')

print('X_train length : ', len(X_train))

X_train, y_train = np.array(X_train), np.array(y_train)
min_y = min(y_train)
max_y = max(y_train)
print ('Range of label values : ', str(min_y), ',', max_y)
print()
print('First image file : ', X_train[0])

X_train, y_train, X_valid, y_valid = train_validation_split(X_train, y_train, 0.9)

end_time = time.time()

print('Seconds to run: ', end_time - start_time)