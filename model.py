import tensorflow as tf
import numpy as np
import csv
import time

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.optimizers import Adam

from shared import generate_batches, train_validation_split

training_percent = 0.9      # proportion of data to use for training

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'model_data/IMG/', 'The directory of the image data.')
flags.DEFINE_string('data_file', 'model_data/driving_log', 'The path to the csv of training data for lane centre.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 5, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.0001, 'The learning rate for training.')

# Adding multiple sets of training data here
# Some are centre-images
# Some are left/right side recovery images
# These flags define the number of files to process - sequential naming
#   driving_log_centre_1, driving_log_centre_2, driving_log_left_1 etc
flags.DEFINE_integer('centre_count', 4, 'The number of centre-lane driving files.')
flags.DEFINE_integer('left_recover_count', 2, 'The number of left-of-centre recovery files.')
flags.DEFINE_integer('right_recover_count', 2, 'The number of right-of-centre recovery files.')

def main(_):
    # Main model-building process

    print('NVIDIA model')
    print()
    print('Starting data load...')
    print(time.strftime('%H:%M:%S'))
    print()
    start_time = time.time()

    # First load in the data

    # Will have centre_count main driving files - each of which are frames taken when driving centrally
    # Naming driving_log_centre_X.csv where X is sequential numbering
    # Will have left_recover_count files - each of which represents sets of frames taken when recovering from the left
    #   When processing these files will just take +ve steering angle imags, ignoring all others
    # Will have right_recover_count files - each of which represents set of frames taken when recovering from the right
    #   When processing these files will just take -ve steering angle images, ignoring all others
    # Note that both the left_ and right_ recovery files will have some normal driving frames with +ve/-ve steering angles
    #   and these will just add to the set of data

    # Initialise variables
    X_train = []                # our training data image paths
    y_train = []                # our training data labels
    n_min_y = 0.0               # will use these two variables to track the min/max steering angles found in the data
    n_max_y = 0.0

    # Read in the centre images data
    for file_num in range(1,FLAGS.centre_count+1):
        sFilename = FLAGS.data_file + '_centre_' + str(file_num) + '.csv'
        with open(sFilename, 'r') as f:
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

        # Get the centre camera view and steering angle - this is our training input and label
        for i in range(len(data)):
            if data[i][0] != 'center':  # ignore the header row
                # sort out the image path, if using the Udacity provided data (different path showing in CSV file)
                s = str(data[i][0])             # get the filename from the csv data
                s = s[s.rfind('/')+1:]          # trim off any paths
                s = FLAGS.imgs_dir + s          # put back the path to the images location
                X_train.append(s)
                y_label = float(data[i][3])
                y_train.append(y_label)
                if y_label < n_min_y:
                    n_min_y = y_label
                if y_label > n_max_y:
                    n_max_y = y_label

    print('Imported ', len(X_train), ' centre frames')
    print('Total frames so far :', len(X_train))
    print('Range of label values : ', n_min_y, ' to ', n_max_y)
    print()
    n_centre_frames = len(X_train)

    # Read in the left-of-centre recovery images data
    for file_num in range(1,FLAGS.left_recover_count+1):
        sFilename = FLAGS.data_file + '_left_' + str(file_num) + '.csv'
        with open(sFilename, 'r') as f:
            reader = csv.reader(f)
            # data is a list of tuples (img path, steering angle)
            data = np.array([row for row in reader])

        for i in range(len(data)):
            if data[i][0] != 'center':  # ignore the header row, if there is one
                if float(data[i][3]) > 0.0:      # only interested in +ve steering angle frames
                    # sort out the image path, if using the Udacity provided data (different path showing in CSV file)
                    s = str(data[i][0])             # get the filename from the csv data
                    s = s[s.rfind('/')+1:]          # trim off any paths
                    s = FLAGS.imgs_dir + s          # put back the path to the images location
                    X_train.append(s)
                    y_label = float(data[i][3])
                    y_train.append(y_label)
                    if y_label < n_min_y:
                        n_min_y = y_label
                    if y_label > n_max_y:
                        n_max_y = y_label

    n_left_frames = len(X_train) - n_centre_frames
    print('Imported ', n_left_frames, ' left frames')
    print('Total frames so far :', len(X_train))
    print('Range of label values : ', n_min_y, ' to ', n_max_y)
    print()

    # Read in the right-of-centre recovery images data
    for file_num in range(1, FLAGS.right_recover_count + 1):
        sFilename = FLAGS.data_file + '_right_' + str(file_num) + '.csv'
        with open(sFilename, 'r') as f:
            reader = csv.reader(f)
            # data is a list of tuples (img path, steering angle)
            data = np.array([row for row in reader])

        for i in range(len(data)):
            if data[i][0] != 'center':  # ignore the header row, if there is one
                if float(data[i][3]) < 0.0:  # only interested in -ve steering angle frames
                    # sort out the image path, if using the Udacity provided data (different path showing in CSV file)
                    s = str(data[i][0])  # get the filename from the csv data
                    s = s[s.rfind('/') + 1:]  # trim off any paths
                    s = FLAGS.imgs_dir + s  # put back the path to the images location
                    X_train.append(s)
                    y_label = float(data[i][3])
                    y_train.append(y_label)
                    if y_label < n_min_y:
                        n_min_y = y_label
                    if y_label > n_max_y:
                        n_max_y = y_label

    n_right_frames = len(X_train) - n_left_frames - n_centre_frames
    print('Imported ', n_right_frames, ' right frames')
    print('Total frames so far :', len(X_train))
    print('Range of label values : ', n_min_y, ' to ', n_max_y)
    print()

    print('Total frames : ', len(X_train))
    print()

    # Split into training and validation data
    X_train, y_train, X_valid, y_valid = train_validation_split(X_train, y_train, training_percent)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_valid = np.array(X_valid)
    y_valid = np.array(y_valid)

    print('Split training/validation sets (', training_percent, '/', (1.0 - training_percent), ')')

    print('Training set size   : ', len(X_train))
    print('Validation set size : ', len(X_valid))
    print()

    # Define Model layout - this is based on the NVidia model, with some dropout added

    print('Creating model...')
    model = Sequential([
        Conv2D(24, 5, 5, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        Conv2D(36, 5, 5, border_mode='same', activation='relu'),
        Conv2D(48, 5, 5, border_mode='same', activation='relu'),
        Dropout(0.5),
        Conv2D(64, 3, 3, border_mode='same', activation='relu'),
        Conv2D(64, 3, 3, border_mode='same', activation='relu'),
        Dropout(0.5),
        Flatten(),
        Dense(1164, activation='relu'),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    model.compile(optimizer=Adam(lr=FLAGS.lrate), loss='mse')
    model.summary()
    print('Model created')
    print()

    # Train the model

    # When using a generator, can say how may samples to use per epoch
    # The generator will be creating random batches, so this value can be other than the total samples value
    # But in this case we want to use roughly that size

    n_samples_per_epoch = len(X_train)
    n_validation_size   = len(X_valid)
    history = model.fit_generator(generate_batches(X_train, y_train, FLAGS.batch_size),
                                  n_samples_per_epoch,
                                  FLAGS.num_epochs,
                                  validation_data=generate_batches(X_valid, y_valid, FLAGS.batch_size),
                                  nb_val_samples=n_validation_size)

    # Save model

    json = model.to_json()
    model.save_weights('save/model.h5')
    with open('save/model.json', 'w') as f:
        f.write(json)

    end_time = time.time()
    print()
    print('Training time : ', (end_time - start_time) / 60, ' minutes')

if __name__ == '__main__':
    tf.app.run()