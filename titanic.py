import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt


np.random.seed(1223)                                    #### requiring same set of train and validation data for all the run
pd.set_option('display.expand_frame_repr', False)       #### for showing the full column


df_train = pd.read_csv('train.csv')                     #### reading the csv file #### shape: (891, 12)
df_test = pd.read_csv('test.csv')                       #### reading the csv file #### shape: (418, 11)


df_features = df_train.iloc[:, 2:12]                    #### managing dataset
df_labels = df_train.iloc[:, 1]                         #### managing dataset
x_test = df_test.iloc[:, 1:11]                          #### managing dataset



############################################ feature engineering #######################################################

################# name feature representation ######################
def name_feature_represntation(df_fun):
    def split_title(strings):
        modified_title_list = []
        for string in strings:
            inner_str = string.split(',')[-1].split('.')[0]
            if inner_str == ' Mr':
                modified_title_list.append(1)
            elif inner_str == ' Mrs':
                modified_title_list.append(2)
            elif inner_str == ' Miss':
                modified_title_list.append(3)
            elif inner_str == ' Master':
                modified_title_list.append(4)
            elif inner_str == ' Don':
                modified_title_list.append(5)
            elif inner_str == ' Rev':
                modified_title_list.append(6)
            elif inner_str == ' Dr':
                modified_title_list.append(7)
            elif inner_str == ' Mme':
                modified_title_list.append(8)
            elif inner_str == ' Ms':
                modified_title_list.append(9)
            elif inner_str == ' Major':
                modified_title_list.append(10)
            elif inner_str == ' Lady':
                modified_title_list.append(11)
            elif inner_str == ' Sir':
                modified_title_list.append(12)
            elif inner_str == ' Mlle':
                modified_title_list.append(13)
            elif inner_str == ' Col':
                modified_title_list.append(14)
            elif inner_str == ' Capt':
                modified_title_list.append(15)
            elif inner_str == ' the Countess':
                modified_title_list.append(16)
            else:
                modified_title_list.append(17)


        return np.array(modified_title_list)


    def swap_columns(df, c1, c2):
        df['temp'] = df[c1]
        df[c1] = df[c2]
        df[c2] = df['temp']
        df.drop(columns=['temp'], inplace=True)
        col_list = list(df)
        col_list[0], col_list[1] = col_list[1], col_list[0]
        df.columns = col_list


    df_feature_name = df_fun.iloc[:, 1]
    df1 = pd.DataFrame(split_title(df_feature_name), columns = ['Name'])
    df2 = x_test.drop("Name", axis=1)


    df_modified_after_name = pd.concat([df1, df2], axis=1)
    swap_columns(df_modified_after_name, 'Name', 'Pclass')
    return df_modified_after_name


df_modified_after_name = name_feature_represntation(df_features)                #### name_feature_customization

##############################################################################


######################## sex feature customization ###########################

def sex_feature_represntation(strings, df_modified_after_name_func):
    sex_feature = []
    for string in strings:
        if string=="male":
            sex_feature.append(1)
        else:
            sex_feature.append(2)

    df1 = pd.DataFrame(np.array(sex_feature), columns = ['Sex'])
    df2 = df_modified_after_name_func.drop("Sex", axis=1)
    return pd.concat([df1, df2], axis=1)


df_modified_after_sex = sex_feature_represntation(df_features.iloc[:, 2], df_modified_after_name)       #### sex feature customization

##############################################################################



########################################################################################################################





############################# droping insignificant columns and data normalization #####################################


x_train = df_modified_after_sex.drop("Ticket", axis=1)
x_train = x_train.drop("Cabin", axis=1)
x_train = x_train.drop("Embarked", axis=1)


x_train.fillna(x_train.mean(), inplace=True)                            #### filling the NaN value
x_train=(x_train-x_train.min())/(x_train.max()-x_train.min())           #### normalizing using min-max


########################################################################################################################




#################################### Test Dataset Representation #######################################################

test_name_feature = name_feature_represntation(x_test)
test_sex_feature = sex_feature_represntation(x_test.iloc[:, 2], test_name_feature)
x_test = test_sex_feature.drop("Ticket", axis=1)
x_test = x_test.drop("Cabin", axis=1)
x_test = x_test.drop("Embarked", axis=1)

x_test.fillna(x_test.mean(), inplace=True)                            #### filling the NaN value
x_test=(x_test-x_test.min())/(x_test.max()-x_test.min())              #### normalizing using min-max


########################################################################################################################




##################################### building and compiling the model #################################################

def create_model(units_hidden_layers, input_shape):
    model = models.Sequential()
    model.add(layers.Dense(units_hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    hidden_layers = len(units_hidden_layers)
    for i in range(1, hidden_layers - 1):
        model.add(layers.Dense(units_hidden_layers[i], activation='relu'))

    model.add(layers.Dense(units_hidden_layers[hidden_layers - 1], activation='sigmoid'))
    return model



units_in_h_l = [64, 32, 1]
input_shape = 7
model = create_model(units_in_h_l, input_shape)                 #### creating model


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, df_labels, epochs=20, batch_size=128)



########################################################################################################################





########################################## making the predictions ######################################################

test_pred = pd.DataFrame(model.predict(x_test, batch_size=128))
test_pred.index.name = 'PassengerId'
test_pred = test_pred.rename(columns = {0: 'Survived'}).reset_index()
test_pred['PassengerId'] = test_pred['PassengerId'] + 892
test_pred['Survived'] = (test_pred['Survived'] > 0.7).astype(int)

test_pred.to_csv('titanic_submission.csv', index = False)


########################################################################################################################

"""#### spliting dataset to train and test
x_train, x_validation, y_train, y_validation = train_test_split(df_features, df_labels, test_size=0.2, random_state=1223)
x_train = x_train.as_matrix().reshape(33600,784)
x_validation = x_validation.as_matrix().reshape(8400, 784)
x_test = x_test.as_matrix().reshape(28000, 784)


#### data normalization
x_train = x_train.astype('float32') / 255
x_validation = x_validation.astype('float32') / 255
x_test = x_test.astype('float32') / 255


#### encoding lebels
y_train = to_categorical(y_train)
y_validation = to_categorical(y_validation)


#### creating the model
#model = models.Sequential()
#model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
#model.add(layers.Dense(10, activation='softmax'))







def create_model(units_hidden_layers, input_shape):
    model = models.Sequential()
    model.add(layers.Dense(units_hidden_layers[0], activation='relu', input_shape=(input_shape,)))
    hidden_layers = len(units_hidden_layers)
    for i in range(1, hidden_layers - 1):
        model.add(layers.Dense(units_hidden_layers[i], activation='relu'))

    model.add(layers.Dense(units_hidden_layers[hidden_layers - 1], activation='softmax'))
    return model


#### generating model using create_model() function
units_in_h_l = [512, 256, 128, 64, 10]
input_shape = 784
model = create_model(units_in_h_l, input_shape)


#print(model.summary())


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=200, batch_size=256, validation_data=(x_validation, y_validation))
print(history.history['acc'])


test_pred = pd.DataFrame(model.predict(x_test, batch_size=256))
test_pred = pd.DataFrame(test_pred.idxmax(axis=1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1

test_pred.to_csv('mnist_submission.csv', index = False)"""
