
import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout



# Load the dataset and split into train and test
data = pd.read_csv('Credit Score Classification Dataset.csv')
data_train, data_test = train_test_split(data, train_size=0.8, test_size=0.2, random_state=100)
xtrain = data_train.drop('Credit Score', axis=1)
ytrain = data_train['Credit Score']
xtest = data_test.drop('Credit Score', axis=1)
ytest = data_test['Credit Score']

# Streamlit interface to get user inputs
st.title('Credit Score Classification')
st.write("User can enter their details to obtain their credit scores")

# Input features from the user
age = st.number_input('Enter your age', min_value=18, max_value=100, value=18, step=1)
gender = st.radio('Enter your gender', ['Male', 'Female'])
income = st.number_input('Enter your income', min_value=0, value=0, step=10000)
education = st.radio("Enter your qualification", ["Bachelor's Degree", "Master's Degree", "Doctorate", "High School Diploma", "Associate's Degree"])
marital_status = st.radio("Enter your marital status:", ["Single", "Married"])
children = st.number_input('Enter the number of children you have', min_value=0, value=0, step=1)
home = st.radio("Is your home owned or rented?", ["Owned", "Rented"])

# Add new row to xtest
newrow = [age, gender, income, education, marital_status, children, home]
xtest.loc[-1] = newrow
xtest.index = xtest.index + 1
xtest = xtest.sort_index()

# Print the last row of xtest
st.write("data entered successfully.")
#st.write(newrow)
#st.write(xtest.tail(1))

# Debugging checks
#st.write("Type of xtest:", type(xtest))  # Check the type of xtest
#st.write(xtest)# Check the columns of xtest
def neuralnet(x_train, x_test, y_train, y_test):#neural network code
 EPOCHS = 50
 BATCH_SIZE = 4
 VERBOSE = 1
 NB_CLASSES = 3
 N_HIDDEN = 64  # Increased the number of neurons
 VALIDATION_SPLIT = 0.2

 le = LabelEncoder()
 y_train = le.fit_transform(y_train)
 y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
 y_test = le.transform(y_test)
 y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
 x_train = pd.get_dummies(x_train, columns=['Gender', 'Education', 'Marital Status', 'Home Ownership'])
 x_test = pd.get_dummies(x_test, columns=['Gender', 'Education', 'Marital Status', 'Home Ownership'])
 x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

 scaler = StandardScaler()
 x_train = scaler.fit_transform(x_train)
 x_test = scaler.fit_transform(x_test)

 model = tf.keras.models.Sequential()
 model.add(Dense(N_HIDDEN, input_shape=(x_train.shape[1],), activation='relu', name='denselayer0'))
 model.add(Dropout(0.5))
 model.add(Dense(N_HIDDEN, activation='relu'))
 model.add(Dropout(0.5))
 model.add(Dense(NB_CLASSES, activation='softmax', name='outputlayer'))

 opt = tf.keras.optimizers.Adam(learning_rate=0.001)
 model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

 history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE,
                     validation_split=VALIDATION_SPLIT)

 #test_loss, test_acc = model.evaluate(x_test, y_test)
 y_pred = model.predict(x_test)
 print(type(y_pred))
 print(y_pred.shape)
 result = y_pred[0,:]
 #st.write(y_pred[0,:])
 #print(y_pred)
 #print('Test accuracy:', test_acc)
 keys = ['Low','High','Average']
 row_dict = dict(zip(keys, result))
 max_key = max(row_dict, key=row_dict.get)
 st.write(f'your credit score is {max_key}')

if st.button('final submit'):
 neuralnet(xtrain, xtest, ytrain, ytest)
