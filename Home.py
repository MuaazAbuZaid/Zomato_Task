# importing necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Page content
st.markdown(" <center>  <h1> Zomato </h1> </font> </center> </h1> ",
            unsafe_allow_html=True)

st.markdown(''' <center>  <h6>
    This app is created to predict successful restaurants  </center> </h6> ''', unsafe_allow_html=True)

# the path of a photo provided from the path of this file
st.image('Sources/zomato.jpeg', use_column_width=True)

# titles
st.markdown(" <center>  <h2> Enter Your Data </h2> </font> </center> </h2> ",
            unsafe_allow_html=True)

# reading data
df = pd.read_csv('Sources/cleaned_data.csv')

# page content
# radio buttons for binary classes
bin_class = ['Yes', 'No']
online_book_cont = st.container()
online_order_col, book_table_col = online_book_cont.columns(2)
with online_order_col:
    online_order = st.radio(
                "Online order",
                bin_class,
                horizontal= True,
            )
with book_table_col:
    book_table = st.radio(
                "Book table",
                bin_class,
                horizontal= True,
            )
    
# select box for location
locations = tuple(df.location.unique()) 
location = st.selectbox('Select city', locations)

# select box for restaurant type
rest_types = tuple(df.rest_type.unique()) 
rest_type = st.selectbox('Select the type of restaurant', rest_types)

# select box for restaurant style
types = tuple(df.type.unique()) 
type = st.selectbox('Select the style of restaurant', types)

# slider for average cost per person
avg_cost = st.slider('Average cost per person', 0, int(10), int((df.avg_cost.max())))

# multi select box for dish_liked
df.dish_liked = df.dish_liked.str.split(', ')
# use the explode method to create a new DataFrame with one word per row
df_dish_liked = df['dish_liked'].explode()
# use the unique method to get the unique dish_liked in the column
unique_dish_liked = list(df_dish_liked.unique())
dish_liked = st.multiselect('Dish you suppose will be liked', unique_dish_liked)
dish_liked = ', '.join(dish_liked)

# multi select box for cuisines
df.cuisines = df.cuisines.str.split(', ')
# use the explode method to create a new DataFrame with one word per row
df_cuisines = df['cuisines'].explode()
# use the unique method to get the unique cuisines in the column
unique_cuisines = list(df_cuisines.unique())
cuisines = st.multiselect('Cuicines', unique_cuisines)
cuisines = ', '.join(cuisines)
# st.write(dish_liked)

# functions used in transformer
bin_class_cols = ['online_order', 'book_table']
def bin_class(value):
    return 1 if value == 'Yes' else 0
df.online_order = df.online_order.apply(bin_class)
df.book_table = df.book_table.apply(bin_class)

ohenc_col = ['type', 'location', 'rest_type']
df = pd.get_dummies(df, columns= ohenc_col)

mlb_cols = ['cuisines','dish_liked']
mlb = MultiLabelBinarizer()
def mlb_trans_train(df, mlb_cols):
    for col in mlb_cols:
        # df[col] = df[col].str.split(', ')
        enc_column = pd.DataFrame(mlb.fit_transform(df[col]),
                        columns=mlb.classes_,
                        index=df[col].index)
        df = df.drop([col], axis=1)
        df = pd.concat([df, enc_column], axis=1, sort=False)
    df = df.loc[:, ~df.columns.duplicated()]    
    return df
def mlb_trans_sample(df, mlb_cols):
    for col in mlb_cols:
        df[col] = df[col].str.split(', ')
        enc_column = pd.DataFrame(mlb.fit_transform(df[col]),
                        columns=mlb.classes_,
                        index=df[col].index)
        df = df.drop([col], axis=1)
        df = pd.concat([df, enc_column], axis=1, sort=False)
    df = df.loc[:, ~df.columns.duplicated()]    
    return df
df = mlb_trans_train(df, mlb_cols)

# splitting data
X = df.drop('success', axis= 1)
y = df['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.15, random_state= 10)

sc_col = ['avg_cost']
sc = RobustScaler()
sc.fit(pd.DataFrame(X_train.avg_cost))

# transformer function
def transform(X_train, test_sample, bin_class_cols, bin_class, ohenc_col, mlb_cols, mlb_trans, sc_col, sc):

    # binary class manipulation function 
    for col in bin_class_cols:
        test_sample[col] = test_sample[col].apply(bin_class)

    # one hot encoding function    
    test_sample = pd.get_dummies(test_sample, columns= ohenc_col)

    # multi label binarizer function
    test_sample = mlb_trans(test_sample, mlb_cols)

    # robust scaling function
    test_sample[sc_col] = sc.transform(pd.DataFrame(test_sample[sc_col]))

    # handling columns that don't exist in the saple dataframe
    missing_cols = set(X_train.columns) - set(test_sample.columns)
    for col in missing_cols:
        test_sample[col] = 0

    return test_sample # as a value 
 
# model
m = pickle.load(open('DT.pkl', 'rb'))

# show the prediction when pressing the button
if st.button('Predict'):
    
    test_sample = pd.DataFrame({'online_order': online_order, 'book_table': book_table, 'location': location, 'rest_type': rest_type, 
                            'avg_cost': avg_cost, 'type': type, 'dish_liked': dish_liked, 'cuisines': cuisines}, index= [0])        
    res_smpl = transform(X_train, test_sample, bin_class_cols, bin_class, ohenc_col, mlb_cols, mlb_trans_sample, sc_col, sc)
    prediction = m.predict(res_smpl)[0]
    # yes means successful restaurant
    st.write('Yes') if prediction == 1 else st.write('No')
