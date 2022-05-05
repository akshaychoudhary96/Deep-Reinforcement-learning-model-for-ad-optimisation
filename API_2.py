# import main Flask class and request object
from crypt import methods
from pyexpat import model
import uuid
from flask import Flask, request, jsonify

#libraries for DRL part 
from collections import namedtuple
from numpy.random import uniform as U
import pandas as pd
import numpy as np
import io
import requests
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
import random

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

#functions for DRL

def get_ad_inventory():
    
    ad_inv_prob = {'morning': 0.9, 
                   'afternoon':  0.7, 
                   'evening':  0.7, 
                   'night':  0.9}
    ad_inventory = []
    for level, prob in ad_inv_prob.items():
        if U() < prob:
            ad_inventory.append(level)
    # Make sure there are at least one ad
    if not ad_inventory:
        ad_inventory = get_ad_inventory()
    return ad_inventory

def get_ad_click_probs():
    base_prob = 0.8
    delta = 0.3
    
    time_of_day = {'morning':1,'afternoon':2,'evening':3,'night':4}                
    

    ad_click_probs = {l1: {l2: max(0, base_prob - delta * abs(time_of_day[l1]- time_of_day[l2])) for l2 in time_of_day}
                           for l1 in time_of_day}

    return ad_click_probs


def display_ad(ad_click_probs, user, ad):
    prob = ad_click_probs[ad][user['time_of_day']]
    click = 1 if U() < prob else 0
    return click

def calc_regret(user, ad_inventory, ad_click_probs, ad_selected):
    this_p = 0
    max_p = 0
    for ad in ad_inventory:
        p = ad_click_probs[ad][user['time_of_day']]

        if ad == ad_selected:
            this_p = p
        if p > max_p:
            max_p = p
    regret = max_p - this_p
    return regret

def get_model(n_input, dropout):
    inputs = keras.Input(shape=(n_input,))
    x = Dense(256, activation='relu')(inputs)
    if dropout > 0:
        x = Dropout(dropout)(x, training=True)
    x = Dense(256, activation='relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x, training=True)
    phat = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, phat)
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.binary_accuracy])
    return model

def update_model(model, X, y):
    
    X = np.array(X)
    
    X = X.reshape((X.shape[0], X.shape[2]))
    
    y = np.array(y).reshape(-1)
    
    model.fit(X, y, epochs=40)
    
    return model

def ad_to_one_hot(ad):
   
    ad_levels = ['morning','afternoon','evening','night']
    
    ad_input = [0] * len(ad_levels)
    
    if ad in ad_levels:
    
        ad_input[ad_levels.index(ad)] = 1
    
    return ad_input


def select_ad(model, context, ad_inventory):
    selected_ad = None
    selected_x = None
    max_action_val = 0
    for ad in ad_inventory:
        ad_x = ad_to_one_hot(ad)
        x = np.array(context + ad_x).reshape((1, -1))
        action_val_pred = model.predict(x)[0][0]
        if action_val_pred >= max_action_val:
            selected_ad = ad
            selected_x = x
            max_action_val = action_val_pred
    return selected_ad, selected_x

def generate_user(df_data):
    
    user = df_data.sample(1)
    context = user.iloc[:, :-1].values.tolist()[0]
    return user.to_dict(orient='records')[0], context

url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
s=requests.get(url).content
names = ['age', 
           'workclass', 
           'fnlwgt', 
           'time_of_day',
           'education_num',
           'marital_status',
           'occupation',
           'relationship',
           'race',
           'gender',
           'capital_gain',
           'capital_loss',
           'hours_per_week',
           'native_country',
          'income']
usecols = ['age', 
           'time_of_day',
           'gender',
           'native_country'
           ]
df_census = pd.read_csv(io.StringIO(s.decode('utf-8')), 
                        sep=',',
                        skipinitialspace=True,
                        names=names,
                        header=None,
                        usecols=usecols)

# Cleanup
df_census = df_census.replace('?', np.nan).dropna()
edu_map = {'Preschool': 'morning',
           '1st-4th': 'morning',
           '5th-6th': 'morning',
           '7th-8th': 'morning',
           '9th': 'night',
           '10th': 'night',
           '11th': 'night',
           '12th': 'night',
           'Some-college': 'afternoon',
           'Bachelors': 'afternoon',
           'Assoc-acdm': 'afternoon',
           'Assoc-voc': 'afternoon',
           'Prof-school': 'evening',
           'Masters': 'evening',
           'Doctorate': 'evening',
           'HS-grad':'evening'}

for from_level, to_level in edu_map.items():
    df_census.time_of_day.replace(from_level, to_level, inplace=True)
# Convert raw data to processed data
context_cols = [c for c in usecols if c != 'time_of_day']
df_data = pd.concat([pd.get_dummies(df_census[context_cols]),
           df_census['time_of_day']], axis=1)


#implementation of DRL

def simulation():
    ad_click_probs = get_ad_click_probs()
    df_cbandits = pd.DataFrame()
    dropout_levels = [0, 0.01, 0.05, 0.1, 0.2, 0.4]
    # dropout_levels = [0.01,0.05]
    for d in dropout_levels:
        print("Trying with dropout:", d)
        np.random.seed(0)
        context_n = df_data.shape[1] - 1
        ad_input_n = df_data.time_of_day.nunique()
        print(ad_input_n)
        model = get_model(context_n + ad_input_n, 0.01)
        X = []
        y = []
        regret_vec = []
        total_regret = 0
        for i in range(5000):
            if i % 20 == 0:
                print("# of impressions:", i)
        
            user, context = generate_user(df_data)
            ad_inventory = get_ad_inventory()
            
            ad, x = select_ad(model, context, ad_inventory)
            
            click = display_ad( ad_click_probs,user, ad)
            regret = calc_regret(user,ad_inventory, ad_click_probs, ad)
            total_regret += regret
            regret_vec.append(total_regret)
            X.append(x)
            y.append(click)
            
            if (i + 1) % 500 == 0:
            
                model = update_model(model, X, y)
                X = []
                y = []
                
        
        return model


# create the Flask app
app = Flask(__name__)

context_ads = {1:{"userID": 1,
    "name" : "Akshay",
    "location" : "India",
    "age" : 25,
    "gender" : "Male",
    "time_of_day" : "morning",
    "predicted_ad":"",
    "reward":""}}

@app.route('/get-ad/<int:userID>',methods=['GET'])
def get_ad(userID):
    obj = {}
    for i,items in context_ads.items():
        if items["userID"]==userID:
            obj[i]= items["predicted_ad"]
            return jsonify(obj)

@app.route('/load-simulation',methods=['POST'])
def load_simulation():
    global test_model 
    test_model = simulation()

    return jsonify('Simulation model is loaded'),201

@app.route('/get-contexts', methods=['GET'])
def get_contexts():
    return jsonify({'contexts': context_ads})
    
@app.route('/update_feedback/<int:sessionid>',methods=['PUT'])
def update_feedback(sessionid):
    for i,item in context_ads.items():
        if i == sessionid:
            item["reward"]= request.json["click"]
    

    return jsonify({'Updated contexts for Session ID':context_ads[sessionid]})


@app.route('/user-context', methods=['POST'])
def create_contextual_data():
    request_data = request.get_json()

    user_id = request_data['userID']
    name = request_data['name']
    location = request_data['location']
    time_of_day = request_data['time_of_day']
    age = request_data['age']

    gender = request_data['gender']

    sessionid = uuid.uuid1().int

    contextual_data = {
        "userID": user_id,
        "name": name,
        "location": location,
        "age": age,
        "gender": gender,
        "time_of_day":time_of_day,
        "predicted_ad":"",
        "reward":""
    }
    col = {"age":0 , "gender_Female":0 , "gender_Male":0 , "native_country_Cambodia":0 , "native_country_Canada":0 , "native_country_China":0 , "native_country_Columbia":0 , "native_country_Cuba":0 , "native_country_Dominican-Republic":0 , "native_country_Ecuador":0 , "native_country_El-Salvador":0 , "native_country_England":0 , "native_country_France":0 , "native_country_Germany":0 , "native_country_Greece":0 , "native_country_Guatemala":0 , "native_country_Haiti":0 , "native_country_Holand-Netherlands":0 , "native_country_Honduras":0 , "native_country_Hong":0 , "native_country_Hungary":0 , "native_country_India":0 , "native_country_Iran":0 , "native_country_Ireland":0 , "native_country_Italy":0 , "native_country_Jamaica":0 , "native_country_Japan":0 , "native_country_Laos":0 , "native_country_Mexico":0 , "native_country_Nicaragua":0 , "native_country_Outlying-US(Guam-USVI-etc)":0 , "native_country_Peru":0 , "native_country_Philippines":0 , "native_country_Poland":0 , "native_country_Portugal":0 , "native_country_Puerto-Rico":0 , "native_country_Scotland":0 , "native_country_South":0 , "native_country_Taiwan":0 , "native_country_Thailand":0 , "native_country_Trinadad&Tobago":0 , "native_country_United-States":0 , "native_country_Vietnam":0 , "native_country_Yugoslavia":0 ,"time_of_day":0}
    df_data = pd.DataFrame(col,index=[0])

    dummy_gender = 'gender_'+gender
    dummy_native_country = 'native_country_'+location
    for key,values in col.items():
        if key==dummy_gender:
            df_data[key]=1
        elif key==dummy_native_country:
            df_data[key]=1
        elif key=='age':
            df_data[key]=age
        elif key== 'time_of_day':
            df_data[key]= time_of_day
        else:
            df_data[key]=0
    
    context = df_data.iloc[:, :-1].values.tolist()[0]
    
    ad_inventory = get_ad_inventory()
    
    ad, x = select_ad(test_model, context, ad_inventory)
    
    ad_displayed = ''
    if ad=='morning':
        ad_displayed= 'A'
    elif ad == 'afternoon':
        ad_displayed = 'B'
    elif ad == 'evening':
        ad_displayed = 'C'
    else:
        ad_displayed = 'D'
    contextual_data["predicted_ad"] = ad_displayed
    context_ads[sessionid] = contextual_data

    return jsonify({'data': contextual_data}),201

if __name__ == '__main__':
    
    app.run(debug=True, port=8000)
