import requests
import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

url = 'https://api.spacexdata.com/v4/'

def getBoosters(data):
  for x in data['rocket']:
    res = requests.get(url+ 'rockets/' + str(x)).json()
    BoosterVersion.append(res['name'])
     

def getLaunchSite(data):
    for x in data['launchpad']:
        response = requests.get(url + "launchpads/" + str(x)).json()
        Longitude.append(response['longitude'])
        Latitude.append(response['latitude'])
        LaunchSite.append(response['name'])
     

def getPayloadData(data):
    for load in data['payloads']:
        response = requests.get(url + "payloads/" + load).json()
        PayloadMass.append(response['mass_kg'])
        Orbit.append(response['orbit'])
     

def getCoreData(data):
    for core in data['cores']:
            if core['core'] != None:
                response = requests.get(url + "cores/" + core['core']).json()
                Block.append(response['block'])
                ReusedCount.append(response['reuse_count'])
                Serial.append(response['serial'])
            else:
                Block.append(None)
                ReusedCount.append(None)
                Serial.append(None)
            Outcome.append(str(core['landing_success'])+' '+str(core['landing_type']))
            Flights.append(core['flight'])
            GridFins.append(core['gridfins'])
            Reused.append(core['reused'])
            Legs.append(core['legs'])
            LandingPad.append(core['landpad'])

spacex_url="https://api.spacexdata.com/v4/launches/past"
response = requests.get(spacex_url)
     

data = pd.json_normalize(response.json())

data = data[['rocket', 'payloads', 'launchpad', 'cores', 'flight_number', 'date_utc']]

data = data[data['cores'].map(len)==1]
data = data[data['payloads'].map(len)==1]

data['cores'] = data['cores'].map(lambda x : x[0])
data['payloads'] = data['payloads'].map(lambda x : x[0])

BoosterVersion = []
PayloadMass = []
Orbit = []
LaunchSite = []
Outcome = []
Flights = []
GridFins = []
Reused = []
Legs = []
LandingPad = []
Block = []
ReusedCount = []
Serial = []
Longitude = []
Latitude = []

launch_dict = {
'FlightNumber': list(data['flight_number']),
'Date': list(data['date']),
'BoosterVersion':BoosterVersion,
'PayloadMass':PayloadMass,
'Orbit':Orbit,
'LaunchSite':LaunchSite,
'Outcome':Outcome,
'Flights':Flights,
'GridFins':GridFins,
'Reused':Reused,
'Legs':Legs,
'LandingPad':LandingPad,
'Block':Block,
'ReusedCount':ReusedCount,
'Serial':Serial,
'Longitude': Longitude,
'Latitude': Latitude}

df = pd.DataFrame.from_dict(launch_dict)

df = df[df['BoosterVersion']=='Falcon 9']

df.loc[:,'FlightNumber'] = list(range(1, df.shape[0]+1))

df['PayloadMass'].fillna(df['PayloadMass'].mean(), inplace=True)

dfa = df['Outcome'].value_counts()
bad_outcomes = dfa.keys()[[1,3,5,6,7]]
bad_outcomes

df['outcome_type'] = df['Outcome'].apply(lambda x: 0 if x in bad_outcomes else 1)

features = df.drop(['Date', 'Outcome', 'Year',  'LandingPad'], axis=1)
     

features.drop(['BoosterVersion'], axis = 1, inplace=True)

orbit_dummies = pd.get_dummies(features['Orbit'], prefix='Orbit', dtype=int)
features = pd.concat([features, orbit_dummies], axis=1)
features.drop(['Orbit'], axis=1, inplace=True)

lsite_dummies = pd.get_dummies(features['LaunchSite'], prefix='LaunchSite', dtype=int)
features = pd.concat([features, lsite_dummies], axis=1)
features.drop(['LaunchSite'], axis=1, inplace=True)

boolMap = {
    False : 0,
    True : 1
}

features['GridFins'] = features['GridFins'].map(boolMap)
features['Reused'] = features['Reused'].map(boolMap)
features['Legs'] = features['Legs'].map(boolMap)

features.drop(['Serial'], axis = 1, inplace=True)

Y = df['outcome_type']
X = features.drop(['outcome_type'], axis = 1)

X = transform.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

param_grids = {
    'Logistic Regression': {
              'C': [0.1, 1, 10],
              'penalty': ['l1', 'l2'],
              'solver': ['liblinear']
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    },
    'Decision Tree': {
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
}

best_estimators = {}
best_params = {}
best_scores = {}

for model_name in models.keys():
    grid_search = GridSearchCV(models[model_name], param_grids[model_name], cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model
    grid_search.fit(x_train, y_train)

    # Store the best model, parameters, and score
    best_estimators[model_name] = grid_search.best_estimator_
    best_params[model_name] = grid_search.best_params_
    best_scores[model_name] = grid_search.best_score_

finalmodel = LogisticRegression(C=0.1, penalty='l1', solver='liblinear')
finalmodel.fit(x_train, y_train)

model_pkl_file = "logreg.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(finalmodel, file)