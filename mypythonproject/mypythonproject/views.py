import numpy as np
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

def home(request):
    return render (request, 'home.html')
def milkquality(request):
    return render (request, 'milkquality.html')
def mobileprice(request):
    return render (request, 'mobileprice.html')
def result(request):

    data = pd.read_csv(r'C:/Users/cayen/Desktop/Dataset/milk_quality.csv')

    x = data.drop('grade', axis=1)
    y = data['grade']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])


    pred = model.predict([[val1,val2,val3,val4,val5,val6,val7]])

    result1 = ""
    if pred == [0]:
        result1 = "Low"
    elif pred == [1]:
        result1 = "Medium"
    else:
        result1 = "High"
    return render(request,"milkquality.html",{"result2":result1})


def resulta(request):
    data = pd.read_csv(r'C:/Users/cayen/Desktop/Dataset/mobile_phone_price.csv')
    data = data.drop(['product_id'], axis=1)
    x = data.drop('price', axis=1)
    y = data['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    model = LinearRegression()
    model.fit(x_train, y_train)

    kaj1 = float(request.GET['p1'])
    kaj2 = float(request.GET['p2'])
    kaj3 = float(request.GET['p3'])
    kaj4 = float(request.GET['p4'])
    kaj5 = float(request.GET['p5'])
    kaj6 = float(request.GET['p6'])
    kaj7 = float(request.GET['p7'])
    kaj8 = float(request.GET['p8'])

    pred = model.predict(np.array([kaj1,kaj2,kaj3,kaj4,kaj5,kaj6,kaj7,kaj8]).reshape(1,-1))
    pred = round(pred[0])

    price = "The predicted price is $"+str(pred)

    return render(request, "mobileprice.html", {"resulta": price})

    #ang result sa mobile kay ma result1 tas ang result2 kay ma result 3 tas ang last na result 1 diri kay mahimong result2