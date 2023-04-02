from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler
import pickle
#model.pkl -trained ml model

#Desirilze-read the binary file-ML model
clf=pickle.load(open('model.pkl','rb'))

#for getting range dexided on
#step 1 reading the datsset
import pandas as pd
df=pd.read_csv("SUV_Purchase.csv")

#step 2 feature Engineering - drop unecessay or unimportant features - simplifying the dataset
df=df.drop(['User ID','Gender'],axis =1)#axis=1 i.e columns  ....axis =0 i.e rows

#step 3- loading the data
#setting the data into input and output values
X=df.iloc[:,:-1].values #iloc==>index location 2D array
Y=df.iloc[:,-1:].values #2D array

#step 4 - Split dataset into training in test
#Training and Testing the dataset
#more data-Trainig; Less data-Testing datai.e Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


app=Flask(__name__)

@app.route('/')  #Annnotation triggers the methods following-defalut annotation that randers the 1st web page to the browser
def hello():
    return render_template('index.html')

#jinja2-template engine-which would be going to template folder and selecting the webpage-hence folder name should be template




@app.route('/predict',methods=['post','Get'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()]
    print(features)
    sst=StandardScaler().fit(X_train)


    output=clf.predict(sst.transform([features]))
    print(output)

    if output[0]==0:
        return render_template('index.html',pred=f'The person will not be able to purchase the SUV')
    else:
        return render_template('index.html',pred=f'The person will be able to purchase the SUV')

if __name__ =="__main__":
    app.run(debug=True)
