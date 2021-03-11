from flask import Flask, render_template, request, url_for
import numpy as np
import pickle


model = pickle.load(open('Final_project_model.p','rb'))


#print(model.predict([[10925,0.512,17223497,8.6,3.2,1,0,0,6.343079e-04]]))


app = Flask(__name__)

@app.route("/")
def page():
    return render_template("base.html")

@app.route('/predict')
def man():
    return render_template('home.html')

@app.route('/prediction', methods=["POST"])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


@app.route("/dataset")
def dataset():
    return render_template("output_final.html")

@app.route("/useful_links")
def useful_links():
    return render_template("useful_links.html")

@app.route("/about_us")
def about_us():
    return render_template("about_us.html")


if __name__ == "__main__":
    app.run(debug=True)