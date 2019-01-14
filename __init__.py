from flask import Flask, render_template, redirect, url_for, request
import sklearn.datasets as dataset
from flask.json import jsonify
import seaborn as sns

app = Flask(__name__)


@app.route('/')
def index():
    list_content = [
        {"href": "/linear-regression", "content": "Linear Regression",
         "icon": "mif-chart-dots mif-lg "},
        {"href": "/logistic-regression", "content": "Logistic Regression",
         "icon": "fas fa-braille"}
    ]
    return render_template('index.html', list_content=list_content)


@app.route('/linear-regression')
def liner_regression():
    return render_template('linear_regression.html')


@app.route('/linear-regression/getDataSet')
def getDataSet():
    dataID = request.args.get('data')

    print(dataID)
    dataSet = None
    response = {}
    if dataID == "1":
        dataSet = dataset.load_boston()
        response = {
            'data': dataSet.data[0:10:, 0:].tolist(),
            'feature': dataSet.feature_names.tolist(),
            'description': dataSet.DESCR,
            'target': dataSet.target[0:10].tolist()
        }
    if dataID == "2":
        print("213")
        dataSet = dataset.load_diabetes()
        response = {
            'data': dataSet.data[0:10:, 0:].tolist(),
            'feature': dataSet.feature_names,
            'description': dataSet.DESCR,
            'target': dataSet.target[0:10].tolist()
        }

    return jsonify(response)

@app.route('/linear-regression/plot')
def plotDateSet():
    #sns_plot = sns.pairplot(df, hue='species', size=2.5)
    #sns_plot.savefig("output.png")
    return None

if __name__ == "__main__":
    app.run(debug=True)
