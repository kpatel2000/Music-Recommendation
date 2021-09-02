from flask import Flask,request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)     

KMeansModel = pickle.load(open('KMeansModel.pkl','rb'))
KNNModel = pickle.load(open('KnnModel.pkl','rb'))

@app.route('/')
def hello_world():   
    return render_template("music.html")

@app.route('/recommend',methods=['POST']) 
def recommend():                            #Get Recommendation of Songs       
    song_name = [str(x) for x in request.form.values()]
    features =get_features(song_name[0])
    if features == -1:
        return render_template('music.html', pred = "Song Not Found")    
    features = np.array([features])
    cluster_number = KMeansModel.predict(features)
    print(cluster_number)
    result = ClusterIndicesNumpy(cluster_number, KMeansModel.labels_)
    n = 10
    inter_neigh=set()
    while len(inter_neigh) < 7:
        neighbor = KNNModel.kneighbors(features,n_neighbors=n, return_distance=False)
        neighbor = neighbor.flatten()
        inter_neigh = np.intersect1d(neighbor, result)
        n += 100

    recommended_list = getNames(inter_neigh[:7])
    recommended_list = [song.capitalize() for song in recommended_list] 
    print(recommended_list)
    return render_template('music.html',lst = recommended_list)
    

def ClusterIndicesNumpy(clustNum, labels_array): 
    return np.where(labels_array == clustNum)[0]

def get_features(song_name):        # Getting Features on basis of Song Name
    song_data = pd.read_csv('song_data.csv')
    for song in song_data.index:
        if song_data['name'][song] == song_name:
            pred_data = [song_data['genre_ids'][song],song_data['language'][song]]
            break
        else:
            pred_data = -1
    return pred_data        

def getNames(index_list):
    song_data = pd.read_csv('song_data.csv')
    names = []
    for value in index_list:
        print(song_data['name'][value])
        names.append(song_data['name'][value])
        print(names)
        
    return names

if __name__ == '__main__':
    app.run(debug=True)
