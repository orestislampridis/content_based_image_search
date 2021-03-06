import os

import cv2
import pandas.io.sql as sqlio
import psycopg2 as pq
from flask import Flask, render_template, request, jsonify
from skimage import io

from ColorDescriptor import ColorDescriptor
from Searcher import Searcher
from ShapeDescriptor import ShapeDescriptor

app = Flask(__name__, static_url_path='/static')

db_URL = os.environ.get('DATABASE_URL')


# main route
@app.route('/')
def index():
    global cn
    cr = 2
    try:
        cn = pq.connect(db_URL)
    except (Exception, pq.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    return render_template('index.html')


# search route
@app.route('/search', methods=['POST'])
def search():
    if request.method == "POST":
        RESULTS_ARRAY = []
        image_names = []

        # get url
        image_url = request.files['file_image']
        method = request.form.get('method')
        distance = request.form.get('distance')
        number_of_neighbors = request.form.get('knn_slider')

        try:
            # load the query image and describe it
            img = io.imread(image_url)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            results = list()

            if method == "color":
                cr = cn.cursor()
                sql = 'SELECT orig_filename, color_descriptor FROM files;'
                cr.execute(sql)
                tmp = cr.fetchall()
                df = sqlio.read_sql_query(sql, cn)
                #sampled_df = df.sample(n=100, random_state=42)

                cd = ColorDescriptor((8, 12, 3))
                color_features = cd.describe(img)
                searcher = Searcher(color_features, method=method, distance=distance, limit=int(number_of_neighbors),
                                    dataframe=df)
                results = searcher.search()
            else:
                sd = ShapeDescriptor(32)
                kaze_features, orb_features = sd.describe(img)
                if method == "kaze":
                    cr = cn.cursor()
                    sql = 'SELECT orig_filename, kaze FROM files;'
                    cr.execute(sql)
                    tmp = cr.fetchall()
                    df = sqlio.read_sql_query(sql, cn)
                    #sampled_df = df.sample(n=100, random_state=42)

                    searcher = Searcher(kaze_features, method, distance, limit=int(number_of_neighbors),
                                        dataframe=df)
                    results = searcher.search()
                if method == "orb":
                    cr = cn.cursor()
                    sql = 'SELECT orig_filename, orb FROM files;'
                    cr.execute(sql)
                    tmp = cr.fetchall()
                    df = sqlio.read_sql_query(sql, cn)
                    #sampled_df = df.sample(n=100, random_state=42)

                    searcher = Searcher(orb_features, method, distance, limit=int(number_of_neighbors),
                                        dataframe=df)
                    results = searcher.search()

            # loop over the results, displaying the score and image name
            for (score, resultID) in results:
                RESULTS_ARRAY.append(
                    {"image": str(resultID), "score": str(score)})
                image_names.append(resultID)

            # return success
            return render_template("results.html", image_names=image_names[:int(number_of_neighbors)])

        except:
            # return error
            jsonify({"sorry": "Sorry, no results! Please try again."}), 500


@app.route('/<filename>')
def send_image(filename):
    print(filename)
    path = filename
    start = "images/"
    relative_path = os.path.relpath(path, start)
    print(relative_path)
    # return send_from_directory("/static/image", relative_path)
    return relative_path


if __name__ == "__main__":
    app.run(debug=True)
