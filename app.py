import os

from flask import Flask, render_template, request, jsonify, send_from_directory

from ColorDescriptor import ColorDescriptor
from Searcher import Searcher
from ShapeDescriptor import ShapeDescriptor

app = Flask(__name__)


# main route
@app.route('/')
def index():
    return render_template('index.html')


db_URL = os.environ.get('DATABASE_URL')


# search route
@app.route('/search', methods=['POST'])
def search():
    if request.method == "POST":
        RESULTS_ARRAY = []
        image_names = []

        # get url
        # db_URL = 'postgres://lwonnzirfqjlrj:239cab6b982fbb869cbb8ca219e068622bbe0c3bb05da6c06af5837659e6bcec@ec2-46-137-177-160.eu-west-1.compute.amazonaws.com:5432/d9a5legl875uce'
        image_url = request.files['file_image']
        method = request.form.get('method')
        distance = request.form.get('distance')
        number_of_neighbors = request.form.get('knn_slider')

        try:
            # initialize the image and shape descriptors
            cd = ColorDescriptor((8, 12, 3))
            sd = ShapeDescriptor(32)

            # load the query image and describe it
            from skimage import io
            import cv2
            img = io.imread(image_url)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            results = list()

            if method == "color":
                color_features = cd.describe(img)
                searcher = Searcher(color_features, method=method, distance=distance, limit=int(number_of_neighbors),
                                    database_url=db_URL)
                results = searcher.search()
            else:
                kaze_features, orb_features = sd.describe(img)
                if method == "kaze":
                    searcher = Searcher(kaze_features, method, distance, limit=int(number_of_neighbors),
                                        database_url=db_URL)
                    results = searcher.search()
                if method == "orb":
                    searcher = Searcher(orb_features, method, distance, limit=int(number_of_neighbors),
                                        database_url=db_URL)
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
    return send_from_directory("images", relative_path)


if __name__ == "__main__":
    app.run(debug=True)
