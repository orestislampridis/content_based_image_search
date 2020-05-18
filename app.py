import os

from flask import Flask, render_template, request, jsonify, send_from_directory

from ColorDescriptor import ColorDescriptor
from Searcher import Searcher
from ShapeDescriptor import ShapeDescriptor

app = Flask(__name__, static_url_path="/static")


# main route
@app.route('/')
def index():
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
            # initialize the image and shape descriptors
            cd = ColorDescriptor((8, 8, 8))
            sd = ShapeDescriptor(32)

            # load the query image and describe it
            from skimage import io
            import cv2
            query = io.imread(image_url)
            query = (query * 255).astype("uint8")
            # (r, g, b) = cv2.split(query)
            # query = cv2.merge([b, g, r])

            results = list()

            if method == "color":
                color_features = cd.describe(query)
                searcher = Searcher(color_features, method=method, distance=distance, limit=int(number_of_neighbors))
                results = searcher.search()
            else:
                sift_features, surf_features, kaze_features, orb_features = sd.describe(query)
                if method == "sift":
                    searcher = Searcher(sift_features, method, distance, limit=int(number_of_neighbors))
                    results = searcher.search()
                if method == "surf":
                    searcher = Searcher(surf_features, method, distance, limit=int(number_of_neighbors))
                    results = searcher.search()
                if method == "kaze":
                    searcher = Searcher(kaze_features, method, distance, limit=int(number_of_neighbors))
                    results = searcher.search()
                if method == "orb":
                    searcher = Searcher(orb_features, method, distance, limit=int(number_of_neighbors))
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


@app.route('/<path:filename>')
def send_image(filename):
    print(filename)
    path = filename
    start = "static/"
    relative_path = os.path.relpath(path, start)
    print(relative_path)
    return send_from_directory("images", relative_path)


if __name__ == "__main__":
    app.run(debug=True)
