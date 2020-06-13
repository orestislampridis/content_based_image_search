import argparse
import glob
import sys

import cv2
import pandas.io.sql as sqlio
import psycopg2 as pq

from ColorDescriptor import ColorDescriptor
from Searcher import Searcher
from ShapeDescriptor import ShapeDescriptor


def main(argv):
    global results
    parser = argparse.ArgumentParser()
    parser_action = parser.add_mutually_exclusive_group(required=True)
    parser_action.add_argument("--store", action='store_const', const=True,
                               help="Load all images in path and save them in the DB")
    parser_action.add_argument("--search", action='store_const', const=True,
                               help="Give an image path and search for most similar images")
    parser.add_argument("filename", help="Path to directory where to get images from")

    args = parser.parse_args(argv[1:])

    # Initialize the color descriptor
    cd = ColorDescriptor((8, 12, 3))

    # Initialize the shape descriptors
    sd = ShapeDescriptor(32)

    try:
        connection = pq.connect(user="postgres",
                                password="1234",
                                host="127.0.0.1",
                                port="5432",
                                database="imagesdb",
                                sslmode="disable")

    except (Exception, pq.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    # Ensure DB structure is present
    cursor = connection.cursor()
    cursor.execute("SELECT 1 FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
                   ('public', 'files'))
    result = cursor.fetchall()

    # If db is empty create table
    if len(result) == 0:
        create_table_query = """
            CREATE TABLE files (
                id serial primary key,
                orig_filename text not null,
                color_descriptor numeric[] not null,
                sift numeric[] not null,
                surf numeric[] not null,
                kaze numeric[] not null,
                orb numeric[] not null
            )
            """

        cursor.execute(create_table_query)
        connection.commit()

    # Run the command
    if args.store:
        # Reads all files in path into memory.
        path = args.filename + "/*.jpg"
        for fname in glob.glob(path):
            print(fname)
            f = open(fname, 'rb')
            image = cv2.imread(fname)

            # describe the image by using our descriptors
            try:
                color_features = cd.describe(image)
                kaze_features, orb_features = sd.describe(image)
            except ValueError:
                continue

            cursor.execute("INSERT INTO files(id, orig_filename, color_descriptor, kaze, orb)"
                           "VALUES (DEFAULT,%s,%s,%s,%s) RETURNING id",
                           (fname, color_features, kaze_features, orb_features))

            returned_id = cursor.fetchone()[0]
            f.close()
            connection.commit()
            print("Stored {0} into DB record {1}".format(args.filename, returned_id))

    elif args.search:
        # Fetches the file from the DB into memory then writes it out.
        # Same as for store, to avoid that use a large object.
        print(args.filename)

        image = cv2.imread(args.filename)

        # Initialize the searcher
        method = 'kaze'
        distance = 'euclidean'

        # Number of nearest neighbors
        limit = 10

        if method == "color":
            cr = connection.cursor()
            sql = 'SELECT orig_filename, color_descriptor FROM files;'
            cr.execute(sql)
            tmp = cr.fetchall()
            df = sqlio.read_sql_query(sql, connection)
            # sampled_df = df.sample(n=100, random_state=42)

            color_features = cd.describe(image)
            searcher = Searcher(color_features, method=method, distance=distance, limit=limit,
                                dataframe=df)
            results = searcher.search()
        else:

            if method == "kaze":
                nsd = ShapeDescriptor(64)
                kaze_features, _ = nsd.describe(image)

                cr = connection.cursor()
                sql = 'SELECT orig_filename, kaze FROM files;'
                cr.execute(sql)
                tmp = cr.fetchall()
                df = sqlio.read_sql_query(sql, connection)
                # sampled_df = df.sample(n=100, random_state=42)

                searcher = Searcher(kaze_features, method, distance, limit=limit,
                                    dataframe=df)
                results = searcher.search()
            if method == "orb":
                nsd = ShapeDescriptor(128)
                _, orb_features = nsd.describe(image)

                cr = connection.cursor()
                sql = 'SELECT orig_filename, orb FROM files;'
                cr.execute(sql)
                tmp = cr.fetchall()
                df = sqlio.read_sql_query(sql, connection)

                searcher = Searcher(orb_features, method, distance, limit=limit,
                                    dataframe=df)
                results = searcher.search()

        # Print the results in console
        print(results)

        # Load the query image and display it
        cv2.imshow("Query", image)

        # Loop over the results
        for (score, resultID) in results:
            # Load the result image and display it
            result = cv2.imread("static/" + resultID)
            cv2.imshow("Result", result)
            cv2.waitKey(0)

    connection.close()


if __name__ == '__main__':
    main(sys.argv)
