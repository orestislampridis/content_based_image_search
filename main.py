import argparse
import glob
import sys

import cv2
import psycopg2 as pq

from ColorDescriptor import ColorDescriptor
from Searcher import Searcher
from ShapeDescriptor import ShapeDescriptor


def main(argv):
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
        method = 'color'
        distance = 'euclidean'

        color_features = cd.describe(image)
        kaze_features, orb_features = sd.describe(image)

        searcher = Searcher(color_features, method, distance, limit=10)

        # Perform the search using the current query
        results = searcher.search()

    connection.close()


if __name__ == '__main__':
    main(sys.argv)
