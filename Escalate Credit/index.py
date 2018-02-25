from flask import Flask, render_template, request, redirect, Response

import psycopg2

path = "./encryptionkey.txt"
key = open(path, 'r')
import random, json

app = Flask(__name__)
app.config.from_pyfile('config.py')


@app.route('/')
def output():
    # serve index template
    return render_template('index.html', name='Joe')


@app.route('/receiver', methods=['POST'])
def worker():

    # read json + reply
    data = request.get_json()
    result = ''

    for item in data:
        # loop over every row
        result += str(item['make']) + '\n'
    try:
        connect_str = "dbname='testpython' user='hackutd' host='localhost' " + \
                      "password='HackUTDRHS'"
        # use our connection values to establish a connection
        conn = psycopg2.connect(connect_str)
        # create a psycopg2 cursor that can execute queries
        cursor = conn.cursor()
        # create a new table with a single column called "name"
        cursor.execute("""CREATE TABLE tutorials (name char(40));""")
        cursor.execute("INSERT INTO test (num, data) VALUES (pgp_sym_encrypt(%s, key), pgp_sym_encrypt(%s, key))", data[0], data[1])
        # run a SELECT statement - no data in there, but we can try it
        cursor.execute("""SELECT * from tutorials""")
        rows = cursor.fetchall()
        print(rows)
    except Exception as e:
        print("Uh oh, can't connect. Invalid dbname, user or password?")
        print(e)
    return result


if __name__ == '__main__':
    # run!
    app.run()
