import sqlite3, hashlib, time, math, psycopg2

class Database:

    def __init__(self):
        self.user = "postgres"
        self.password = "password"
        self.host = "172.16.15.75"
        self.port = 5432
        self.database = "faceripper"
        
        self.connectToDatabase()
        #self.createTables()

    def connectToDatabase(self):
        self.conn = psycopg2.connect(
            user = self.user,
            password = self.password,
            host = self.host,
            port = self.port,
            database = self.database
        )
        self.cursor = self.conn.cursor()

    def createTables(self):
        sql = "CREATE TABLE IF NOT EXISTS encodings (id text PRIMARY KEY, encoding cube)"
        label_sql = "CREATE TABLE IF NOT EXISTS labels (id text PRIMARY KEY, name text NOT NULL, label text NOT NULL, directory text NOT NULL)"
        index_sql = "CREATE INDEX IF NOT EXISTS encoding_cube ON encodings (encoding);"

        truncate_label = "TRUNCATE labels"
        truncate_encodings = "TRUNCATE encodings"

        self.cursor.execute(sql)
        self.cursor.execute(label_sql)
        self.cursor.execute(index_sql)
        self.cursor.execute(truncate_label)
        self.cursor.execute(truncate_encodings)

        self.conn.commit()

    def storeFaceEncoding(self, encoding, name, label, directory):
        try:
            id = self.sha256(label)
            print('[ DB ] Storing against id: {}'.format(id))
            #label_sql = "INSERT INTO labels (id, name, label, directory) VALUES ('{}', '{}', '{}', '{}')".format(id, name, label, directory)

            encoding_sql = "INSERT INTO encodings (id, encoding) VALUES ('{}', CUBE(array{}))".format(id, list(encoding))

            print(encoding_sql)
            
            #self.cursor.execute(label_sql)
            res = self.cursor.execute(encoding_sql)
            print(res)
            self.conn.commit()
        except Exception as e: print(e)

    def createFaceEncoding(self, encoding, label):
        try:
            id = self.sha256(label)
            encoding_sql = "INSERT INTO encodings (id, encoding) VALUES ('{}', CUBE(array{}))".format(id, list(encoding))
            self.cursor.execute(encoding_sql)
            self.conn.commit()
        except Exception as e: print(e)

    def compareFaceEncodings(self, encoding, threshold = 0.6):
        minFX = []
        maxFX = []
        for i in range(len(encoding)):
            enc = encoding[i]
            minFX.append(enc - 0.1 * abs(encoding[i]))
            maxFX.append(enc + 0.1 * abs(encoding[i]))

        sql = "SELECT labels.name AS name, 1 - MIN(sqrt(power(cube_distance(CUBE(array{}), encoding), 2))) as distance, count(labels.name) AS num_ref FROM encodings INNER JOIN labels ON labels.id = encodings.id GROUP BY name ORDER BY distance DESC LIMIT 100;".format(list(encoding))
        with open('select.sql', 'w+') as f:
            f.write(sql)
            f.close()
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        return results #print(results)

    def getLabel(self, id):
        sql = "SELECT name, label FROM labels WHERE id = '{}'".format(id)
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        return results

    def sha256(self, text):
        m = hashlib.sha256()
        m.update(str(text).encode('utf-8'))
        m.update(str(time.time()).encode('utf-8'))
        return m.hexdigest()