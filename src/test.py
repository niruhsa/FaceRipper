from db import Database
from urllib.parse import urlparse
from urllib import request
from multiprocessing import Pool
import cv2, argparse, face_recognition, numpy as np, os, json, time, math
from pprint import pprint
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

class Test:

    def __init__(self, **kwargs):
        self.db = Database()
        self.cascade = cv2.CascadeClassifier('res/haarcascade.xml')

        self.image = kwargs['image']
        self.config = kwargs['config']
        self.tolerance = kwargs['tolerance']
        self.workers = kwargs['threads']

        print('ok')

        software_names = [SoftwareName.CHROME.value]
        operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value]   

        self.user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)

        if self.config: self.loadFaces()
        else: self.detect()
    
    def detect(self):
        is_url = self.detectURL(self.image)
        if self.detectURL(self.image):
            img = self.URLToImage(self.image)
        else:
            img = cv2.imread(self.image)

        st = time.time()        
        faces = self.getFaces(img)

        encodings = []

        for face, coords in faces:
            encoding = face_recognition.face_encodings(face, num_jitters=0, model="large")
            if len(encoding) > 0:
                for enc in encoding: encodings.append((face, enc))

        for face, enc in encodings:
            start = time.time()
            results = self.db.compareFaceEncodings(enc, threshold = self.tolerance)
            pprint(results)
            print('[ OK ] Took {}s to query SQL'.format(time.time() - start))
        end = time.time()
        print('[ OK ] Took {}s in total post-download'.format(end - st))
        
    def getFaces(self, item, padding = 0):
        faces = self.cascade.detectMultiScale(item, 1.05, 1, minSize=(25, 25))
        
        ret_faces = []
        for (x, y, w, h) in faces:
            x = x - padding
            y = y - padding
            w = w + padding
            h = h + padding

            if x < 0: x = 0
            if y < 0: y = 0
            if x + w > item.shape[1]: w = item.shape[1] - x
            if y + h > item.shape[0]: h = item.shape[0] - y

			#cv2.imshow('title', item[y:y + h, x:x + w])
			#cv2.waitKey(0)
            image = item[y:y + h, x:x + w]
            image = cv2.resize(image, (224, 224))
            coords = [0, image.shape[1], image.shape[0], 0]
            ret_faces.append([image, coords])
        return ret_faces

    def URLToImage(self, url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = request.urlopen(request.Request(url, data=None, headers = { "User-Agent": self.user_agent_rotator.get_random_user_agent() }))
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image

    def detectURL(self, url):
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc: return True
        else: return False

    def loadFaces(self):
        print('loading faces')
        #self.db.createTables()
        if os.path.isfile(self.config):
            print('[ OK ] Loading config...')
            config = []
            with open (self.config, 'rb') as file:
                file = file.read()
                config = json.loads(file)
            
            for name in config:
                print('[ OK ] Loading faces for {}'.format(name))
                person = config[name]
                label = person["label"]
                if os.path.exists(person["faces_dir"]):
                    for _dir, _subdir, files in os.walk(person["faces_dir"]):
                        try:
                            data = []
                            for i in range(len(files)):
                                file = files[i]
                                data.append((name, label, os.path.join(person["faces_dir"], file)))
                            
                            pool = Pool(self.workers)
                            results = pool.map(loadFace, data)
                            pool.close()

                            print(len(results))

                            for encoding, data in results:
                                print(data)
                                if type(encoding) == np.ndarray:
                                    print('storing encoding')
                                    self.db.createFaceEncoding(encoding, label)
                                    print('stored encoding')
                        except Exception as e: print(e)
        else: print("[ERROR] Config file does not exist!")

def loadFace(file):
    try:
        name, label, file = file
        image = face_recognition.load_image_file(file)
        faces = face_recognition.face_locations(image, model="hog")
        encoding = face_recognition.face_encodings(image, faces, num_jitters=0, model="large")[0]

        return encoding, {
            "name": name,
            "label": label
        }
    except: return None, False

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--image', default=None, type=str, help='Image file to test agains')
    args.add_argument('--config', default=None, type=str, help='Load config file to create encodings in database')
    args.add_argument('--tolerance', default=0.35, type=float, help='Tolernace to detect faces at')
    args.add_argument('--threads', default=4, type=int, help='Number of workers to use to load faces into database')
    args = args.parse_args()

    Test(**vars(args))