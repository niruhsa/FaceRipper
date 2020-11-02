import argparse, os, json, face_recognition, cv2, sys

class ReferenceDetections:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.config = self.kwargs['config']
        self.jitters = self.kwargs['jitters']
        self.model = self.kwargs['model']

        if not self.config: print('[ERROR] A config file must be supplied')
        else:
            try: self.loadFaces()
            except KeyboardInterrupt: sys.exit(0)

    def loadFaces(self):
        if os.path.isfile(self.config):
            print('[ OK ] Loading config...')
            config = []
            with open (self.config, 'rb') as file:
                file = file.read()
                config = json.loads(file)

            for name in config:
                print('[ OK ] Loading faces for {}'.format(name))
                person = config[name]
                if os.path.exists(person["faces_dir"]):
                    for _dir, _subdir, files in os.walk(person["faces_dir"]):
                        for file in files:
                            try:
                                file = os.path.join(person["faces_dir"], file)
                                cv2_image = cv2.imread(file)
                                image = face_recognition.load_image_file(file)
                                faces = face_recognition.face_locations(image, model=self.model)
                                encoding = face_recognition.face_encodings(image, num_jitters=self.jitters, model="large")
                                for i in range(len(encoding)):
                                    face = faces[i]
                                    top, right, bottom, left = face
                                    imshow = cv2_image[top:bottom, left:right]
                                    imshow = cv2.resize(imshow, (224, 224))
                                    cv2.imshow(name, imshow)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()
                            except Exception as e: print('[ERROR] Failed loading file: {}, error: {}'.format(file, str(e)))
                else: return print('[ERROR] Faces directory does not exist for person: {}'.format(name))
            for name in config:
                print('[ OK ] Loading faces for {}'.format(name))
                person = config[name]
                label = person["label"]
                os.makedirs(os.path.join(self.target_dir, label), exist_ok=True)
                if os.path.exists(person["faces_dir"]):
                    for _dir, _subdir, files in os.walk(person["faces_dir"]):
                        for file in files:
                            try:
                                file = os.path.join(person["faces_dir"], file)
                                image = face_recognition.load_image_file(file)
                                faces = face_recognition.face_locations(image, model=self.model)
                                encoding = face_recognition.face_encodings(image, faces, num_jitters=self.jitters, model="large")[0]

                                self.encodings.append(encoding)
                                self.faces.append({
                                    "name": name,
                                    "label": label,
                                    "group": person["group"]
                                })
                            except Exception as e: print('[ERROR] Failed loading file: {}, error: {}'.format(file, str(e)))
        else: print("[ERROR] Config file does not exist!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract faces from reference images for visualization of what faces FaceRipper is using from the reference images')
    parser.add_argument('--config', type=str, default=None, help='Config file to parse in')
    parser.add_argument('--jitters', type=int, default=0, help='Number of times to resample face detection for reference images')
    parser.add_argument('--model', type=str, default='hog', help='(UNSTABLE) Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available). Default is "hog". This can only be used when using 1 thread')
    args = parser.parse_args()

    ReferenceDetections(**vars(args))