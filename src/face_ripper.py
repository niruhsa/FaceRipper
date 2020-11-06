import argparse, sys, os, ffmpeg, threading, queue, asyncio, json, face_recognition, cv2, time, numpy as np, string, random
from db import Database
from multiprocessing import Process, Manager, Pool
from worker import FaceRipperWorker

class FaceRipper:

	def __init__(self, **kwargs):
		self.kwargs = kwargs

		self.config = self.kwargs['config']
		self.videos = self.kwargs['video_dir']
		self.target_dir = self.kwargs['target_dir']
		self.tolerance = self.kwargs['tolerance']
		self.workers = self.kwargs['threads'] if self.kwargs['threads'] else 4
		self.model = self.kwargs['model'] if self.kwargs['model'] else "hog"
		self.jitters = self.kwargs['jitters']
		self.blur = self.kwargs['blur']
		self.save_blurry = self.kwargs['save_blurry'] if self.kwargs['save_blurry'] != None else True
		self.database = self.kwargs['database'] if self.kwargs['database'] != None else True
		self.createDatabase = self.kwargs['create_database'] if self.kwargs['database'] != None else True
		self.useDatabase = self.kwargs['use_database'] if self.kwargs['use_database'] != None else True
		self.faces = []
		self.encodings = []
		self.queue = []
		self.threads = []

		if self.model == "cnn": print('[WARNING] CUDA support is unstable!')
		else: os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)

		if not self.database and not self.useDatabase:
			self.createDatabase = True

		if self.workers <= 0: self.workers = 1

		self.manager = Manager()
		self.shared_list = self.manager.list()

		self.loop = asyncio.get_event_loop()

		#if self.createDatabase: self.loadFaces()
		try:
			self.loop.run_until_complete(self.startWorkerThreads())
			time.sleep(3)
			self.loop.run_until_complete(self.extractVideoFrames())
		except Exception as e: 	
			print(e)
			print('[ OK ] Attempting to stop threads!')
			for thread in self.threads:
				thread.terminate()
				thread.join()

	def loadFaces(self):
		if self.database:
			db = Database()
			db.createTables()

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
				os.makedirs(os.path.join(self.target_dir, label, ".blurry"), exist_ok=True)
				if os.path.exists(person["faces_dir"]):
					for _dir, _subdir, files in os.walk(person["faces_dir"]):
						data = []
						for i in range(len(files)):
							file = files[i]
							data.append((name, label, os.path.join(person["faces_dir"], file)))

						pool = Pool(self.workers)
						results = pool.map(loadFace, data)
						pool.close()

						for encoding, data in results:
							if data:
								self.encodings.append(encoding)
								self.faces.append(data)
								if self.database: db.storeFaceEncoding(encoding, name, label, person["faces_dir"])
		else: print("[ERROR] Config file does not exist!")

	async def startWorkerThreads(self):
		for i in range(self.workers):
			thread = FaceRipperWorker(
				queue = self.shared_list,
				id = (i + 1),
				tolerance = self.tolerance,
				encodings = self.encodings,
				faces = self.faces,
				target_dir = self.target_dir,
				model = self.model,
				blur = self.blur,
				save_blurry = self.save_blurry,
			)
			
			thread.start()
			self.threads.append(thread)

	async def extractVideoFrames(self):
		for _dir, _subdir, files in os.walk(self.videos):
			for file in files:
				index = files.index(file) + 1
				start = time.time()
				file = os.path.normpath(os.path.join(self.videos, file))
				frames = []
				frame_count = 0

				print('[ OK ] Extracting faces from video: {}'.format(file))
				vc = cv2.VideoCapture(file)
				while vc.isOpened():
					if len(self.shared_list) < self.workers * 4:
						ret, frame = vc.read()
						if not ret: break

						frame = frame[:, :, ::-1]
						frame_count += 1

						self.shared_list.append(frame)
						if frame_count % 100 == 0:
							per_second = frame_count / (time.time() - start)
							print('[ VIDEO ] Video {}/{} || Completed {} frames. {}/frames a second'.format(index, len(files), frame_count, per_second))
			
			print('[ OK ] Finished...')
			for thread in self.threads:
				thread.terminate()

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
	parser = argparse.ArgumentParser(description='Extract faces from videos based on predefined settings', add_help=True)
	parser.add_argument('--config', type=str, default=None, help='Config file to parse in')
	parser.add_argument('--video-dir', type=str, default=None, help='Directory containing all the videos to extract faces from')
	parser.add_argument('--target-dir', type=str, default=None, help='Output folder to store the faces in')
	parser.add_argument('--tolerance', type=float, default=0.325, help='Tolerance for facial recognition, ranging from 0.0 - 1.0 where the lower the number the stricter the matching. The more reference images you have, I recommend lowering it. Default is 0.325 (32.5%)')
	parser.add_argument('--threads', type=int, default=4, help='Number of threads to use. Default is 4')
	parser.add_argument('--jitters', type=int, default=0, help='Number of times to resample face detection for reference images. Higher is more accuracy but slower. e.g, 100 is a 100x more slower.')
	parser.add_argument('--model', type=str, default='hog', help='(UNSTABLE) Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available). Default is "hog". This can only be used when using 1 thread')
	parser.add_argument('--blur', type=float, default=250, help='Blur detection threshold, if an face falls under this threshold it is considered blurry and discarded. Default is 250.')
	parser.add_argument('--save-blurry', type=bool, nargs='?', default=False, help='Save the blurry images in a "blurry" folder under each label. Default is false.')
	parser.add_argument('--database', type=bool, nargs='?', default=False, help="Store face encodings in PostgreSQL database")
	parser.add_argument('--create-database', type=bool, nargs='?', default=False, help="Generate face encodings, if not using PostgreSQL database, this will be auto enabled each script run")
	parser.add_argument('--use-database', type=bool, nargs='?', default=False, help="Don't generate face encodings of target images, instead use the ones stored in the PostgreSQL database")
	args = parser.parse_args()

	clean = FaceRipper(**vars(args))