import argparse, sys, os, ffmpeg, queue, json, face_recognition, cv2, time, numpy as np, pickle, string, random
from multiprocessing import Manager, Process
from PIL import Image

class FaceRipperWorker(Process):

	def __init__(self, queue = None, id = None, tolerance = None, encodings = None, faces = None, target_dir = None, model = None, blur = None, save_blurry = False):
		super(FaceRipperWorker, self).__init__()
		self.queue = queue
		self.id = id if id != None else 1
		self.tolerance = tolerance
		self.encodings = encodings
		self.target_dir = target_dir
		self.faces = faces
		self.model = model
		self.blur = blur
		self.save_blurry = save_blurry
		self.start_time = time.time()
		self.compute_times = [ self.start_time - self.start_time + 0.01 ]
		self.items_pass = 0
		self.items = 0
		self.completed_faces = 0

	def run(self):
		print('[ WORKER #{} ] Starting worker with {}% tolerance and {}/{} known encodings (model = "{}")...'.format(self.id, self.tolerance * 100, len(self.encodings), len(self.faces), self.model))
		self.startWorking()

	def startWorking(self):
		while True:
			if len(self.queue) > 0:
				try:
					item = self.queue.pop()
					self.work(item)
				except IndexError: pass

	def work(self, item):
		try:
			self.start_time = time.time()
			if self.items_pass % 15 == 0: print('[ WORKER #{} ] Processing item, average time to process per item is {:,.2f}ms. Completed {:,} items so far, and {:,} faces extracted.'.format(self.id, np.average(self.compute_times), self.items_pass, self.completed_faces))
			faces = face_recognition.face_locations(item, model=self.model)
			encodings = face_recognition.face_encodings(item, model="large")
			for i in range(len(encodings)):
				encoding = encodings[i]
				face = faces[i]
				for index in range(len(self.encodings)):
					enc = self.encodings[index]
					results = face_recognition.face_distance([ enc ], encoding)
					if results[0] <= self.tolerance:
						filename = self.generateFileName(os.path.join(self.target_dir, self.faces[index]["label"]), extension = ".jpg")
						output_path = os.path.normpath(os.path.join(self.target_dir, self.faces[index]["label"], filename))
						escaped_path = output_path.replace("\\", "\\\\")
						top, right, bottom, left = face
						image = Image.fromarray(item[top:bottom, left:right])
						width, height = image.size
						blurry = self.isBlurry(item[top:bottom, left:right])
						if width > 100 and height > 100:
							if not blurry:
								image.save(escaped_path)
								self.completed_faces += 1
							else:
								if self.save_blurry:
									image.save(os.path.normpath(os.path.join(self.target_dir, self.faces[index]["label"], ".blurry", filename)))
			self.items += 1
		except RuntimeError:
			self.queue.append(item)
			pass
		except Exception as e: print('[ WORKER #{} ] Error induced while working on item: {}'.format(self.id, str(e)))
		self.calculateComputeTime()
		self.items_pass += 1

	def calculateComputeTime(self):
		if len(self.compute_times) >= 50: self.compute_times = self.compute_times[0:48]
		self.compute_times.insert(0, time.time() - self.start_time)

	def randomFileName(self, length=16, extension = ''):
		chars = string.ascii_lowercase + string.ascii_uppercase + string.digits
		return ''.join(random.choice(chars) for i in range(length)) + extension
 
	def generateFileName(self, target_dir, length = 16, extension = ''):
		id = self.randomFileName(length = length, extension = extension)
		while os.path.exists(os.path.join(target_dir, id)): id = self.randomFileName(extension)

		return id

	def isBlurry(self, image): return cv2.Laplacian(image, cv2.CV_64F).var() < self.blur