import argparse, sys, os, ffmpeg, queue, json, face_recognition, cv2, time, numpy as np, pickle, string, random
from multiprocessing import Manager, Process
from PIL import Image
from db import Database

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
		db = Database()
		cascade = cv2.CascadeClassifier('res/haarcascade.xml')
		
		while True:
			if len(self.queue) > 0:
				try:
					item = self.queue.pop()
					self.work2(item, db, cascade)
				except IndexError: pass
				except: pass

	def resizeImage(self, item, size=(1280, 720)):
		original_size = (item.shape[1], item.shape[0])
		item = cv2.resize(item, size)
		return item, original_size

	def upscaleFaces(self, faces, item, size=(1280, 720)):
		height, width = item.shape
		
		if height > size[1] and width > size[0]:
			uheight, uwidth = height / size[1], width / size[0]
		else:
			uheight, uwidth = size[1] / height, size[0] / width

		return_faces = []
		for x, y, w, h in faces:
			x = x // uwidth
			y = y // uheight
			w = math.ceil(x * uwdth)
			y = math.ceil(y * uheight)

			if x < 0: x = 0
			if y < 0: y = 0
			if x + w > width: w = width - x
			if y + h > height: h = height - y

			return_faces.append((x, y, w, h))
		return return_faces

	def work2(self, item, db, cascade = None):
		try:
			if item.shape[0] > 2160 or item.shape[1] > 3840:
				return False, "Image is too large, maximum of 3840x2160"
			faces = self.getFaces(item, cascade = cascade, padding=75)
			encodings = []

			for face in faces:
				encoding = face_recognition.face_encodings(face)
				if len(encoding) > 0:
					for enc in encoding: encodings.append((face, enc))

			for face, enc in encodings:
				results = db.compareFaceEncodings(enc, threshold = self.tolerance)
				if len(results) > 0:
					data = results[0]
					name, label = data
					image = Image.fromarray(face)
					#print('[ WORKER #{} ] Found matching face for "{}" in frame'.format(self.id, name))
					directory = os.path.join(self.target_dir, label, ".blurry")
					os.makedirs(directory, exist_ok = True)
					filename = self.generateFileName(os.path.join(self.target_dir, label), extension = ".jpg")
					output_path = os.path.normpath(os.path.join(self.target_dir, label, filename))
					escaped_output = output_path.replace("\\", "\\\\")
					blurry = self.isBlurry(face)
					width, height = image.size
					if width >= 100 and height >= 100:
						if not blurry:
							image.save(escaped_output)
							self.completed_faces += 1
						else:
							if self.save_blurry:
								blurry_path = os.path.normpath(os.path.join(self.target_dir, label, ".blurry", filename))
								image.save(blurry_path.replace("\\", "\\\\"))
								self.completed_faces += 1
			self.items += 1
			self.calculateComputeTime()
			if self.items % 100 == 0:
				print('[ WORKER #{} ] Completed {} items & extracted {} faces. Average time per frame is {}s'.format(self.id, self.items, self.completed_faces, np.average(np.array(self.compute_times) / 1000)))
		except: pass

	def getFaces(self, item, cascade = None, padding = 75):
		#gray = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
		faces = cascade.detectMultiScale(item, 1.1, 5, minSize=(30, 30))
		
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
			ret_faces.append(item[y:y + h, x:x + w])
		return ret_faces

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