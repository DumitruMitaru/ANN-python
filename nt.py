import numpy as np
from mnist import MNIST # package for opening and reading mnist data

def convertLabels(labels):
	converted_labels = np.zeros((labels.shape[0], 10), dtype=np.int32)
	converted_labels[np.arange(labels.shape[0]), labels] = 1

	return converted_labels

def getData():
	mndata = MNIST('data')
	training_images, training_labels = mndata.load_training()
	test_images, test_labels = mndata.load_testing()

	training_images = np.array(training_images, dtype=np.float64) / 255.0								# convert to float and scale from 0-1`
	training_images = np.append(training_images, np.ones((training_images.shape[0],1), dtype=np.float64), 1)# append 1 for the bias
	training_labels = convertLabels(np.array(training_labels))            									# convert training labels from integers to one hot encoding
	test_images = np.array(test_images, dtype=np.float64) / 255.0
	test_images = np.append(test_images, np.ones((test_images.shape[0],1)), 1)            # append 1 for the bias
	test_labels = np.array(test_labels, dtype=np.int32)

	return (training_images, training_labels, test_images, test_labels)

class Network:
	def __init__(self):
		self.weights = np.random.uniform(low=-0.5, high=0.5, size=(10, 28*28 + 1))       # init w & b 
		self.train_images, self.train_labels, self.test_images, self.test_labels = getData() 
	
	def train(self, epochs=70, batch_size=60, step=0.01):
		image_batches = [self.train_images[k:k+batch_size] for k in range(0,self.train_images.shape[0], batch_size)]
		label_batches = [self.train_labels[k:k+batch_size] for k in range(0,self.train_labels.shape[0], batch_size)]
		prev = 0
		for e in range(epochs):
			tr_acc = self.trainAccuracy()	
			print  e, tr_acc, self.testAccuracy()
			if tr_acc - prev < 0.01:
				break
			prev = tr_acc
			for b in range(len(image_batches)):
				self.sgd(step, image_batches[b], label_batches[b])

		print self.confusionMatrix()
		

	# perform sgd with batch of images and labels
	def sgd(self, step, images, labels):
		output = self.feedThrough(images)
		difference = labels - output
		total_wi = [np.tensordot(difference[k], images[k], axes=0) for k in range(difference.shape[0])]
		delta_w = (step / len(total_wi)) * np.sum(total_wi, axis=0)
		self.weights = self.weights + delta_w




	def trainAccuracy(self):
		output = self.feedThrough(self.train_images)
		return 100.0 * np.count_nonzero(np.bitwise_and(output, self.train_labels)) / float(self.train_labels.shape[0])
	
	def testAccuracy(self):
		output = self.feedThrough(self.test_images).argmax(1)
		return 100.0 * np.sum(output == self.test_labels) / float(output.shape[0])

	
	def feedThrough(self, images):
		output = self.weights.dot(images.transpose())
		output = output.transpose()
		# set max element of each output to 1 and other to 0
		thresholded = np.zeros_like(output, dtype=np.int32)
		thresholded[np.arange(output.shape[0]), output.argmax(1)] = 1
		return thresholded

	def confusionMatrix(self):
		output = self.feedThrough(self.test_images).argmax(1)
		C_matrix = np.zeros((10,10), dtype=np.int32)
		np.add.at(C_matrix, (self.test_labels, output), 1)
		return C_matrix


