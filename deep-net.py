import tensorflow as tf
from tensorflow.examples.tutorials.minist import input_data

minist = input_data.read_data_sets("/tmp/data", one_hot=True)

n_nodes_1 = 500
n_nodes_2 = 500
n_nodes_3 = 500

n_classes = 10
batch_size = 100

# height * width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float')

def neural_network_model(data):

	hidden_1_layer = {
		'weights': tf.Variable(tf.random_normal([784, n_nodes_1])),
		'bias': tf.Variable(tf.random_normal([n_nodes_1])),
	}

	hidden_2_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_1, n_nodes_2])),
		'bias': tf.Variable(tf.random_normal([n_nodes_2])),
	}

	hidden_3_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_2, n_nodes_3])),
		'bias': tf.Variable(tf.random_normal([n_nodes_3])),
	}

	output_layer = {
		'weights': tf.Variable(tf.random_normal([n_nodes_3, n_classes])),
		'bias': tf.Variable(tf.random_normal([n_classes])),
	}

	# (input_data * weight) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['bias'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['bias'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['bias']


def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y))
	optimizer = rf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:

		for epoch in range(hm_epochs):
			epoch_loss = 0

			for _ in range(int(minist.train.num_examples/batch_size)):
				x, y = minist.train.next_batch(batch_size)
				_, c = sess.run([optimzier, cost], feed_dict = {x:x, y:y})
				epoch_loss += c
			print('Epoch', epoch, 'complete out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print("Accuracy", accuracy.eval({x:minist.test.images, y:minist.test.labels}))

train_neural_network(x)

