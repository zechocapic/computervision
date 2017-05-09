import tensorflow as tf

# create data features and labels
listFileNames = ['renault/download.jpg', 'renault/download1.jpg', 'renault/download2.jpg']
listLabels = ['car0', 'car1', 'car2']

inputQueue = tf.train.slice_input_producer([listFileNames, listLabels], shuffle=False, seed=False)

content = tf.read_file(inputQueue[0])
image = tf.image.decode_jpeg(content, channels=1)
resizedImage = tf.image.resize_images(image, [4, 4])

trainFeature = resizedImage
trainLabel = inputQueue[1]

trainFeatures = tf.train.batch([trainFeature], 2)
trainLabels = tf.train.batch([trainLabel], 2)

# tensorflow features and labels
features = tf.placeholder(tf.float32, [3, 16])
labels = tf.placeholder(tf.float32, [3, 1])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    f, l = sess.run([trainFeatures, trainLabels])
    print(f)
    print(l)
    print(trainFeature)
    print(trainFeatures)

    coord.request_stop()
    coord.join(threads=threads)
