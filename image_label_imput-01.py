import tensorflow as tf

# create data features and labels
listFileNames = ['renault/download00.jpg', 'renault/download01.jpg', 'renault/download02.jpg',
                 'renault/download03.jpg', 'renault/download04.jpg', 'renault/download05.jpg',
                 'renault/download06.jpg', 'renault/download07.jpg', 'renault/download08.jpg',
                 'renault/download09.jpg', 'renault/download10.jpg', 'renault/download11.jpg',
                 'renault/download12.jpg', 'renault/download13.jpg']

inputQueue = tf.train.string_input_producer(listFileNames, shuffle=True, seed=True)

reader = tf.WholeFileReader()
key, content = reader.read(inputQueue)
image = tf.image.decode_jpeg(content, channels=1)
resizedImage = tf.image.resize_images(image, [4, 4])
lab = key

# trainFeatures = tf.train.batch([trainFeature], 3)
# trainLabels = tf.train.batch([key], 3)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(50):
        feature, label = sess.run([resizedImage, lab])
        print("Feature")
        print(feature)
        print("Label")
        print(label)

    coord.request_stop()
    coord.join(threads=threads)
