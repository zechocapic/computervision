import tensorflow as tf

# create data features and labels
list_filename = ['cars/download00.jpg', 'cars/download01.jpg', 'cars/download02.jpg',
                 'cars/download03.jpg', 'cars/download04.jpg', 'cars/download05.jpg',
                 'cars/download06.jpg', 'cars/download07.jpg', 'cars/download08.jpg',
                 'cars/download09.jpg', 'cars/download10.jpg', 'cars/download11.jpg',
                 'cars/download12.jpg', 'cars/download13.jpg']

input_queue = tf.train.string_input_producer(list_filename, shuffle=True, seed=True)

reader = tf.WholeFileReader()
key, content = reader.read(input_queue)
image = tf.image.decode_jpeg(content, channels=1)
resized_image = tf.image.resize_images(image, [4, 4])
lab = key

# trainFeatures = tf.train.batch([trainFeature], 3)
# trainLabels = tf.train.batch([key], 3)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(50):
        feature, label = sess.run([resized_image, lab])
        print("Feature")
        print(feature)
        print("Label")
        print(label)

    coord.request_stop()
    coord.join(threads=threads)
