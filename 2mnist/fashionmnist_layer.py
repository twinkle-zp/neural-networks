import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import  datetime
from    matplotlib import pyplot as plt
import  os
import  io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#预处理
def preprocess(x, y):
    # x: [0~255] => [0~1.]
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x,y


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(images):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title='name')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return figure

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()   #fashion mnist与mnist类似，十种衣帽类的图片，较为复杂一点
print(x.shape, y.shape) #(60000, 28, 28) (60000,)

batchsz = 128

#map()进行预处理，shuffle()进行打乱数据
db = tf.data.Dataset.from_tensor_slices((x,y))   #加载数据
db = db.map(preprocess).shuffle(10000).batch(batchsz)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)

db_iter = iter(db)
sample = next(db_iter)     #取样
print('batch:', sample[0].shape, sample[1].shape)   #batch: (128, 28, 28) (128,)


model = Sequential([
    layers.Dense(256, activation=tf.nn.relu), # [b, 784] => [b, 256]
    layers.Dense(128, activation=tf.nn.relu), # [b, 256] => [b, 128]
    layers.Dense(64, activation=tf.nn.relu), # [b, 128] => [b, 64]
    layers.Dense(32, activation=tf.nn.relu), # [b, 64] => [b, 32]
    layers.Dense(10) # [b, 32] => [b, 10], 330是参数量，330 = 32*10 + 10
])
model.build(input_shape=[None, 28*28])
model.summary()    #调试功能，打印网络结构
# w = w - lr*grad
optimizer = optimizers.Adam(lr=1e-3)

#创建日志
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

# get x from (x,y)
sample_img = next(iter(db))[0]
# get first image instance
sample_img = sample_img[0]
sample_img = tf.reshape(sample_img, [1, 28, 28, 1])
with summary_writer.as_default():
    tf.summary.image("Training sample:", sample_img, step=0)

def main():

    for epoch in range(30):

        for step, (x,y) in enumerate(db):
            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])

            with tf.GradientTape() as tape:
                # [b, 784] => [b, 10]
                logits = model(x)    #前向传播得到[b,10]
                y_onehot = tf.one_hot(y, depth=10)
                # [b]
                loss_mse = tf.reduce_mean(tf.losses.MSE(y_onehot, logits)) #tf.reduce_mean对返回的实例求均值
                loss_ce = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)

            grads = tape.gradient(loss_ce, model.trainable_variables)#model.trainable_variables得到所有参数
            optimizer.apply_gradients(zip(grads, model.trainable_variables))#参数原地更新，zip将对应位置元素拼一起


            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss_ce), float(loss_mse))
                with summary_writer.as_default():
                    tf.summary.scalar('train-loss', float(loss_ce), step=step)

                    # test
        total_correct = 0
        total_num = 0
        for x,y in db_test:

            # x: [b, 28, 28] => [b, 784]
            # y: [b]
            x = tf.reshape(x, [-1, 28*28])
            # [b, 10]
            logits = model(x)
            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1)    #概率最大的即预测的结果
            pred = tf.cast(pred, dtype=tf.int32)
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_correct += int(correct)
            total_num += x.shape[0]

        acc = total_correct / total_num
        print(epoch, 'test acc:', acc)

        # print(x.shape)
        '''val_images = x[:25]
        val_images = tf.reshape(val_images, [-1, 28, 28, 1])
        with summary_writer.as_default():
            tf.summary.scalar('test-acc', float(total_correct / total_num), step=step)
            tf.summary.image("val-onebyone-images:", val_images, max_outputs=25, step=step)

            val_images = tf.reshape(val_images, [-1, 28, 28])
            figure = image_grid(val_images)
            tf.summary.image('val-images:', plot_to_image(figure), step=step)'''

if __name__ == '__main__':
    main()