import tensorflow as tf
import os.path as opath
import os


class tf_dataset:

    def __init__(self, path, batch_size=10):
        dateset = []
        posi_path = opath.join(path, "positive.txt")
        nega_path = opath.join(path, "negative.txt")
        part_path = opath.join(path, "part.txt")

        if opath.exists(posi_path):
            dateset.extend(open(posi_path).readlines())
        if opath.exists(nega_path):
            dateset.extend(open(nega_path).readlines())
        if opath.exists(part_path):
            dateset.extend(open(part_path).readlines())

        names = []
        conds = []
        offsets = []

        for i in dateset:
            strs = i.split()
            names.append(strs[0])
            conds.append(float(strs[1]))
            offsets.append([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])])

            self.dataset = tf.data.Dataset.from_tensor_slices((names, conds, offsets))
            self.dataset = self.dataset.map(
                lambda f_name, cond, offset: tuple(
                    tf.py_func(self.my_func, [f_name, cond, offset], [tf.string, cond.dtype, offset.dtype]))
            )
            self.dataset = self.dataset.repeat(-1)
            self.dataset = self.dataset.shuffle(100)
            self.dataset = self.dataset.batch(batch_size)

            self.iterator = self.dataset.make_one_shot_iterator()
            self.next_element = self.iterator.get_next()

    def get_batch(self, sess):
        return sess.run(self.next_element)

    def my_func(self, file_name, cond, offset):
        return "dx_" + bytes.decode(file_name), cond, offset


if __name__ == '__main__':
    dataset = tf_dataset(r"D:\AI\celebA\CelebA", 10)
    with tf.Session() as sess:
        for i in range(10):
            print(dataset.get_batch(sess))
        print("-----------------------------------")
