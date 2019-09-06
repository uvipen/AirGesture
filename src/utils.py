"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_chrome_dino.utils.wrappers import make_dino
import tensorflow as tf
from time import sleep
from src.config import HAND_GESTURES


def is_in_triangle(point, triangle):
    # barycentric coordinate system
    x, y = point
    (xa, ya), (xb, yb), (xc, yc) = triangle
    a = ((yb - yc)*(x - xc) + (xc - xb)*(y - yc)) / ((yb - yc)*(xa - xc) + (xc - xb)*(ya - yc))
    b = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / ((yb - yc) * (xa - xc) + (xc - xb) * (ya - yc))
    c = 1 - a - b
    if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
        return True
    else:
        return False


def load_graph(path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(image, graph, sess):
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    count = 0
    results = {}
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results[count] = [x_min, x_max, y_min, y_max, category]
            count += 1
    return results


def mario(v, lock):
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    done = True
    while True:
        if done:
            env.reset()
            with lock:
                v.value = 0
        with lock:
            u = v.value
        _, _, done, _ = env.step(u)
        env.render()
        sleep(0.01)

def dinosaur(v,lock):
    env = gym.make('ChromeDino-v0')
    env = make_dino(env, timer=True, frame_stack=True)
    done = True
    while True:
        if done:
            env.reset()
            with lock:
                v.value = 0
        with lock:
            u = v.value
        _, _, done, _ = env.step(u)
