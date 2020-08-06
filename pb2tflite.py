from tensorflow.python.framework import graph_util
import tensorflow as tf

graph_def_file = "preds_frozen0.pb"
input_arrays = ["start"]
output_arrays = ["preds"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("start_0.tflite", "wb").write(tflite_model)
