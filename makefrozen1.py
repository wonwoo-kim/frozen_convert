import tensorflow as tf
from tensorflow.python.tools import freeze_graph

freeze_graph.freeze_graph(input_graph='./regression.pbtxt',
                          input_saver="",
                          input_binary=False,
                          input_checkpoint='./final.ckpt',
                          output_node_names='preds',
                          restore_op_name="",
                          filename_tensor_name="",
                          output_graph='./preds_frozen0.pb',
                          clear_devices=False, initializer_nodes="")
