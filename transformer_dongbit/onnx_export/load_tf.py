import onnx
import warnings
from onnx_tf.backend import prepare
import numpy as np



warnings.filterwarnings('ignore') # Ignore all the warning messages in this tutorial
model = onnx.load('trm.onnx') # Load the ONNX file
tf_rep = prepare(model) # Import the ONNX model to Tensorflow


#load pytorch tensors
src = np.loadtxt('src.txt')
trg = np.loadtxt('trg.txt')

output = tf_rep.run((src,trg))
print(output)
