
import tensorflow as tf

print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))
import time
t=time.time()
tf.constant(1.0) + tf.constant(1.0)  # trigger a device kernel launch
print('first op took', time.time()-t, 's')
