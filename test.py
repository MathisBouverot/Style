from tensorflow.python.keras.layers import Dense, Conv2D
import numpy as np

lay = Conv2D(filters = 10, kernel_size = (3, 3))
print(lay.get_weights())
lay.set_weights(lay.get_weights())

print(lay.get_config())

new_lay = Conv2D.from_config(lay.get_config())
print(new_lay.get_weights())
