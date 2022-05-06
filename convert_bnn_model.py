#https://docs.larq.dev/compute-engine/end_to_end/

import larq as lq
import larq_compute_engine as lce
import larq_zoo as lqz

# load model from larq zoo
model = lqz.literature.BinaryAlexNet(input_shape=None,input_tensor=None,weights="imagenet",include_top=True,num_classes=1000)
lq.models.summary(model)
model.save("BinaryAlexNet.h5")

# Convert our Keras model to a TFLite flatbuffer file
with open("BinaryAlexNet.tflite", "wb") as flatbuffer_file:
    flatbuffer_bytes = lce.convert_keras_model(model)
    flatbuffer_file.write(flatbuffer_bytes)