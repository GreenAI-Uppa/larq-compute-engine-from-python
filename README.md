# Binary model inference with Larq Compute Engine from Python
With this repostory, you can now use the [Larq](https://docs.larq.dev/larq/) inference engine, [Larq Compute Engine (LCE)](https://docs.larq.dev/compute-engine/) in a python script, to perform image inference with a Binary model on a 64 bit ARM system.

LCE provides a collection of hand-optimized TensorFlow Lite custom operators for supported instruction sets, developed in C++ using compiler intrinsics. Using the LCE converter, the Binary model built with Larq is converted to tensorflow Lite to take into account these custom operators.
In order to take advantage of these optimizations, a C++ script, based on the TF Lite minimal example, is used to create a tflite interpreter with the custom LCE operations and perform the inference with the converted model.

To perform the inference from a Python script, we have created a Python module of the C++ script with the open source software [SWIG](https://www.swig.org/Doc1.3/Python.html). This module allows to call the C++ function that performs the inference as if it was a Python function.

The different experiments have been done on a Jetson Nano ARM64. But you have the choice to use other devices with an ARM system like the Raspberry. However the conversion of the keras model built from Larq to Tensorflow tflite with the LCE converter can only be done on an x86 system ie you train the binary network and convert it to tflite on a x86 machine, and then you can use our script for fast inference of your model on our ARM device.

See article [Binary Neural Network part 2](https://medium.com/@fkinesow/binary-neural-network-part-2-cecbe5761b78) for more information on the C++ script to perform the inference with the LCE.
A comparison of the performance of Vanilla and Binary AlexNet is also presented in this article. 

## Requirement
### To convert Larq Keras model to tflite
From x86 host machine
* Python version 3.6, 3.7, 3.8, or 3.9
* Tensorflow version 1.14, 1.15, 2.0, 2.1, 2.2, 2.3, 2.4 or 2.5 (recommended):

### To perform inference on an ARM system  
From Jetson Nano
* Python version 3.6, 3.7, 3.8, or 3.9
* C++ compiler 
* swig (install with ```sudo apt install swig``` on linux system)

## Clone repostory and installation
* From x86 host machine to create Keras model or import model from [Larq zoo](https://docs.larq.dev/zoo/)
  * Install Larq
  ```
  $ pip install Larq
  ```
  * Install Larq Compute Engine
  ```
  $ pip install larq-compute-engine
  ```
  * Install [Larq zoo](https://docs.larq.dev/zoo/) for importing Larq model or create a custom Binary model with Larq (refer to [this](https://docs.larq.dev/larq/tutorials/mnist/))
  ```
  $ pip install larq-zoo
  ```
* From ARM system
  * Clone this repostory 
   ```
   $ git clone https://github.com/GreenAI-Uppa/larq-compute-engine-from-python.git
   ```
  * Intall OpenCV for input image processing 
  
    As the Larq compute engine C++ API is used to perform the inference on the Jetson, it will be necessary to install OpenCV C++.
    See [this article](https://medium.com/@pokhrelsuruchi/setting-up-opencv-for-python-and-c-in-ubuntu-20-04-6b0331e37437) for installing OpenCV C++ on linux system or [this article](https://automaticaddison.com/how-to-install-opencv-4-5-on-nvidia-jetson-nano/) for installation on Nvidia Jetson Nano
  * Go into directory python_module and open setup.py file 
    ```
    $ cd larq-compute-engine-from-python/python_module
    ```
  * Add the path to the opencv installation directory in the INCLUDE variable
    
    In `setupp.py`change ```/usr/local/include/opencv4``` to ```path/to/your_opencv/directory```
    
  * Intall the LCE module with the setup.py file
     ```
     $ python3 setup.py install
     ```
  If you're using another system such as a Raspberry, it is better to create another module rather than using the one provided in this repository.  In this case you'll need to run the following commands:
   ```
   $ swig -c++  -python lce.i
   $ python3 setup.py  build_ext --inplace
   ```
  The module is now ready to be used in a python script.
  
 ## USAGE
 ### Convert keras model to tflite model
 Convert the binary model (download from Larq zoo or create with Larq) using the Larq Compute Engine module from your x86 source machine. See `convert_bnn_model.py`.

 Next, move the tflite model to the arm device and proceed with the inference.
 
 ### Run inference from ARM system
 Once the model is converted and imported on the device where the module is installed, you can perform the inference of this model from a python script.
 
 The `run.py` file gives an example to use the module. 
 
 Different parameters are required to run this script:
 * `--tflite` : path of the tflite model. Default value is BinaryAlexNet.tflite : a binary AlexNet pre-trained on ImageNet data. This model was downloaded from Larq zoo and converted to tflite with the LCE converter. (See `convert_bnn_model.py` file)
 * `--source` : path of image or directory of images to predict
 * `--classesNames` : path of the text file that contains the class names (must have the same structure as ìmagenet_label.txt` file)
 * `--imgsz` : input size h,w of your model. Default value 224 for AlexNet
 * `--channels`: number of image channels.  Default value  3

Example of using run.py:
* With an image as source
  ```
  $ python3 run.py --tflite BinaryAlexNet.tflite --source test.jpg \
                      --classesNames  ìmagenet_label.txt --imgsz 224 --channels 3
  ```
* With an image directory as source  
  ```
  $ python3 run.py --tflite BinaryAlexNet.tflite --source path/to/directory \
                      --classesNames  ìmagenet_label.txt --imgsz 224 --channels 3
  ```

The LCE module returns as output the image names, the predicted class names, the confidence and the inference time of each prediction. 

Example: For the prediction of an image, we have the following result
```
      Image                                ClassName  Confidence Inference time
   test.jpg  ballpoint, ballpoint pen, ballpen, Biro    0.013324             88
```
