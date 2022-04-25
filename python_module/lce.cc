#include <cstdio>

#include <iostream>
#include <iomanip>

#include "../larq_compute_engine/tflite/kernels/lce_ops_register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "opencv2/opencv.hpp"


// This file is based on the TF lite minimal example where the
// "BuiltinOpResolver" is modified to include the "Larq Compute Engine" custom
// ops. Here we read a binary model from disk and perform inference by using the
// C++ interface. See the BUILD file in this directory to see an example of
// linking "Larq Compute Engine" cutoms ops to your inference binary.

using namespace std;
using namespace tflite;
typedef cv::Point3_<float> Pixel;
const uint WIDTH = 224;
const uint HEIGHT = 224;
const uint CHANNEL = 3;
const uint OUTDIM = 1000;

// to normalize input image to -1 & 1
void normalize(Pixel &pixel){
    pixel.x = ((pixel.x / 255.0)-0.5)*2.0;
    pixel.y = ((pixel.y / 255.0)-0.5)*2.0;
    pixel.z = ((pixel.z / 255.0)-0.5)*2.0;
}


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

/*
 * It will iterate through all the lines in file ClassName and
 * put them in given vector
 */
bool getClassName(string fileName, vector<string> & vecOfStrs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vecOfStrs.push_back(str);
    }
    //Close The File
    in.close();
    return true;
}

/*
 * This function predicts the class of the image given with the model convert to tflite by the LCE converter
 * It returns the class of the image, the confidence and the inference time 
 * As input:
 * tflitePath (= /path/to/yourBinaryModel.tflite) 
 * image (=/path/to/yourImage)
 * n : number of class that you have 
 * ClassPath (=/path/to/FileClasseName.txt)
 * img_size : inference size default value 224
 * channels : number channel of images = default value 3
 */
std::string lce(char* tflitePath, char* image, char* ClassPath, int n, int img_size = 224, int channels=3) {

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(tflitePath);
  TFLITE_MINIMAL_CHECK(model != nullptr);

 
  // create a builtin OpResolver
  tflite::ops::builtin::BuiltinOpResolver resolver;
  // register LCE custom ops
  compute_engine::tflite::RegisterLCECustomOps(&resolver);

  // Build the interpreter
  InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors
  
  // read image file
  cv::Mat img = cv::imread(image,cv::IMREAD_COLOR);

  // convert to float; BGR -> RGB
  cv::Mat inputImg;
  img.convertTo(inputImg, CV_32FC3);
  cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);

  // normalize to -1 & 1
  Pixel* pixel = inputImg.ptr<Pixel>(0,0);
  const Pixel* endPixel = pixel + inputImg.cols * inputImg.rows;
  for (; pixel != endPixel; pixel++)
    normalize(*pixel);

  // resize image as model input
  cv::resize(inputImg, inputImg, cv::Size(WIDTH, HEIGHT));
  int bytes = img.total() * img.elemSize();
 
  // get input & output layer
  float* inputLayer = interpreter->typed_input_tensor<float>(0);
 

  // flatten rgb image to input layer.
  float* inputImg_ptr = inputImg.ptr<float>(0);
  memcpy(inputLayer, inputImg.ptr<float>(0),
          img_size * img_size * channels * sizeof(float));
  
  //for (int i=0; i<(WIDTH * HEIGHT * CHANNEL); i++){
    //cout<<inputLayer[i]<<endl;
  //}
 
  // Run inference
  auto t1 = std::chrono::high_resolution_clock::now();
  interpreter->Invoke();
  auto t2 = std::chrono::high_resolution_clock::now();
  //printf("\n\n=== Post-invoke Interpreter State ===\n");
  //tflite::PrintInterpreterState(interpreter.get());
  
  // Read output buffers
  // TODO(user): Insert getting data out code.
  float* outputLayer = interpreter->typed_output_tensor<float>(0);

  //printf("\n=========== Output predict ==========\n");
  //Get id class with the best score (confidence)
  int idx = 0;
  float v = 0.0f;
  for(int i=1; i<=n; i++){
      float vi = outputLayer[i-1];
      if(vi > v){
        idx = i-1;
        v = vi;
      }
  }
  // Get inference time
  int inf_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
  // vector for class names
  vector<string> classNames;
  // Get the contents of file in a vector
  bool ispredit = getClassName(ClassPath, classNames);
  string result= "";
  if(ispredit)
  {
      // Get class name of idx
      string className = classNames[idx];
      //printf("Id_class':%d,'class':'%s','prop':%f,'inference_time':%d\n", idx, className.c_str(), v,inf_time);
      // return result into json format
      result = string("{\"Image\":\"")+string(image)+ string("\",\"ClassName\":")+className+
        string(",\"Confidence\":")+std::to_string(v)+string(",\"Inference time\":")+std::to_string(inf_time)+string("}");
  }
  return result;
}
