YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset. The frame code is from [github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5). 


<!-- 
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start</div>

<details close>
<summary>Installation</summary>

[**Python>=3.6.0**](https://www.python.org/) and [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/) are required:

```
git clone https://github.com/Techyhans/yolov5-portholes.git
cd yolov5
```

Create a new environment(recommended). For example, a conda environment: 
```
conda create -n yolov5 python=3.8.5
conda activate yolov5
``` 

Install dependencies:
```
pip install -r requirements.txt
```

</details>

<details close>
<summary>Training</summary>

Check out [ultralytics Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) to prepare your data and labels. 

You can arrange the data follows the sample located at `data/sample`. Remember to change the config according to your dataset in the `.ymal` file.

Start your training by specifying the following arguments: 
- `--img`: image size 
- `--batch`: batch size 
- `--epochs`: epochs number
- `--data`: custom data config `.yaml` files 
- `--weights yolov5s.pt`: pretrained weights  (recommended)
    - or `--weights '' --cfg yolov5s.yaml`: randomly initialized 

Example:
```
python train.py --img 640 --batch 16 --epochs 300 --data sample/data.yaml --weights yolov5s.pt
```

</details>

<details close>
<summary>Export</summary>
There is extra export requirements.

If you want to export to `.onnx`:
```
pip install -U coremltools onnx scikit-learn==0.19.2 
```
If you want to export to `.tflite`:
```
pip install -U coremltools tensorflow scikit-learn==0.19.2 
```
Export `.pt` to other format by specifying the following arguments:
- `--weights`: model weight to convert 
- `--include`: export format (torchscript onnx coreml saved_model pb tflite tfjs)
- `--data`: your data config

Example:
```
python export.py --weights best.pt --include tflite --tf-nms --agnostic-nms --data data.yaml
```
Read more at [ultralytics export guide](https://github.com/ultralytics/yolov5/issues/251) and their [export.py](https://github.com/ultralytics/yolov5/blob/master/export.py)

Note that there are some limitations with Object Detector TFLite file we export with `--tf-nms` and `--agnostic-nms`, read this [discussion](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2Fultralytics%2Fyolov5%2Fdiscussions%2F2095).
</details>

<details close>
<summary>TFLite Metadata Writer</summary>

There are two versions of Metadata writer:
- V1 attaches the model default name and description
- V2 allows you to specify your model name and description

### Version 1:
`metadata_writer_v1.py` attaches the default name and description as follows:
```
"name": "ObjectDetector",
"description": "Identify which of a known set of objects might be present and provide information about their positions within the given image or a video stream."
```

Start generating the metadata by specifying the following arguments:
- `--model_file`: TFLite model 
- `--label_file`: txt file that list the labels

Example:
```
python metadata_writer_v1.py --model_file best-fp16.tflite --label_file labels.txt 
```

### Version 2:
Update the following in the python file according to your model details.
```
# Your model details here
model_path = 'best-fp16.tflite'
label_path = 'labels.txt'
model_meta.name = "Model name"
model_meta.description = (
    "decription line ..."
    "decription line ..."
    )
```
Then, generate the TFLite with metadata:
```
python metadata_writer_v2.py
```

 Read more at Tensorflow [TensorFlow metadata writer tutorial](https://tensorflow.google.cn/lite/convert/metadata_writer_tutorial#object_detectors).

</details>



