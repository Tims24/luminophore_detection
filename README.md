# luminophore detection using tensorflow object detection

* data: https://drive.google.com/drive/folders/1Uibpo8Uxk95mCvoQv3EYiC-kJ9IhUMJT?usp=sharing
***
<br />

### 1. Run make_dir.py to create directories

### 2. Download tensorflow models zoo and TFOD

    Run the verifiction script to check whether TFOD is installed succesfully
* More details: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

### 3. Download pretrained model, which you want
    "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8" - current model, you can change it in "PRETRAINED_MODEL_NAME"

### 4. Create the train and test records for model
### 5. Copy and update the model config

## Steps 2-5 can bo done with TFOD_Zoo_install.ipynb

### 6. Move train and test data to 
    \Tensorflow\workspace\images

### 7. Train the model with this command: 
```
python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --num_train_steps=2000
```
* You can change the number of train steps, default - 50000
### 8. Evaluate the model with this command:
    "python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet"

***

# Detection
* detect.py - image detection
* tensorflow_cv2.py - cv2 parameters calculation with beam detection from tensorflow