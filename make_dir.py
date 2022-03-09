import os
import yaml

config = yaml.safe_load(open("config.yaml"))

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow', 'scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow', 'models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace', 'annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace', 'images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace', 'pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace', 'models', config['model']['CUSTOM_MODEL_NAME']),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace', 'models',  config['model']['CUSTOM_MODEL_NAME'], 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace', 'models', config['model']['CUSTOM_MODEL_NAME'], 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace', 'models', config['model']['CUSTOM_MODEL_NAME'], 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow', 'protoc')
 }


def main():
    for path in paths.values():
        if not os.path.exists(path):
            os.mkdir(path)


if __name__ == "__main__":
    main()
