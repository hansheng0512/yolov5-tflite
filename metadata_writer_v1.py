# Object detector TFLite metadata writer
import argparse
from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata

def metadata_writer(model_path="best-fp16.tflite", label_path="labels.txt"):
    ObjectDetectorWriter = object_detector.MetadataWriter
    _MODEL_PATH = model_path
    _LABEL_FILE = label_path
    _SAVE_TO_PATH = "best-fp16-metadata-v1.tflite"

    writer = ObjectDetectorWriter.create_for_inference(
        writer_utils.load_file(_MODEL_PATH), [127.5], [127.5], [_LABEL_FILE])
    writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)

    # Verify the populated metadata and associated files.
    displayer = metadata.MetadataDisplayer.with_model_file(_SAVE_TO_PATH)
    print("Metadata populated:")
    print(displayer.get_metadata_json())
    print("Associated file(s) populated:")
    print(displayer.get_packed_associated_file_list())
    print(f'Success!\nMetadata and the label file have been written into {_SAVE_TO_PATH}.')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="Model file to add metadata.",)
    parser.add_argument("--label_file", type=str, help="Label file that contains the class name.",)
    opt = parser.parse_args()
    return opt


def main(opt):
    metadata_writer(opt.model_file, opt.label_file)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

