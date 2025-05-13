import tensorflow as tf
import sys
import os
from woundclassifier import WoundClassifier
def main():
    if len(sys.argv) != 4:
        print("Usage: python main.py <method> [<data_path>] [<model_path>] [<img_path>]")
        return
    
    elif sys.argv[1] == "train":
        data_path = sys.argv[2]
        save_path = sys.argv[3]
        classifier = WoundClassifier(model_path=None)
        classifier.train(dir=data_path, save_dir=save_path)
        
    elif sys.argv[1] == "process":
        model_path = sys.argv[2]
        img_path = sys.argv[3]
        classifier = WoundClassifier(model_path=model_path)

        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist.")
            return
        
        if not os.path.exists(img_path):
            print(f"Image path {img_path} does not exist.")
            return
        
        for img in os.listdir(img_path):
            full_img_path = os.path.join(img_path, img)
            if os.path.isfile(full_img_path):
                print(f"Processing image: {full_img_path}")
                res = classifier.take_action(full_img_path)
                print(f"Result for {full_img_path}: {res}")
    
if __name__ == "__main__":
    main()