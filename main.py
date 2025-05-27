import tensorflow as tf
import sys
import os
from woundclassifier import WoundClassifier
from sklearn.metrics import classification_report

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
        
        y_true, y_pred = [], []
        for class_folder in os.listdir(img_path):
            class_folder_path = os.path.join(img_path, class_folder)
            for img in os.listdir(class_folder_path):
                full_img_path = os.path.join(class_folder_path, img)
                if os.path.isfile(full_img_path):
                    print(f"Processing image: {full_img_path}")
                    pred_label, _ = classifier.predict(full_img_path)
                    res = classifier.take_action(full_img_path)
                    print(f"Result for {full_img_path}: {res}")
                    y_true.append(class_folder)
                    y_pred.append(pred_label)
        print(classification_report(y_true, y_pred, target_names=classifier.class_labels))
    
if __name__ == "__main__":
    main()