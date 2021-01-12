import os
import json


def save_predictions(input_data, prediction_dict, config, name):
    folder_name = "-".join([name, config["task"], config["model"]["name"],
                            "-".join(str(v) for v in config["model"]["hyperparams"]),
                            str(config["features"]["selection"]), str(config["features"]["pca"])])
    save_dir = os.path.join(config["outputDir"], folder_name)
    filenames = [filename for filename, _, _ in input_data]
    texts = [text for _, text, _ in input_data]
    predictions = zip(filenames, texts, prediction_dict)
    for filename, text, prediction in predictions:
        new_filename = os.path.splitext(filename)[0] + '.truth'
        json_dict = {"text": text, "result": prediction}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, new_filename), "w", encoding="utf8") as f:
            json.dump(json_dict, f, ensure_ascii=False)


def save_vectors(input_data, vectors, config, folder_name):
    save_dir = os.path.join(config["vectorsDir"], folder_name)
    filenames = [filename for filename, _, _ in input_data]
    texts = [text for _, text, _ in input_data]
    predictions = zip(filenames, texts, vectors)
    for filename, text, prediction in predictions:
        new_filename = os.path.splitext(filename)[0] + '.truth'
        json_dict = {"text": text, "vector": prediction}
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, new_filename), "w", encoding="utf8") as f:
            json.dump(json_dict, f, ensure_ascii=False)