import json


def retrieve_meta_data(path):
    MetaJson = json.load(open(path, "r"))
    classes_carprts = []
    class_titles = []
    associated_colors_carprts = []
    for cls in MetaJson["classes"]:
        classes_carprts.append(cls["id"])
        class_titles.append(cls["title"])
        associated_colors_carprts.append(cls['color'])
    return classes_carprts, class_titles, associated_colors_carprts
