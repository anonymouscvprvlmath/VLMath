import json

if __name__ == "__main__":

    with open("./mathvision_processed.json") as f:
        loaded_list = json.load(f)

    print(loaded_list[0])

    ids = [int(item["id"]) for item in loaded_list]

    for i in range(4000):
        if not (i in ids):
            loaded_list.append({"id": i})

    loaded_list = sorted(loaded_list, key=lambda x: int(x["id"]))

    with open("./mathvision_processed_v2.json", "w") as f:
        json.dump(loaded_list, f)
