
import json

if __name__ == "__main__":
    print('here')

    dic = {"teacher": 5, "student": 99}


    with open("./mathvision_test.json") as f:
        loaded_list = json.load(f)

    print(loaded_list[0])

    ids = [int(item["id"]) for item in loaded_list]

    for i in range(3041):
        if not(i in ids):
            loaded_list.append({"id": str(i)})

    loaded_list = sorted(loaded_list, key=lambda x: int(x["id"]))


    with open("./mathvision_v2.json", "w") as f:
        json.dump(loaded_list, f)
        
