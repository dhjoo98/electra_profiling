import json


with open("dev.json", "r") as st_json:

    st_python = json.load(st_json)
    #st_python.slice[0](1,3)
    print(st_python["data"][0]["title"])
    #print(st_python.keys())
    new_dict = {'data':[st_python["data"][0]]}

    with open("new_dev.json", "w") as json_file:

        json.dump(new_dict, json_file)
