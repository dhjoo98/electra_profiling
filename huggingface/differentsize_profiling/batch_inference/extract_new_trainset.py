import json
import re
'''
dataset =

data = DataLoader("./train-v2.0.json", batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
'''
with open("train-v2.0.json", "r") as st_json:
    with open("train_context_extract.json", "w") as json_file:
            st_python = json.load(st_json)
            save_list = []
            for i,li in enumerate((st_python["data"][0]["paragraphs"])):
                    hash = re.sub(r'[^\w\s]', '', li["context"])
                    hash = hash.split()
                    hash = hash * 20
                    save_list.append(hash[0:512])
                    print(len(hash[0:512]))
            print("savelist: ", len(save_list))
            json.dump(save_list[0:64],json_file)
            #print(len(save_list[0:64]))

    #print(st_python.keys())
    #new_dict = {'data':[st_python["data"][0]]}

    #with open("new_dev.json", "w") as json_file:

        #json.dump(new_dict, json_file)
