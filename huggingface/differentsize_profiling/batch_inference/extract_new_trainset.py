import json
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
                    save_list.append(li["context"][0:512])
            json.dump(save_list,json_file)

    #print(st_python.keys())
    #new_dict = {'data':[st_python["data"][0]]}

    #with open("new_dev.json", "w") as json_file:

        #json.dump(new_dict, json_file)
