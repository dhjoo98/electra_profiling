#no_grad enabled.
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
from torch import nn
#from torch.profiler import profile, record_function, ProfilerActivity #a new API from pytorch v1.8, used to be torch.autograd.profiler
import argparse
from torch.utils.data import DataLoader, Dataset
import json

#필요한 길이만큼 잘라서, iterator에 넣기.

with torch.no_grad():
    parser = argparse.ArgumentParser(description="choose with model size to use: small/base/large")
    parser.add_argument('--model_size', "-m", default='small', type=str)
    parser.add_argument('--input_length',"-l", default=128, type=int)
    parser.add_argument('--batch_size', '-b', default=8, type=int)
    args = parser.parse_args()
    model_size = args.model_size
    input_length = args.input_length
    batch_size = args.batch_size
    url = "google/electra-"+model_size+"-discriminator"
    #tb_output = './log/electra_'+model_size

    class squadcontextDataset(Dataset):
        def __init__(self):
            with open("/home/dhjoo/work/jinshim_profile/hf_squad_batch/train_context_extract.json", "r") as st_json:
                self.input_list = json.load(st_json)
                self.output_list = []
                for i in self.input_list:
                    self.output_list.append(" ".join(i[0:input_length]))
            self.len = len(self.output_list)

        def __getitem__(self,index):
            return self.output_list[index]

        def __len__(self):
            return self.len

    class electra(nn.Module):
        def __init__(self):
            super(electra, self).__init__()
            self.tokenizer = ElectraTokenizerFast.from_pretrained(url) #identical to Bert tokenizer fake_sentence
            self.discriminator = ElectraForPreTraining.from_pretrained(url).eval()
            #ElectraForPreTraining: parameters in 'config' ; inherits from PreTrainedModel ; also a nn.Module subclass.
            #from_pretrained: a string, model id of a pretrained model hosted inside a model repo on huggingface.co

        def forward(self, fake_sentence):
            #fake_tokens = self.tokenizer.tokenize(fake_sentence) #break apart each words.
            fake_inputs = self.tokenizer.encode(fake_sentence, return_tensors="pt") #start, end token added. #padding이 따로 안보이는데? #return as pytorch tensor
            discriminator_outputs = self.discriminator(fake_inputs)
            predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
            return predictions

    if __name__ == '__main__':
        print("num of threads used: ", torch.get_num_threads())
        dataset = squadcontextDataset()
        train_loader = DataLoader(dataset=dataset,batch_size=batch_size,num_workers=0)
        test_model=electra().eval()
        output = []
        for _, data in enumerate(train_loader):
            for i in range(batch_size):
                #print(data[0])
                output.append(test_model.forward(data[0]))
        print("done!:with batch length: ", len(output))
