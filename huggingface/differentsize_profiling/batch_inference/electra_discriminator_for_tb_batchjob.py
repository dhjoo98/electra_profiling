

#그리 정확한 구현은 아닌 것 같다.
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity #a new API from pytorch v1.8, used to be torch.autograd.profiler
import argparse
#from torch.utils.data import DataLoader, Dataset
import json

#필요한 길이만큼 잘라서, iterator에 넣기.

parser = argparse.ArgumentParser(description="choose with model size to use: small/base/large")
parser.add_argument('--model_size', "-m", default='small', type=str)
parser.add_argument('--input_length',"-l", default=128, type=int)
args = parser.parse_args()
model_size = args.model_size
input_length = args.input_length
url = "google/electra-"+model_size+"-discriminator"
tb_output = './log/electra_'+model_size
input_multiplier = int(input_length/8)

class electra(nn.Module):
    def __init__(self):
        super(electra, self).__init__()
        self.tokenizer = ElectraTokenizerFast.from_pretrained(url) #identical to Bert tokenizer fake_sentence
        self.discriminator = ElectraForPreTraining.from_pretrained(url)
        #ElectraForPreTraining: parameters in 'config' ; inherits from PreTrainedModel ; also a nn.Module subclass.
        #from_pretrained: a string, model id of a pretrained model hosted inside a model repo on huggingface.co

    def forward(self, fake_sentence):
        #fake_tokens = self.tokenizer.tokenize(fake_sentence) #break apart each words.
        fake_inputs = self.tokenizer.encode(fake_sentence, return_tensors="pt") #start, end token added. #padding이 따로 안보이는데? #return as pytorch tensor
        discriminator_outputs = self.discriminator(fake_inputs)
        predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
        return predictions

#with profile(activities=[ProfilerActivity.CPU],profile_memory=False,with_stack=True,record_shapes=True) as prof: #for CLI
with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_output),profile_memory=True,with_stack=True,record_shapes=True) as prof:
    with open("train_context_extract.json", "r") as st_json:
        input_list = json.load(st_json)
        input_iter = iter(input_list)
    test_model = electra()
    output = []
    for i in range(len(input_list)):
        output.append(test_model.forward(next(input_iter)[0:input_length]))
    #predictions = test_model.forward(fake_sentence)
    #[print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]
    print("done!")

    #for tensorboard
