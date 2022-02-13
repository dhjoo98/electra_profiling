#그리 정확한 구현은 아닌 것 같다.
from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
from torch import nn
#from torch.utils.tensorboard import SummaryWriter
#from pytictoc import TicToc

#for non-tensorboard profiler
import numpy as np
#import torch.autograd.profiler as profiler

#writer = SummaryWriter() # Writer will output to ./runs/ directory by default

class electra(nn.Module):
    def __init__(self):
        super(electra, self).__init__()
        self.tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator") #identical to Bert tokenizer fake_sentence
        self.discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
        '''
        ElectraForPreTraining: parameters in 'config' ; inherits from PreTrainedModel ; also a nn.Module subclass.
        from_pretrained: a string, model id of a pretrained model hosted inside a model repo on huggingface.co
        '''

    def forward(self, fake_sentence):
        #with profiler.record_function('forward pass'):
        fake_tokens = tokenizer.tokenize(fake_sentence) #break apart each words.
        fake_inputs = tokenizer.encode(fake_tokens, return_tensors="pt") #start, end token added. #padding이 따로 안보이는데? #return as pytorch tensor
        discriminator_outputs = self.discriminator(fake_inputs)
        predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
        return predictions

test_model = electra()
#with profiler.profile(with_stack=True, profile_memory=True) as prof:
predictions = test_model.forward(fake_inputs)
[print("%7s" % token, end="") for token in fake_tokens]
[print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]

#print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'),row_limit=5)

#writer.add_graph(test_model, fake_inputs)
#writer.close()
#t.toc()
