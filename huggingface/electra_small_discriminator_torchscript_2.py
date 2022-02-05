from transformers import ElectraForPreTraining, ElectraTokenizer
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
#from pytictoc import TicToc

#for non-tensorboard profiler
import numpy as np
import torch.autograd.profiler as profiler

#t = TicToc() #to see if OMP works
#t.tic()

#writer = SummaryWriter() # Writer will output to ./runs/ directory by default

tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator") #identical to Bert tokenizer fake_sentence #시작을 못하는거네.

class electra(nn.Module): #굳이 이미 모델인 녀셕을 한 번 더?
    def __init__(self):
        super(electra, self).__init__()
        self.discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator", torchscript=True) #to enable torchscript
        '''
        ElectraForPreTraining: parameters in 'config' ; inherits from PreTrainedModel ; also a nn.Module subclass.
        from_pretrained: a string, model id of a pretrained model hosted inside a model repo on huggingface.co
        '''

    def forward(self, fake_inputs):
        discriminator_outputs = self.discriminator(fake_inputs)
        predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
        return predictions

sentence = "The quick brown fox jumps over the lazy dog"
fake_sentence = "The quick brown fox fake hi the lazy dog"
fake_tokens = tokenizer.tokenize(fake_sentence)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt") #start, end token added. #padding이 따로 안보이는데?
'''
print("tokens: " , fake_tokens, "   length: ", len(fake_tokens)) # tokens:  ['the', 'quick', 'brown', 'fox', 'fake', 'hi', 'the', 'lazy', 'dog']    length:  9
#print(tokenizer.encode(fake_sentence))
print("inputs: ", fake_inputs, "   length:", fake_inputs.size()) # inputs:  tensor([[  101,  1996,  4248,  2829,  4419,  8275,  7632,  1996, 13971,  3899, 102]])    length: torch.Size([1, 11])
print("-----------------")
'''
test_model = electra()
predictions = test_model.forward(fake_inputs)
[print("%7s" % token, end="") for token in fake_tokens]
[print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]

#creating the trace
traced_model = torch.jit.trace(test_model,fake_inputs) #"dummy input", it is called
torch.jit.save(traced_model, "traced_elecetra.pt")
print("traced_model exported!")
#writer.add_graph(test_model, fake_inputs)
#writer.close()
#t.toc()

#export pretrained tokenizer from hf repo as local json file.
