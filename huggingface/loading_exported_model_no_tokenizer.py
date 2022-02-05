#from transformers import ElectraTokenizerFast
import torch
from torch import nn
#from torch.utils.tensorboard import SummaryWriter
#from pytictoc import TicToc

#for non-tensorboard profiler
#import numpy as np
#import torch.autograd.profiler as profiler

#writer = SummaryWriter('pytorch-profile-infer') #add profiling to PyTorch model, based on torch.utils.tensorboard

torch.set_num_threads(1)
torch.set_num_interop_threads(1) #inter-op parallelism
print("(Intra)Thread number: ", torch.get_num_threads(), '\n') #default 2
print("(Inter)Thread number: ", torch.get_num_interop_threads(), '\n') #default 2
#tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator") #identical to Bert tokenizer fake_sentence #시작을 못하는거네.
#gotta export tokenizer as well
loaded_model = torch.jit.load("traced_elecetra.pt")
#print(type(loaded_model)) #<class 'torch.jit.RecursiveScriptModule'>

fake_sentence = "The quick brown fox fake hi the lazy dog"

#fake_tokens = tokenizer.tokenize(fake_sentence)
#fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt") #start, end token added. #padding이 따로 안보이는데?
fake_inputs = torch.Tensor([[101,  1996,  4248,  2829,  4419,  8275,  7632,  1996, 13971,  3899, 102]]).int()
#save input tensor - there is a 'save to buffer implementation too '
#torch.save(fake_inputs,'tensor.pt')
print("tensor exported!\n")
print('___________________\n')
print(fake_inputs, type(fake_inputs))
print('\n___________________\n')
predictions = loaded_model.forward(fake_inputs)

#[print("%7s" % token, end="") for token in fake_tokens]
print("input sentence cannot be printed as it has been tokenized since input\n")
[print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]
print("\ncompleted forward pass")
#good this works!

#writer.add_graph(test_model, fake_inputs)
#writer.close()

#must create a NLP data loader, for inter-op parallelism
