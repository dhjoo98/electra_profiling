from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pytictoc import TicToc

#for non-tensorboard profiler
import numpy as np
import torch.autograd.profiler as profiler

#t = TicToc()
#t.tic()

#writer = SummaryWriter() # Writer will output to ./runs/ directory by default

tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator") #identical to Bert tokenizer fake_sentence


class electra(nn.Module): #굳이 이미 모델인 녀셕을 한 번 더?
    def __init__(self):
        super(electra, self).__init__()
        self.discriminator = ElectraForPreTraining.from_pretrained("google/electra-small-discriminator")
        '''
        ElectraForPreTraining: parameters in 'config' ; inherits from PreTrainedModel ; also a nn.Module subclass.
        from_pretrained: a string, model id of a pretrained model hosted inside a model repo on huggingface.co
        '''

    def forward(self, fake_inputs):
        with profiler.record_function('forward pass'):
            discriminator_outputs = self.discriminator(fake_inputs)
            predictions = discriminator_outputs
            #predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
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
#with profiler.profile(with_stack=True, profile_memory=True) as prof:
predictions = test_model.forward(fake_inputs)
print(predictions) # 잘 실행되고 있는지 확인: 다행히 잘 되고 있다.
'''
ElectraForPreTrainingOutput(loss=None, logits=tensor([[-9.5733, -3.4939, -2.1147, -2.1195, -2.3746, -1.5209, -0.8096, -1.8821,
         -2.8919, -2.8594, -9.6136]], grad_fn=<SqueezeBackward1>), hidden_states=None, attentions=None)
'''
#[print("%7s" % token, end="") for token in fake_tokens]
#[print("%7s" % int(prediction), end="") for prediction in predictions.squeeze().tolist()]

#print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total'),row_limit=5)

#writer.add_graph(test_model, fake_inputs)
#writer.close()
#t.toc()
