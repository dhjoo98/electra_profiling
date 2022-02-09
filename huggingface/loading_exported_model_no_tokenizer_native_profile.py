#from transformers import ElectraTokenizerFast
import torch
from torch import nn
#from torch.utils.tensorboard import SummaryWriter
#from pytictoc import TicToc

#to profile
#import torchvision.models as tvmodels
from torch.profiler import profile, record_function, ProfilerActivity #a new API from pytorch v1.8, used to be torch.autograd.profiler


#tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator") #identical to Bert tokenizer fake_sentence #시작을 못하는거네.
#gotta export tokenizer as well

#for profiling
with profile(activities=[ProfilerActivity.CPU],profile_memory=False,with_stack=True,record_shapes=True) as prof: #for CLI
#with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/electra'),profile_memory=True,with_stack=True,record_shapes=True) as prof: #for tensorboard
    with record_function("electra_inference"):
        #multithreadding should go here? 
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1) #inter-op parallelism
        print("(Intra)Thread number: ", torch.get_num_threads(), '\n') #default 2
        print("(Inter)Thread number: ", torch.get_num_interop_threads(), '\n') #default 2


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

#end of profile scope
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)) #print cpu profile result
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)) #print profile result
#print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)) #print profile result
#prof.export_chrome_trace("trace.json") #tracing fuctionality
#print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=2))


#writer.add_graph(test_model, fake_inputs)
#writer.close()

#must create a NLP data loader, for inter-op parallelism
