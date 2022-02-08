#since we are modeling BERT, the bidirectional encoder representation for Transformers. We leave only the encoder part, which gets really simple.

from model import *
#from utils import *
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#to profile
#import torchvision.models as tvmodels
#from torch.profiler import profile, record_function, ProfilerActivity #a new API from pytorch v1.8, used to be torch.autograd.profiler

INPUT_DIM = 7855 #from colab
OUTPUT_DIM = 5893 #from colab
HIDDEN_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

max_len = 50

#토크나이저는 생략. colab에서 토크나이저 거친 데이터구조를 입력해주자.
SRC_PAD_IDX = 1
TRG_PAD_IDX = 1

# 인코더(encoder)와 디코더(decoder) 객체 선언
enc = Encoder(INPUT_DIM, HIDDEN_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
#dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
#model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

#mock forward pass
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

enc.apply(initialize_weights) #그냥 이런 식으로 apply 하는구나. (weight initialize)


src_indexes = [2, 8, 364, 10, 134, 70, 624, 565, 19, 780, 200, 20, 88, 4, 3]
src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

src_mask = [[[[True, True, True, True, True, True, True, True, True, True, True,True, True, True, True]]]]
src_mask = torch.Tensor(src_mask)

#인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
enc_src = enc(src_tensor, src_mask)
#print(enc_src)
print('inference of encoder complete!')
print('saving as torchscript')
traced_model = torch.jit.trace(enc,[src_tensor,src_mask])
torch.jit.save(traced_model,'traced_enc.pt')
print('export complete!')
