#is this batch job. 

from model import *
#from utils import *
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#to profile
#import torchvision.models as tvmodels
from torch.profiler import profile, record_function, ProfilerActivity #a new API from pytorch v1.8, used to be torch.autograd.profiler

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
dec = Decoder(OUTPUT_DIM, HIDDEN_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

# Transformer 객체 선언
model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

#mock forward pass
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights) #그냥 이런 식으로 apply 하는구나. (weight initialize)

#skip training: this is for the sake of model inference profiling

# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval() # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 출력 단어의 마지막 인덱스(<eos>)는 제외
            # 입력을 할 때는 <sos>부터 시작하도록 처리
            output, _ = model(src, trg[:,:-1])

            # output: [배치 크기, trg_len - 1, output_dim]
            # trg: [배치 크기, trg_len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            # 출력 단어의 인덱스 0(<sos>)은 제외
            trg = trg[:,1:].contiguous().view(-1)

            # output: [배치 크기 * trg_len - 1, output_dim]
            # trg: [배치 크기 * trg len - 1]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)



#번역 함수 def translate_sentence에서 토크나이저를 안써도 되게 수정하자.
model.eval() # 평가 모드

batch_src_indexes = [[2, 8, 364, 10, 134, 70, 624, 565, 19, 780, 200, 20, 88, 4, 3] for i in range(10)] # batch화 하기.


#with profile(activities=[ProfilerActivity.CPU],profile_memory=False,with_stack=True,record_shapes=True) as prof: #for CLI
with profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/batched_trm'),profile_memory=True,with_stack=True,record_shapes=True) as prof: #for tensorboard
    with record_function("model_inference"):
        for idx, src_indexes in enumerate(batch_src_indexes):
            src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

            # 소스 문장에 따른 마스크 생성
            src_mask = model.make_src_mask(src_tensor)

            # 인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
            with torch.no_grad(): #disable gradient calculation.
                enc_src = model.encoder(src_tensor, src_mask)

            # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
            trg_indexes = [2] #타겟의 embedding 값, 처음엔 <sos>로 시작하고, 하나씩 .append()한다.

            for i in range(max_len): #아 이런 식으로 inference 하는구나.
            #
                trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

                # 출력 문장에 따른 마스크 생성
                trg_mask = model.make_trg_mask(trg_tensor) #근데 이미 딱 앞의 정보 밖에 없는데? , 아 그래도 길이는 max_length니깐.

                with torch.no_grad():
                    output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask) #trg을 디코더에 돌린다.

                # 출력 문장에서 가장 마지막 단어만 사용
                pred_token = output.argmax(2)[:,-1].item() #마지막 문장을
                trg_indexes.append(pred_token) # 출력 문장에 더하기

                # <eos>를 만나는 순간 끝
                if pred_token == 3:
                    break
            print("task # ", idx, "complete! ")

#end of profile scope
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)) #print cpu profile result
#print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)) #print profile result
#print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10)) #print profile result
#prof.export_chrome_trace("trace.json") #tracing fuctionality
#print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total", row_limit=2))

# 각 출력 단어 인덱스를 실제 단어로 변환- -> 필요 없음
trg_tokens = trg_indexes#[trg_field.vocab.itos[i] for i in trg_indexes]

# 첫 번째 <sos>는 제외하고 출력 문장 반환
print(trg_tokens[1:]) #, attention

#for onnx
'''
print('exporting model')
torch.save(model.state_dict(), 'trm.pth')
print('export complete')
'''
