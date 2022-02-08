import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded = torch.jit.load('traced_enc.pt')

src_indexes = [2, 8, 364, 10, 134, 70, 624, 565, 19, 780, 200, 20, 88, 4, 3]
src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

src_mask = [[[[True, True, True, True, True, True, True, True, True, True, True,True, True, True, True]]]]
src_mask = torch.Tensor(src_mask)

#인코더(endocer)에 소스 문장을 넣어 출력 값 구하기
enc_src = loaded(src_tensor, src_mask)

print('loading & execution complete!')
