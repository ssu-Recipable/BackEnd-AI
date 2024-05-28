from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
#from keras.preprocessing.sequence import pad_sequences
from typing import List
import numpy as np
import pandas as pd
import uvicorn

tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
model = BertForSequenceClassification.from_pretrained('kykim/bert-kor-base', num_labels=318)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

def pad_sequences(input_ids, maxlen) :
    result = []
    for i in input_ids :
        while len(i) != maxlen :
            i.append(0)
        result.append(i)
    return result

# 입력 데이터 변환
def convert_input_data(sentences):
    sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in sentences]
    # BERT의 토크나이저로 문장을 토큰으로 분리
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # 입력 토큰의 최대 시퀀스 길이
    MAX_LEN = 64

    # 토큰을 숫자 인덱스로 변환
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    
    # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN)

    # 어텐션 마스크 초기화
    attention_masks = []

    # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
    # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # 데이터를 파이토치의 텐서로 변환
    inputs = torch.tensor(input_ids)
    masks = torch.tensor(attention_masks)

    return inputs, masks



# 문장 테스트
def test_sentences(sentences):

    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = convert_input_data(sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs
    b_input_mask = masks
            
    # 그래디언트 계산 안함
    with torch.no_grad():     
        # Forward 수행
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)

    # 로스 구함
    logits = outputs[0]

    # CPU로 데이터 이동
    logits = logits.detach().cpu().numpy()

    return logits

app = FastAPI()

class InputData(BaseModel):
    sentences: List[str]

@app.post("/predict/")
async def predict(input_data: InputData):
    sentences = input_data.sentences
    logits = test_sentences(sentences=sentences)
    data = pd.read_csv("label.csv")
    result = []
    for a in logits :
        result.append(data[data['number'] == np.argmax(a)]['label'].values[0])
    return {"predictions" : result}

if __name__ == "__main__" :
	uvicorn.run("main:app", reload=True, port=8080)
