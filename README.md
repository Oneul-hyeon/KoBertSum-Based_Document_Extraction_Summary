# Korean-BertSum
## Before starting

이 코드는 https://github.com/uoneway/KoBertSum.git 에서 fork 했음을 알려드립니다.

기존의 코드는 torch 버전 1.1.0에서 수행되었습니다.

하지만 이것과 더불어 몇 개의 문법적인 부분은 수정되었기 때문에 업데이트가 필요합니다.

따라서 이 코드는 torch version 2.0.1에서도 원활히 작동되도록 수정되었습니다.

추가적으로, 기존에 KoBertSum에서 사용되었던 데이콘의 한국어 문서 추출 및 요약 AI 공모전에서 제공한 [Bflysoft-뉴스기사 데이터셋](https://dacon.io/competitions/official/235671/data/) 데이터는 더 이상 해당 페이지에서 데이터를 제공하지 않으며, AI Hub의 [문서요약 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=97)로 이전되었기 때문에 해당 이슈에 대한 해결이 적용된 버전임을 알려드립니다.

## Introduce Model
### BertSum이란?

BertSum은 BERT 위에 inter-sentence Transformer 2-layers 를 얹은 구조를 갖습니다. 이를 fine-tuning하여 extract summarization을 수행하는 `BertSumExt`, abstract summarization task를 수행하는 `BertSumAbs` 및 `BertSumExtAbs` 요약모델을 포함하고 있습니다.

- 논문:  [Text Summarization with Pretrained Encoders](https://arxiv.org/abs/1908.08345) (EMNLP 2019 paper)
- 원코드: https://github.com/nlpyang/PreSumm

기 Pre-trained BERT를 summarization task 수행을 위한 embedding layer로 활용하기 위해서는 여러 sentence를 하나의 인풋으로 넣어주고, 각 sentence에 대한 정보를 출력할 수 있도록 입력을 수정해줘야 합니다. 

이를 위해

- Input document에서 매 문장의 앞에 [CLS] 토큰을 삽입하고
    ( [CLS] 토큰에 대응하는 BERT 결과값(T[CLS])을 각 문장별 representation으로 간주)

- 매 sentence마다 다른 segment embeddings 토큰을 더해주는 interval segment embeddings을 추가합니다.

  ![BERTSUM_structure](/images/BERTSUM_structure.PNG)

### Korean-BertSum이란?

Korean-BertSum은 ext 및 abs summarizatoin 분야에서 우수한 성능을 보여주고 있는 [BertSum모델](https://github.com/nlpyang/PreSumm)을 한국어 데이터에 적용할 수 있도록 수정한 한국어 요약 모델입니다.

현재는

- Pre-trained BERT로 [KoBERT](https://github.com/SKTBrain/KoBERT)를 이용합니다. 원활한 연결을 위해 [Transformers(](https://github.com/monologg/KoBERT-Transformers)[monologg](https://github.com/monologg/KoBERT-Transformers)[)](https://github.com/monologg/KoBERT-Transformers)를 통해 Huggingface transformers 라이브러리를 사용합니다.

- `BertSumExt`모델만 지원합니다.

- 이용 Data로 AI Hub의 [문서요약 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=97)에 맞춰져 있습니다.

업데이트 계획은 다음과 같습니다.

- [ ] Pre-trained BERT로 [KoBERT ](https://github.com/SKTBrain/KoBERT)외 타 모델 지원(Huggingface transformers 라이브러리 지원 모델 위주)


## Environment Setting

  Korean-BertSum에 사용된 환경에 대해 소개합니다.

  ```
  GPU : GeForce RTX 3080
  python : 3.8.18
  CuDA : 11.6
  ```

## Install
### 필요 라이브러리 설치

```
python main.py -task install
```

## Usage
### 1. 데이터 Preprocessing
  fork 이전의 방식을 통해 `make_data.py`를 수행하기 위해서는 [문서요약 텍스트](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=97)의 json 형식의 데이터를 jsonl 형식으로 변환해 `ext/data/raw`에 저장해야 할 필요가 있습니다.

  이를 위해 데이터를 `json_raw_data`에 넣어줍니다.

  `json_raw_data` 폴더의 구조는 아래와 같이 되어야 합니다.

  ```
  Training
    > 법률_train_original
      >> train_original.json
    > 사설_train_original
      >> train_original.json
    > 신문기사_train_original
      >> train_original.json
  Validation
    > 법률_valid_original
      >> valid_original.json
    > 사설_valid_original
      >> valid_original.json
    > 신문기사_valid_original
      >> valid_original.json
  ```

  데이터를 json 형식으로 변환하기 위해 아래의 코드를 실행합니다.

  ```
  python Datamaking.py
  ```
  
  이 코드의 결과로 `exe/data/raw` 에 `train.jsonl`, `test.jsonl` 파일이 생성됩니다.

  다음을 실행하여 BERT 입력을 위한 형태로 변환합니다.

  - `n_cpus`: 연산에 이용할 CPU 수

  ```
  python main.py -task make_data -n_cpus 2
  ```
  
  결과는 `ext/data/bert_data/train_abs` 및  `ext/data/bert_data/valid_abs` 에 저장됩니다.

### 2. Fine-tuning

  KoBERT 모델을 기반으로 fine-tuning을 진행하고, 1,000 step마다  Fine-tuned model 파일(`.pt`)을 저장합니다. 

  - `target_summary_sent`: `ext` . ( 현재는 `ext`만 지원) 
  - `visible_gpus`: 연산에 이용할 gpu index를 입력. 
    예) (GPU 3개를 이용할 경우): `0,1,2`

  ```
  python main.py -task train -target_summary_sent abs -visible_gpus 0
  ```

  결과는  `models` 폴더 내 `Oneul_KoBERT` 폴더에 저장됩니다. 

### 3. Validation

Fine-tuned model마다 validation data set을 통해 inference를 시행하고, loss 값을 확인합니다.

model_path: model 파일(.pt)이 저장된 폴더 경로
python main.py -task valid -model_path Oneul_KoBERT
결과는 ext/logs 폴더 내 valid_Oneul_KoBERT.log 형태로 저장됩니다.

valid_Oneul_KoBERT.log 내 마지막 라인에

```
PPL [loss가 적은 순으로 정렬된 모델명] 
```
으로 표현됩니다.

첫 번째 모델이 Validation을 통해 확인한 성능이 가장 우수한 model 파일이 됩니다.

### 4. Prediction

Validation을 통해 확인한 가장 성능이 우수한 model파일을 통해 실제로 텍스트 요약 과업을 수행합니다.

이전 Validation 세션에서 확인한 가장 우수한 model 파일명을 확인해야 합니다.

해당 실험에서는 `model_step_28000.pt`가 가장 우수했고 이에 해당 모델을 이용합니다.

  - `test_from`:  model 파일(`.pt`) 경로
  - `visible_gpus`: 연산에 이용할 gpu index를 입력. 
    예) (GPU 3개를 이용할 경우): `0,1,2`

  ```
  python main.py -task test -test_from Oneul_KoBERT/model_step_28000.pt -visible_gpus 0
  ```

Prediction 결과는 테스트 데이터의 `original_text`, `extractive`, `extractive_sents`와 함께 `result_df.csv`에 저장됩니다.

### 5. Rouge-Score

정답 요약문과 예측 요약문 간 Rouge-Score 값을 산출합니다.

score는 `Rouge-1`, `Rouge-2`, `Rouge-L`에 대해 평가합니다.

  ```
  python calculate_rouge_score.py
  ```
실행 결과로 각 Rouge-Score의 Precision, Recall, F1 평균값이 산출됩니다.


### 6. Test & Demo

Validation을 통해 확인한 가장 성능이 우수한 model파일을 통해 실제로 텍스트 요약을 테스트합니다.

`evaluate.py`를 통해 확인할 수 있습니다.

해당 파일 내 사용하고자 하는 모델의 위치는 변경되어야 합니다.

`get_summary()` 함수를 통해 주어진 테스트 데이터의 인풋을 넣으면 요약된 텍스트가 출력됩니다.

현재는 상위 3개의 텍스트가 출력되도록 설정되어 있습니다.

또한 간단한 Demo로 Gradio 라이브러리를 활용합니다.

`demo.py` 파일을 실행시켜 입력에 대한 요약된 결과를 확인할 수 있습니다.

(파일 내 `demo.launch()`는 해당 Demo를 활용하고자 하는 사용자에 맞게 변경해야 합니다.)
