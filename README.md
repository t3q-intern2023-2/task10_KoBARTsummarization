# KoBART-summarization

## Load KoBART
- huggingface.co에 있는 binary를 활용
  - https://huggingface.co/gogamza/kobart-base-v1

## Download binary
- https://www.hankookilbo.com/News/Read/201910301596013777 [원문기사 링크]

```python
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

text = """
더불어민주당 이해찬 대표가 30일 기자간담회를 열고 ‘조국 사태’와 관련, “국민 여러분께 매우 송구하다”는 입장을 밝혔다. 이 대표는 “검찰 개혁이란 대의에 집중하다 보니, 국민 특히 청년이 느꼈을 불공정에 대한 상대적 박탈감, 좌절감을 깊이 있게 헤아리지 못했다”며 “여당 대표로서 무거운 책임감을 느낀다”고 머리를 숙였다. 조국 전 법무부 장관이 14일 사퇴한 이후 이 대표가 당 안팎의 쇄신 요구에 대해 입장을 표명한 것은 이번이 처음이다.

청와대와 여당은 ‘조국 정국’을 거치며 분출된 ‘공정’과 ‘정의’의 민심을 받들어 검찰 개혁에 매진하겠다면서도 두 달간 극심한 분열과 갈등을 초래한데 대해선 진지하게 성찰하는 모습을 보이지 않았다. 그나마 초선인 이철희 의원이 “당이 대통령 뒤에 비겁하게 숨어 있었다”고 비판했고, 표창원 의원은 “책임을 느끼는 분들이 각자 형태로 그 책임감을 행동으로 옮겨야 할 때”라고 지적했다. 뒤늦게나마 이 대표가 자성의 목소리를 내긴 했으나 당 안팎의 쇄신 요구에 어떻게 응할지 구체적 플랜을 제시하지 못해 여전히 안이하다는 지적도 나온다.

이 대표는 28일 윤호중 사무총장을 단장으로 하는 총선기획단을 발족했고 조만간 인재영입위원회도 출범시킬 계획이라고 밝혔다. 이 대표는 “민주당의 가치를 공유하는 참신한 인물을 영입해 준비된 정책과 인물로 승부하겠다”고 다짐했다. 하지만 당 일각에선 “총선기획단장을 비롯한 당직 인선부터 쇄신 의지를 보여야 한다”는 비판의 목소리가 나온다. 무조건 물러나는 게 능사는 아니지만 국정 혼선을 초래한 데 대해 당 지도부가 겸허하게 책임지는 모습을 보이는 게 쇄신의 출발점이 돼야 한다는 지적도 있다.

선거는 대중의 이해와 요구를 잘 대표하는 정치인을 뽑는 행위다. 민생을 외면하며 낡은 이념과 진영 싸움에 매몰된 구시대 인물들을 과감히 물갈이하라는 게 국민의 요구다. 대신 4차 산업혁명의 거센 파고를 헤쳐나갈 전문성을 갖춘 젊고 유능한 인재들을 널리 구해야 하다. 민주당은 과연 어떤 인물과 정책으로 내년 총선을 치를 것인가. 이해찬 대표의 이날 유감 표명이 여권 전반의 대대적인 인적 쇄신으로 이어지길 기대한다.
"""

text = text.replace('\n', ' ')

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

'이해찬 대표가 조국 사태와 관련 송구한 입장 표명이 과감한 인적 쇄신으로 이어져야 한다.'

```
## Requirements
```
numpy,
pandas,
matplotlib,
transformers==4.8.2,
pyyaml==5.4.1,
torchtext==0.8.0
pytorch_lightning==1.5.2,
pytorch>=1.10.0
transformers==4.16.2
streamlit==1.2.0

```
## Data
✔ **추출모델과 데이터 통일 [TASK 11 팀]**

✔ AI-HUB 문서요약 텍스트 데이터 활용

✔ 학습 데이터에서  Train / Test 데이터를 생성함 (뉴스, 사설)

✔ 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함

✔  default로 data/train.tsv, data/test.tsv 형태로 저장함
  
| news  | summary |
|-------|--------:|
|  원문  | 요약문  |  

##
```
- Train 조건
  -프리트레이닝 모델 : 뉴스, 사설 2만개 학습한 모델
  -프리트레이닝 토크나이저 : gogamza/kobart-base-v1
  ✔ Optimizer : AdamW
  ✔ workers : 4
  ✔ Batch_size:  32
  ✔ lr : 3e-05
  ✔ Max-lengh: 512
  ✔ warmup_ratio: 0.1

```
## How to Train
- KoBART summarization fine-tuning
```bash
pip install -r requirements.txt

[use gpu]
python train.py  --gradient_clip_val 1.0  \
                 --max_epochs 50 \
                 --default_root_dir logs \
                 --gpus 1 \
                 --batch_size 32 \
                 --num_workers 4

[use cpu]
python train.py  --gradient_clip_val 1.0  \
                 --max_epochs 50 \
                 --default_root_dir logs \
                 --strategy ddp \
                 --batch_size 32 \
                 --num_workers 4
```

## Demo
- 학습한 model binary 추출 작업이 필요함
   - pytorch-lightning binary --> huggingface binary로 추출 작업 필요
   - hparams의 경우에는 <b>./logs/tb_logs/default/version_0/hparams.yaml</b> 파일을 활용
   - model_binary 의 경우에는 <b>./logs/kobart_summary-model_chp</b> 안에 있는 .ckpt 파일을 활용
   - 변환 코드를 실행하면 <b>./kobart_summary</b> 에 model binary 가 추출 됨
  
```
 python get_model_binary.py --hparams hparam_path --model_binary model_binary_path
```

- streamlit을 활용하여 Demo 실행
    - 실행 시 Demo page가 실행됨
```
streamlit run infer.py
```

## Model Performance
- TestData 기준으로 rouge score를 산출함(Rouge 결과화면)
  
<img src="C:\Users\ADMIN\Desktop\전유태_프로젝트\T3Q-summarization\KoBART\imgs\rouge.png" alt="drawing" style="width:600px;"/>
<img src="C:\Users\ADMIN\Desktop\전유태_프로젝트\T3Q-summarization\KoBART\imgs\rouge_1.png" alt="drawing" style="width:600px;"/>
<img src="C:\Users\ADMIN\Desktop\전유태_프로젝트\T3Q-summarization\KoBART\imgs\rouge_2.png" alt="drawing" style="width:600px;"/>"# task10_KoBARTsummarization" 
