import torch
from rouge_metric import Rouge
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration
from konlpy.tag import Mecab

model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')

#text = input()
text = '''
수세에 몰린 미국은 2019년 11월 OECD 회의에서 세금 부과 대상에 제조업을 포함하고 명칭도 ‘디지털세(Digital tax)’로 전환하는 안을 내놓았다.
 OECD는 올 7월 주요 20개국(G20)을 포함해 138개국의 승인을 거친 최종 성명문을 발표했다. 
 성명문은 매출 발생국이 연결 매출액 200억 유로(28조여 원)·이익률 10% 이상을 충족하는 다국적 기업에 대해 해외 국가 이익 중 10%를 넘는 초과이익에 25%의 법인세를 부과하는 내용을 
 담고 있다. OECD는 2025년 발효를 목표로 올 하반기에 다자조약안을 공개할 예정이다.

월스트리트저널은 최근 내년 초부터 정보기술(IT) 기업들을 상대로 매출의 3%에 해당하는 
‘디지털서비스세’를 2022년부터 소급 부과하겠다는 캐나다의 방침에 미국이 보복 조치를 경고했다고 보도했다. 
당초 2024년까지 조약이 시행되지 않으면 디지털서비스세 부과를 예고했던 캐나다가 과세의 칼을 꺼내자 양국 간의 갈등이 수면 위로 부상한 것이다.
 디지털세가 도입되면 삼성전자와 SK하이닉스 등도 해외에서 얻은 수익에 대해 세금을 내야 한다. 그럴 경우 우리 기업들의 세 부담 증가와 정부의 세수 감소가 불가피하다. 
 주요국들이 자국우선주의를 노골화하는 상황이므로 정부와 기업이 지혜를 모아 치밀하게 대비해야 한다.
'''


text = text.replace('\n', ' ')

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)


print(output)



'''
Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=1000,
                           length_limit_type='words',

                           alpha=0.5) # Default F1_score

    def __init__(
        self,
        metrics=None,          #['rouge-n', 'rouge-l', 'rouge-w']
        max_n=None,              #max_n=4
        limit_length=True,           ㅇㅇㅇㅇㅇ
        length_limit=1000,            ㅇㅇㅇㅇㅇ 
        length_limit_type="words",      ㅇㅇㅇㅇㅇ 
        apply_avg=True,
        apply_best=False,
        use_tokenizer=True,
        alpha=0.5,                 # Default F1_score
        weight_factor=1.0,
    ):
'''