from .image_base import ImageBaseDataset
import pandas as pd
from .utils.IceCreamEduBench import *
from ..smp import *
import os
from .utils import build_judge

class IceCreamEduBench(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        'IceCreamEdu_TEST': '/Users/yoojin_ha/Desktop/Dev/VLM_Evaluation/VLMEvalKit/LMUData/IceCreamEdu_TEST.tsv',
    }

    DATASET_MD5 = {
        'IceCreamEdu_TEST': '6f4df4ed8a9a62977d253a6509548864'
    }

    def build_prompt(self, line):
        img_path = os.path.join("/Users/yoojin_ha/Desktop/Dev/VLM_Evaluation/VLMEvalKit/LMUData/images/IceCreamEdu_TEST", line['image_path'])
        context = line['context']
        question = line['question']
        a_type = line['a_type']

        prompt = ''

        if not pd.isna(context):
            prompt += f'지문: {context}\n'
        prompt += f'질문: {question}\n'

        # MCQ 유형
        if a_type == "options":
            A = line['보기1']
            B = line['보기2']
            C = line['보기3']
            D = line['보기4']
            E = line['보기5']
            
            options_prompt = '선택지:\n'
            if not pd.isna(A):
                options_prompt += f'1번. {A}\n'
            if not pd.isna(B):
                options_prompt += f'2번. {B}\n'
            if not pd.isna(C):
                options_prompt += f'3번. {C}\n'
            if not pd.isna(D):
                options_prompt += f'4번. {D}\n'
            if not pd.isna(E):
                options_prompt += f'5번. {E}\n'
                
            task_prompt = '질문에 답하고, 정답인 선택지 번호만 출력하세요. 정답이 하나일 경우 번호 하나만(예: 1번), 여러 개일 경우 쉼표로 구분하여 모두 출력하세요(예: 2번, 4번). 해설은 출력하지 마세요.'
            prompt += options_prompt
            prompt += task_prompt
            
        # 숫자 답변
        elif a_type == "number":
            task_prompt = '질문에 답하고, 정답이 되는 숫자만 출력하세요. 정답이 하나일 경우 숫자 하나만, 여러 개일 경우 쉼표나 번호 등으로 구분하여 모두 출력하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt
            
        # 단어 답변
        elif a_type == "word":
            task_prompt = '질문에 답하고, 정답이 되는 단어나 구만 출력하세요. 정답이 하나일 경우 단어 하나만, 여러 개일 경우 쉼표나 번호 등으로 구분하여 모두 출력하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt            
            
        # 기호 답변
        elif a_type == "symbol":
            task_prompt = '질문에 답하고, 정답이 되는 기호만 출력하세요. 정답이 하나일 경우 기호 하나만, 여러 개일 경우 쉼표, 번호, 슬래시 등으로 구분하여 모두 출력하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt            
            
        # 수식 답변
        elif a_type == "formula":
            task_prompt = '질문에 답하고, 정답이 되는 수식을 LaTeX 형식으로 출력하세요. 정답이 하나일 경우 수식 하나만, 여러 개일 경우 쉼표나 번호로 구분하여 모두 출력하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt            
            
        # 마크다운 답변
        elif a_type == "md":
            task_prompt = '질문에 답하고, 정답이 되는 내용을 마크다운 형식의 표로 작성하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt            
            
        # 이미지 답변
        elif a_type == "image":
            task_prompt = '질문에 답하고, 정답만 출력하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt
        
        # 복합 답변
        elif "mixed" in a_type:
            task_prompt = '질문에 답하고, 정답만 출력하세요. 해설이나 설명은 출력하지 마세요.'
            prompt += task_prompt

        elif a_type == "descriptive":
            task_prompt = '질문에 답하고, 문제 해결 과정을 서술한 뒤 최종 정답을 구하세요. 풀이 과정은 문장으로 자연스럽게 설명하고, 계산에 필요한 수치를 포함하세요. 논리적으로 결론까지 도달하도록 작성하세요.'
            prompt += task_prompt

        msgs = []
        if isinstance(img_path, list):
            msgs.extend([dict(type='image', value=p) for p in img_path])
        else:
            msgs = [dict(type='image', value=img_path)]
        msgs.append(dict(type='text', value=prompt.strip()))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
    
        pred_correct = 0

        data = load(eval_file)
        dataset = self.dataset_name
        data['a_type'] = [str(x) for x in data['a_type']]
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        # pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]

        system_prompt = """다음은 정답 유형과 모델의 예측값, 정답 데이터입니다. 모델의 예측값(prediction)이 주어진 정답 유형에 따라 정답(ground_truth)과 의미적으로 일치하면 **1**, 의미가 다르거나 틀렸다면 **0**을 출력하세요. 출력은 반드시 숫자 하나(1 또는 0)만 하세요."""

        model = build_judge(temperature=0.2, system_prompt=system_prompt, **judge_kwargs)

        log_list = []

        for line in lines:
            a_type = line['a_type']
            answer = line['answer']
            prediction = line['prediction']

            if a_type == "options":
                answer = circled_number_to_digit(answer)
                prediction = extract_answer_number(prediction)
                correct = is_correct_prediction(prediction, answer)
            # image 유형 제외
            elif a_type == "":
                user = build_prompt(a_type, prediction, answer)
            elif a_type in []:
                pass






            else:
                # prediction = extract_short_answer(prediction)
                # answer = extract_short_answer(answer)
                correct = eval_multi_choice(prediction, answer)

            if correct:
                pred_correct += 1
                log_list.append('Correct')
            else:
                log_list.append('Wrong')
        
        data['log'] = log_list
        data_log_pth = eval_file.replace('.xlsx', '_log.xlsx')
        dump(data, data_log_pth)
        
        return {'acc': pred_correct / len(lines)}