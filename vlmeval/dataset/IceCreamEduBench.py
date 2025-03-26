from .image_base import ImageBaseDataset
import pandas as pd
from .utils.IceCreamEduBench import *
from ..smp import *

class IceCreamEduBench(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {
        'IceCreamEdu_TEST': '/Users/yoojin_ha/Desktop/개발/VLM_Evaluation/VLMEvalKit/LMUData/IceCreamEdu_TEST.tsv',
    }

    DATASET_MD5 = {
        'IceCreamEdu_TEST': 'd46af35bc5357660a76d058d958fda30'
    }

    def build_prompt(self, line):
        img_path = line['image_path']
        context = line['context']
        question = line['question']
        q_type = line['q_type']

        prompt = ''

        if not pd.isna(context):
            prompt += f'지문: {context}\n'
        prompt += f'질문: {question}\n'

        if q_type == "multi_choices":
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
            # mcq_prompt = '위의 선택지 중 정답이 되는 번호를 작성해 주세요.\n※ 선택지 번호만 작성하고, 여러 개일 경우 쉼표로 구분해 주세요.\n※ 반드시 선택지 번호와 쉼표 외에는 아무것도 쓰지 마세요. (해설, 설명, 단어 등 금지)\n'
            mcq_prompt = '질문에 답하고, 정답인 선택지 번호만 출력하세요. (예: 1번, 2번, 3번 등. 복수 정답일 경우 모든 정답 번호를 적으세요.) 해설은 출력하지 마세요.'
            prompt += options_prompt
            prompt += mcq_prompt
            
        elif q_type == "short_answer":
            sqa_prompt = '질문에 대한 정답을 한 단어 또는 짧은 문장으로 작성해 주세요.'
            prompt += sqa_prompt
        elif q_type == "long_answer":
            lqa_prompt = '질문에 대한 정답을 구체적으로 작성해 주세요.'
            prompt += sqa_prompt

        msgs = []
        if isinstance(img_path, list):
            msgs.extend([dict(type='image', value=p) for p in img_path])
        else:
            msgs = [dict(type='image', value=img_path)]
        msgs.append(dict(type='text', value=prompt.strip()))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
    
        pred_correct = 0
        judge_dict = dict()

        data = load(eval_file)
        dataset = self.dataset_name
        data['q_type'] = [str(x) for x in data['q_type']]
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        lt = len(data)
        # pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]

        for line in lines:
            q_type = line['q_type']
            answer = line['answer']
            prediction = line['prediction']

            if q_type == "multi_choices":
                prediction = extract_answer_number(prediction)
                answer = circled_number_to_digit(answer)
            elif q_type == "short_answer":
                prediction = extract_short_answer(prediction)
                answer = extract_short_answer(answer)
            
            correct = eval_multi_choice(answer, prediction)

            if correct:
                judge_dict[line['index']] = 'Correct'
                pred_correct += 1
            else:
                judge_dict[line['index']] = 'Wrong'
        
        return judge_dict, {'acc': pred_correct / len(lines)}

