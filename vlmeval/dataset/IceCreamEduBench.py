from .image_base import ImageBaseDataset
import pandas as pd
from .utils.IceCreamEduBench import *
from ..smp import *

class IceCreamEduBench(ImageBaseDataset):

    DATASET_URL = {
        'IceCreamEdu_TEST': '/Users/yoojin_ha/Desktop/개발/VLM_Evaluation/VLMEvalKit/LMUData/IceCreamEdu_TEST.tsv',
    }

    DATASET_MD5 = {
        'IceCreamEdu_TEST': '599c102c1486e7da9f3737dc286ad89c', # 변경 필요
        # 'IceCreamEdu_TEST': '6b5a9ce3e6057be4b667ec6551febeef',
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
            
            options_prompt = ''
            if A is not None:
                options_prompt += f'1. {A}\n'
            if B is not None:
                options_prompt += f'2. {B}\n'
            if C is not None:
                options_prompt += f'3. {C}\n'
            if D is not None:
                options_prompt += f'4. {D}\n'
            if E is not None:
                options_prompt += f'5. {E}\n'
            mcq_prompt = '위의 선택지 중 올바른 번호를 선택해 주세요.'

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
        msgs.append(dict(type='text', value=prompt))
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
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]

        for line in lines:
            q_type = line['q_type']
            answer = line['answer']
            prediction = line['prediction']
            
            correct = eval_multi_choice(answer, prediction)

            if correct:
                judge_dict[line['data_id']] = 'Correct'
                pred_correct += 1
            else:
                judge_dict[line['data_id']] = 'Wrong'
        
        return judge_dict, {'acc': pred_correct / len(lines)}

