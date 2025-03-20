


class IceCreamEduBench(ImageBaseDataset):

    def build_prompt(self, line):
        img_path = line['image_path']
        context = line['context']
        question = lint['question']
        q_type = line['q_type']

        prompt = ''

        if context is not None:
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

        msgs = []
        if isinstance(img_path, list):
            msgs.extend([dict(type='image', value=p) for p in img_path])
        else:
            msgs = [dict(type='image', value=image_path)]
        msgs.append(dict(type='text', value=prompt))

    # def evaluate(self, eval_file, **judge_kwargs):
    
