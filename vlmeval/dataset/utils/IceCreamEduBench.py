import re
import os

# def extract_answer_number(text: str) -> str:
#     # 다양한 패턴을 포함하는 정규식
#     match = re.search(r"(정답은|옳은 것은|맞는 것은|답은)\s*(\d+)번", text)
#     if match:
#         return match.group(2)
#     else:
#         return text  # 매칭 안 될 경우 원문 반환

######## options 평가 방법 ########
## 답변 전처리
def circled_number_to_digit(text: str) -> str:
    circled_number_map = {
        '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'
    }
    # 각 문자에 대해 변환, 없으면 그대로 둠
    return ''.join(circled_number_map.get(char, char) for char in text)

## 예측값 전처리
def extract_answer_number(text: str) -> str:
    text = text.strip()
    
    # 쉼표가 있는 경우: 각 항목을 개별 전처리
    if ',' in text:
        parts = text.split(',')
        cleaned_parts = []
        for part in parts:
            part = part.strip()
            match = re.match(r'^(\d+)(번)?(\.)?(\s+.*)?$', part)
            if match:
                cleaned_parts.append(match.group(1))
            else:
                cleaned_parts.append(part)
        return ', '.join(cleaned_parts)
    
    # 쉼표 없는 경우: 단일 처리
    match = re.match(r'^(\d+)(번)?(\.)?(\s+.*)?$', text)
    if match:
        return match.group(1)
    else:
        return text
# def extract_answer_number(text: str) -> str:
#     # "숫자+번" 패턴 모두 찾기
#     matches = re.findall(r"(\d+)번", text)
#     if matches:
#         return ", ".join(matches)
#     else:
#         return text  # 매칭 안 될 경우 원문 반환
    
def normalize_answer(answer: str) -> set:
    # 숫자 추출 → 정수 변환 → 집합
    return set(map(int, re.findall(r'\d+', answer)))

def is_correct_prediction(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


######## judge 평가 방법 ########
def get_eval(judge, message):
    text, image_path = message

    if image_path is None:
        return judge.generate(text)
    else:
        msg = [
            {"type": "text", "value": text},
            {"type": "image", "value": image_path}
        ]
        return judge.generate(msg)

def build_prompt(line):
    a_type = line['a_type']
    question = line['question']
    answer = line['answer']
    prediction = line['prediction']

    user_prompt = (
        f"정답 유형: {a_type}\n문제: {question}\n정답 (ground truth): {answer}"
        f"\n모델 예측값 (prediction): {prediction}\n출력 숫자:"
    )

    if a_type in ["number", "word", "symbol", "formula", "md"] or "mixed" in a_type:
        system_prompt = (
            "다음은 정답 유형과 모델의 예측값, 정답 데이터입니다. "
            "모델의 예측값(prediction)이 주어진 정답 유형에 따라 정답(ground truth)과 의미적으로 일치하면 **1**, "
            "의미가 다르거나 틀렸다면 **0**을 출력하세요. 출력은 반드시 숫자 하나(1 또는 0)만 하세요."
        )
        return system_prompt + "\n\n" + user_prompt, None

    elif a_type == "descriptive":
        system_prompt = (
            "다음은 서술형 정답 유형에 해당하는 문제의 모델 예측값과 정답 데이터입니다. "
            "예측값이 문제 해결 과정을 논리적으로 서술하고, 정답과 동일한 결론에 도달했다면 1, 그렇지 않으면 0을 출력하세요. "
            "풀이 과정이 생략되었거나 논리적으로 타당하지 않거나, 결론이 틀릴 경우에는 0으로 판단하세요. "
            "출력은 반드시 숫자 하나(1 또는 0)만 하세요."
        )
        return system_prompt + "\n\n" + user_prompt, None

    elif a_type == "image":
        system_prompt = (
            "당신은 입력된 이미지를 기반으로 정답 여부를 판단하는 평가자입니다. "
            "이미지를 보고 문제의 정답이 무엇인지 판단한 후, 모델의 예측값(prediction)이 해당 정답(ground truth)과 의미적으로 "
            "일치하는지를 평가하세요. 예측값이 정답과 동일하거나, 이미지의 정답 요소를 정확히 지칭하고 있다면 `1`, "
            "틀리거나 다른 요소를 선택했다면 `0`을 출력하세요. 출력은 반드시 숫자 하나(1 또는 0)만 하세요."
        )
        img_path = os.path.join(
            "/Users/yoojin_ha/Desktop/Dev/Data/VLM_Data/아이스크림에듀/origin_data/image",
            re.match(r"img_(\d+)_\d+", line['answer'].strip("<>")).group(1),
            line['answer'].strip("<>") + ".png",
        )
        return system_prompt + "\n\n" + user_prompt, img_path

    else:
        raise ValueError(f"Unknown answer type: {a_type}")


def IceCreamEduBench_atomeval(model, line):
    message = build_prompt(line)  # returns (text, image_path)
    result = get_eval(model, message)
    return result

##########################

def extract_short_answer(text: str) -> str:
    text = text.strip()

    # 0. 마크다운 표 또는 특수 기호가 포함된 표현이면 그대로 유지
    if any(symbol in text for symbol in ["|", "---"]):
        return text

    # 1. LaTeX 수식 전체가 있는 경우
    latex_match = re.fullmatch(r"\$(.*?)\$", text)
    if latex_match:
        inner = latex_match.group(1)
        # 단순 숫자 수식이면 숫자만 추출
        if re.fullmatch(r"[-+]?\d*\.?\d+", inner):
            return inner
        else:
            return text  # 복잡한 수식은 유지

    # 2. 일반 문자열에서 수식 기호 제거
    cleaned = text.replace("$", "")

    # 3. 숫자만 있을 경우 그대로 반환
    match = re.fullmatch(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        return match.group(0)

    # 4. 숫자가 포함된 경우: 숫자만 추출
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if match:
        return match.group(0)

    # 5. 그 외 (한글, 원형 한글 등): 그대로 반환
    return text




def eval_multi_choice(prediction, answer):

    if answer == prediction:
        correct = True
    else:
        correct = False
    return correct