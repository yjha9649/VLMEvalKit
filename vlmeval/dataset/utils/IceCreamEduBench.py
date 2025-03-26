import re

def extract_answer_number(text: str) -> str:
    # 다양한 패턴을 포함하는 정규식
    match = re.search(r"(정답은|옳은 것은|맞는 것은|답은)\s*(\d+)번", text)
    if match:
        return match.group(2)
    else:
        return text  # 매칭 안 될 경우 원문 반환

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


def circled_number_to_digit(text: str) -> str:
    circled_number_map = {
        '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5'
    }
    return circled_number_map.get(text, text)

def eval_multi_choice(answer, prediction):
    if answer == prediction:
        correct = True
    else:
        correct = False
    return correct