# crownscope

생성형 AI로 대량 생산된 것처럼 보이는 한국어 블로그 글과 쇼핑 리뷰의 순위를 매기고, 모든 점수의 근거를 보여 줍니다.
사람이 검토할 항목을 고르는 선별 도구이며, 저자 판별기가 아닙니다. 판정을 내리지 않습니다.

🌐 [English README](README.md) · 쇼케이스 사이트(English, 한국어, 日本語, 简体中文, 繁體中文): <https://crownscope.codeforlife.dev> (아직 공개 전)

## 현황

초기 단계이며 **공개 배포 전**입니다. PyPI에 올라 있지 않고 라이선스도 아직 정하지 않았습니다. 소스에서 설치합니다.

```
pip install git+https://github.com/zafrem/crownscope             # 코어: PyYAML만 필요
pip install "crownscope[embed] @ git+https://github.com/zafrem/crownscope"   # + 문장 임베딩
pip install "crownscope[model] @ git+https://github.com/zafrem/crownscope"   # + 언어 모델 신호
```

**라벨이 붙은 실제 한국어 데이터로는 아직 평가하지 않았습니다.** 기본 팩(`blog`, `shopping`)의 가중치, 임계값, 기준값은
임시값입니다. 쇼케이스 사이트의 모든 수치는 도구를 시험하려고 만든 합성 데이터에서 나온 것입니다. 지금까지의 언어 모델
평가는 작은 합성 데이터에서 `ppl`을 Qwen2.5 0.5B와 1.5B로 본 것뿐이며, 약한 신호였고 한 데이터에서는 방향이 반대였습니다.

## 사용

```python
from crownscope import Detector

result = Detector(pack="shopping").analyze(text)   # Result: 등급, 점수, 신호별 기여, 근거
```

```
# CSV 순위 매기기. --cumulative는 클러스터, 앞선 문서 재채점, 작성자별 보고서를 더합니다
crownscope analyze reviews.csv --pack shopping --cumulative --report-by author

# 라벨 파일로 팩 측정하기. 신호 그룹을 하나씩 빼며 확인합니다
crownscope evaluate labeled.csv --pack shopping --cumulative --ablation

# 라벨 파일에서 가중치와 임계값 학습하기. 교차 검증 포함
crownscope fit labeled.csv --pack shopping --cumulative -o shopping-fitted.yml
```

출력에는 ID, 메타데이터, 점수, 신호, 근거만 있고 문서 원문은 없습니다. Excel로 저장한 한국어 CSV는 `--encoding cp949`로 읽습니다.

## 신호

명세의 신호 열 개를 모두 구현했습니다. 실행할 수 없는 신호는 평균에서 제외되고 이유를 밝히며, 0으로 치지 않습니다.
500자 미만의 글은 중복이나 반복된 구성이 발견되지 않으면 점수를 매기지 않습니다.

| 신호 | 보는 것 | 필요한 것 | 상태 |
| --- | --- | --- | --- |
| `pattern` | 링크, 전화번호, 홍보 문구, 협찬 고지 | 없음 | 구현됨 |
| `style` | 반복, 어휘 다양성, 문장 리듬, 판에 박힌 문구, 한국어 문장 종결, 번역투 | 없음 | 구현됨 |
| `near_dup` | 창 안의 거의 같은 문서(MinHash, LSH) | 스트림 | 구현됨 |
| `structure` | 반복되는 구성 | 스트림 | 구현됨 |
| `novelty` | 빌려 온 내용, 빠진 사실, 선택적으로 바꿔 쓴 문장 | 스트림, 모델은 선택 | 구현됨 |
| `embed_sim` | 의미로 찾는 바꿔 쓴 복사본 | 스트림, 임베딩 모델 | 구현됨, 실제 모델로는 아직 실행하지 않음 |
| `ppl` | 언어 모델에게 글이 얼마나 예측 가능한지 | 언어 모델 | 구현됨, 첫 한국어 시험에서는 약한 신호 |
| `burst` | 문장마다 달라지는 예측 가능성의 변동 | 언어 모델 | 구현됨, 아직 평가 전 |
| `fast_detect` | 확률 곡률(Fast-DetectGPT 방식) | 언어 모델 | 구현됨, 아직 평가 전 |
| `binoculars` | 관찰자/수행자 퍼플렉시티 비율 | 토크나이저가 같은 모델 두 개 | 구현됨, 아직 평가 전 |

`embed_sim`, `ppl`, `burst`, `fast_detect`, `binoculars`는 기본 팩에서 꺼져 있습니다. 팩의 `signals`와 `weights`에 넣고
옵션에 모델을 지정하면 켜지며, 그렇지 않으면 아무것도 내려받거나 불러오지 않습니다.

## 번역

이 한국어 README와 쇼케이스 사이트의 일본어·중국어 문장은 AI 어시스턴트가 썼으며 원어민 검수를 거치지 않았습니다.

라벨 데이터, 테스트, 명세는 비공개 저장소 `crownscope-forge`에 있습니다.
