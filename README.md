# enthimo ROC 분석 도구

이 저장소에는 그룹이 0/1/2 등으로 표시된 CSV 파일을 대상으로 ROC 커브 통계를 한 번에 계산하는 파이썬 스크립트 `roc_analysis.py`가 포함되어 있습니다. MedCalc로 변수를 하나씩 반복하던 작업을 자동화하여 **민감도(Sensitivity), 특이도(Specificity), AUC, p-값(DeLong et al. 비모수 검정)**을 산출합니다.

## 필요한 패키지
```
pip install pandas numpy scipy scikit-learn
```

## 사용법
기본적으로 모든 숫자형 컬럼(라벨 컬럼 제외)에 대해 **모든 클래스 쌍을 일대일로** 비교합니다.

```bash
python roc_analysis.py <데이터.csv> --label 그룹컬럼이름
```

### 주요 옵션
- `--features col1 col2 ...` : 분석할 변수 목록을 지정합니다(미지정 시 숫자형 컬럼 자동 선택).
- `--classes POS NEG` : 양성/음성으로 비교할 두 개의 클래스만 지정합니다(예: `--classes 1 0`).
- `--output 결과.csv` : 결과 테이블을 CSV로 저장합니다.

### 예시
```bash
# 모든 숫자 컬럼을 이용해 클래스 쌍(0vs1, 0vs2, 1vs2)별 결과 출력
python roc_analysis.py data.csv --label group

# 절대 경로(윈도우)도 그대로 사용 가능 - 백슬래시 이스케이프 방지를 위해 따옴표로 감싸세요
python roc_analysis.py "C:\\Users\\seokj\\OneDrive\\Desktop\\Probability\\ROI_dataset.csv" --label group

# 특정 변수들로 1(양성) vs 0(음성) 비교 후 결과 파일 저장
python roc_analysis.py data.csv --label group --classes 1 0 --features marker1 marker2 --output roc_results.csv
```

### 파이썬 코드에서 직접 사용하기
`path1 = r"..."`처럼 파일 경로 변수를 두고 싶다면 CLI 대신 함수를 호출하면 됩니다. 모든 경로는 상대/절대 경로 모두 허용되며, 반환값은 CLI와 동일한 컬럼을 가진 `pandas.DataFrame`입니다.

```python
from roc_analysis import run_roc_analysis

path1 = r"C:\\Users\\seokj\\OneDrive\\Desktop\\Probability\\ROI_dataset.csv"
results = run_roc_analysis(path1, label="group", class_pair=[1, 0])
print(results)
```

출력에는 변수명, 비교 클래스(양성/음성), AUC, 민감도, 특이도, Youden 지수 기준 임계값, p-값, 각 클래스 샘플 수가 포함됩니다.
