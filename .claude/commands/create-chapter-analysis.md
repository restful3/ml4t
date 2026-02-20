---
description: "Chan's Algorithmic Trading 책의 챕터별 종합 분석 스크립트(run_chapterX_analysis.py)와 마크다운 리포트를 생성합니다. 인자로 챕터 번호(2~8)를 전달하세요."
---

# 챕터 분석 스크립트 생성

Ernest Chan의 "Algorithmic Trading" 책의 챕터 분석 스크립트를 생성합니다.

## 인자

사용자가 전달한 인자: $ARGUMENTS

- 인자가 없으면, 사용자에게 챕터 번호(2~8)를 물어보세요.
- 인자가 있으면 챕터 번호로 사용하세요. (예: `3`, `chapter 5`, `ch4` 등 자연어도 파싱)

## 기본 경로

```
BOOK_ROOT="source/Chan E. Algorithmic Trading - Winning Strategies and Their Rationale 2013"
```

## 실행 절차

### Step 1: 챕터 디렉토리 탐색

`${BOOK_ROOT}/chapter_X_*` 패턴으로 해당 챕터 디렉토리를 찾으세요.
디렉토리가 없으면 사용자에게 알리고 중단하세요.

### Step 2: 참조 자료 수집

다음 파일들을 읽고 분석하세요:

1. **한글 문서**: `chapter_X_*/XX_chapter_X_*_ko.md` 파일을 읽어 핵심 개념과 문제 정의를 파악
2. **기존 소스 코드**: `chapter_X_*/src/*.py` 파일들을 분석하여 사용된 함수, 데이터 처리 방식 파악
3. **데이터 파일**: `chapter_X_*/src/inputData_*.csv` 등 데이터 파일 목록과 내용 확인
4. **Chapter 2 템플릿**: `${BOOK_ROOT}/chapter_2_the_basics_of_mean_reversion/src/run_chapter2_analysis.py`를 구조 참조용으로 읽기

### Step 3: 챕터별 핵심 분석 내용 매핑

아래 가이드를 참고하여 해당 챕터의 핵심 분석 항목을 결정하세요:

| 챕터 | 핵심 개념 | 주요 분석 항목 |
|------|-----------|---------------|
| 3 | 볼린저 밴드, 칼만 필터, 스케일링 인/아웃 | 볼린저 밴드 전략, 칼만 필터 동적 헤지, 스케일링 인 비교 |
| 4 | 횡단면 평균 회귀, 주식 페어 트레이딩 | 횡단면 vs 시계열 평균 회귀, 페어 선택, ETF 공적분 |
| 5 | 롤오버, 캐리 트레이드, 계절성 | 캐리 전략, 롤오버 비용, 계절성 전략 |
| 6 | 시계열/횡단면 모멘텀 | 모멘텀 팩터, 롱/숏 포트폴리오, 모멘텀 크래시 |
| 7 | 오프닝 갭, 일중 모멘텀 | 갭 트레이딩, 일중 가격 패턴, 거래량 분석 |
| 8 | 켈리 공식, VaR, 드로다운 관리 | 최적 레버리지, VaR/CVaR, 드로다운 청산 규칙 |

만약 위 표에 없는 챕터(예: 2)라면 한글 문서와 기존 소스에서 직접 핵심 개념을 추출하세요.

### Step 4: 분석 스크립트 생성

`chapter_X_*/src/run_chapterX_analysis.py` 파일을 생성하세요.

#### 스크립트 구조 요구사항

```python
class ChapterXAnalyzer:
    """Chapter X: [챕터 제목] 종합 분석"""

    def __init__(self, data_dir, report_dir):
        self._setup_korean_font()
        ...

    def _setup_korean_font(self):
        """matplotlib 한국어 폰트 설정 (차트에 한글 사용 시 필수)"""
        import matplotlib.font_manager as fm
        candidates = ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR',
                      'Malgun Gothic', 'AppleGothic']
        for name in candidates:
            if any(name in f.name for f in fm.fontManager.ttflist):
                plt.rcParams['font.family'] = name
                plt.rcParams['axes.unicode_minus'] = False
                return
        print("  ⚠️ 한국어 폰트를 찾을 수 없습니다. 차트 레이블은 영문으로 표시됩니다.")

    def load_data(self):
        """데이터 로드 및 전처리"""
        ...

    def analyze_[핵심개념1](self):
        """첫 번째 핵심 분석"""
        ...

    def analyze_[핵심개념2](self):
        """두 번째 핵심 분석"""
        ...

    def backtest_strategy(self):
        """전략 백테스트"""
        ...

    def generate_report(self):
        """마크다운 리포트 생성"""
        ...

    def run(self):
        """전체 분석 오케스트레이션"""
        ...

if __name__ == "__main__":
    analyzer = ChapterXAnalyzer(data_dir='.', report_dir='reports')
    analyzer.run()
```

#### 코딩 규칙

- pandas 2.0+ 호환성 유지 (`.iloc` 사용, `append` 대신 `pd.concat` 등)
- 코드 주석은 한글로 작성
- 콘솔 출력에 이모지와 구분선 사용하여 진행 상황 표시
- 차트는 `matplotlib`으로 생성하여 `reports/figures/` 에 PNG로 저장
- 차트에 한국어 텍스트(title, label, legend 등)를 사용할 경우 반드시 `_setup_korean_font()` 호출하여 CJK 폰트 설정할 것 (미설정 시 한글이 네모로 렌더링됨)

### Step 5: 리포트 형식

`generate_report()` 메서드가 생성하는 마크다운 리포트(`reports/chapterX_report.md`)는 다음 섹션을 포함해야 합니다:

1. **개요 및 문제 정의**
   - 이 챕터에서 해결하려는 핵심 질문
   - 수학적 개념과 수식 ($LaTeX$ 형식)

2. **사용 데이터**
   - 데이터 파일명, 내용, 기간, 용도를 테이블로 정리
   - 데이터 선정 이유

3. **분석 1: [핵심개념1]**
   - 분석 목적
   - 방법론 설명 (수식 포함)
   - 결과 테이블
   - 해석 및 결론

4. **분석 2: [핵심개념2]**
   - (위와 동일한 형식)

5. **전략 백테스트**
   - 전략 원리 (Python 코드 블록)
   - 성과 지표 (APR, Sharpe Ratio, MDD 등)
   - 차트 (`figures/` 폴더 참조)
   - 차트 title/axis label은 영문 사용 권장, legend는 한/영 혼용 가능 (한국어 사용 시 `_setup_korean_font()` 필수)

6. **결론 및 권고사항**
   - 핵심 발견 요약 테이블
   - 트레이딩 권고사항
   - 주의사항 (bias, 거래비용 등)

### Step 6: 출력 디렉토리 생성

스크립트가 다음 디렉토리 구조를 사용하도록 하세요:

```
chapter_X_[챕터명]/src/
  ├── run_chapterX_analysis.py    # 생성할 스크립트
  └── reports/
      ├── chapterX_report.md      # 생성될 리포트
      └── figures/
          └── *.png               # 생성될 차트
```

### Step 7: 스크립트 실행 및 검증

스크립트 생성 후:

1. 가상환경 활성화: `source ${BOOK_ROOT}/.venv/bin/activate`
2. 스크립트 실행: `python run_chapterX_analysis.py`
3. 오류 발생 시 수정하고 재실행
4. 리포트와 차트가 정상 생성되었는지 확인

## 품질 체크리스트

스크립트 생성과 실행 후 다음을 검증하세요:

- [ ] 한글 문서의 핵심 개념이 리포트에 반영되었는가
- [ ] 수학 공식이 LaTeX 형식으로 올바르게 표현되었는가
- [ ] 데이터 설명과 선정 이유가 포함되었는가
- [ ] 각 분석 결과에 해석과 트레이딩 함의가 있는가
- [ ] 백테스트 결과에 주의사항(bias, 거래비용 등)이 명시되었는가
- [ ] 차트가 `reports/figures/`에 저장되는가
- [ ] pandas 2.0+ 호환성이 유지되는가
- [ ] 스크립트가 오류 없이 실행되는가
