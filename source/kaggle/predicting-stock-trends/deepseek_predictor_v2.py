#!/usr/bin/env python3
"""
====================================================================
Kaggle 주식 트렌드 예측 - DeepSeek-R1 기반 모델
====================================================================

목적 (Purpose):
    DeepSeek-R1 대형 언어 모델의 Chain-of-Thought 추론 능력을 활용하여
    주식 가격이 30 거래일 후 상승(1) 또는 하락(0)할지 예측

대회 정보 (Competition):
    - 이름: Predicting Stock Trends: Rise or Fall?
    - 데이터: 5,000개 종목의 OHLCV + 배당/분할 데이터
    - 목표: 30 거래일 후 종가 상승/하락 예측
    - 평가 지표: Accuracy (정확도)

모델 구조 (Model Architecture):
    - 알고리즘: DeepSeek-R1:14B (Ollama를 통한 로컬 추론)
    - 접근법: 하이브리드 (Traditional Feature Engineering + LLM Reasoning)
    - 특징: 16개 기술적 지표 → 자연어 요약 → LLM 추론
    - 출력: 구조화된 예측 (상승/하락 + 신뢰도 + 추론 과정)

핵심 장점 (Key Advantages):
    1. Chain-of-Thought 추론으로 복잡한 패턴 인식
    2. 설명 가능한 예측 (reasoning 제공)
    3. Few-shot learning으로 사례 기반 학습

작성자: ML4T Project
날짜: 2025
====================================================================
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
from tqdm import tqdm
import os
from typing import Dict, Tuple, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# OpenAI 클라이언트 (vLLM과 호환)
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    print("⚠️  Error: openai 패키지가 설치되지 않았습니다.")
    print("   실행: pip install openai")
    exit(1)

# baseline.py의 함수들을 재사용하기 위한 import
import sys
sys.path.insert(0, os.path.dirname(__file__))

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ====================================================================
# 설정 상수 (Configuration Constants)
# ====================================================================
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
RANDOM_STATE = 42

# vLLM 서버 설정
VLLM_API_BASE = "http://localhost:8000/v1"  # vLLM 서버 주소
VLLM_API_KEY = "EMPTY"  # vLLM은 기본적으로 인증 불필요

# DeepSeek-R1 모델 설정
DEEPSEEK_MODEL = './hf_models/DeepSeek-R1-Distill-Qwen-7B'  # vLLM에 로드된 모델 경로
DEEPSEEK_TEMPERATURE = 0.2  # 0.2로 증가 (더 다양한 응답 생성)
DEEPSEEK_MAX_TOKENS = 500  # 500으로 복원 (충분한 추론 길이)

# 배치 처리 설정 (속도 개선)
CONCURRENT_REQUESTS = 10  # vLLM은 더 많은 동시 처리 가능 (10개 동시 추론)
SAVE_REASONING = True  # 추론 과정 저장 여부

# ====================================================================
# 데이터 로딩 함수 (baseline.py 재사용)
# ====================================================================

def load_data():
    """CSV 파일에서 학습 및 테스트 데이터셋을 로드합니다."""
    print("📂 데이터 로딩 중...")

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    sample_submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")

    return train_df, test_df, sample_submission


def create_features(df):
    """OHLCV 데이터로부터 기술적 특징을 생성합니다 (baseline.py와 동일)."""
    features = pd.DataFrame()

    # 메타데이터
    features['ticker'] = df['Ticker']
    features['date'] = pd.to_datetime(df['Date'])

    # 가격 기반 특징
    features['returns'] = (df['Close'] - df['Open']) / df['Open']
    features['high_low_ratio'] = df['High'] / df['Low']
    features['close_open_ratio'] = df['Close'] / df['Open']

    # 거래량 특징
    features['volume'] = df['Volume']
    features['volume_log'] = np.log1p(df['Volume'])

    # 변동성 특징
    features['daily_range'] = (df['High'] - df['Low']) / df['Open']

    # 기업 활동 특징
    features['has_dividend'] = (df['Dividends'] > 0).astype(int)
    features['has_split'] = (df['Stock Splits'] > 0).astype(int)

    return features


# ====================================================================
# 특징 → 텍스트 변환기 (Feature to Text Converter)
# ====================================================================

class FeatureToTextConverter:
    """
    수치형 특징 벡터를 자연어 설명으로 변환하는 클래스.

    DeepSeek-R1이 이해하기 쉬운 형태로 기술적 지표를 서술합니다.
    """

    @staticmethod
    def convert(features: Dict[str, float], ticker: str) -> str:
        """
        특징 딕셔너리를 자연어 설명으로 변환합니다.

        Args:
            features: 16개 기술적 지표를 담은 딕셔너리
            ticker: 종목 코드 (예: ticker_1)

        Returns:
            자연어로 작성된 종목 분석 요약
        """
        # 안전한 값 추출 (NaN 처리)
        def safe_get(key, default=0):
            val = features.get(key, default)
            return default if pd.isna(val) else val

        # 주요 지표 추출
        returns = safe_get('returns', 0) * 100
        close_open = (safe_get('close_open_ratio', 1) - 1) * 100
        daily_range = safe_get('daily_range', 0) * 100
        volume_log = safe_get('volume_log', 0)

        ma_5 = safe_get('ma_5', 0)
        ma_20 = safe_get('ma_20', 0)
        close = safe_get('close', 0)

        price_to_ma5 = safe_get('price_to_ma5', 1)
        price_to_ma20 = safe_get('price_to_ma20', 1)

        volume_ma_5 = safe_get('volume_ma_5', 0)
        volume = safe_get('volume', 0)

        # 추세 판단
        trend_short = "상승" if price_to_ma5 > 1.02 else "하락" if price_to_ma5 < 0.98 else "횡보"
        trend_long = "상승" if price_to_ma20 > 1.02 else "하락" if price_to_ma20 < 0.98 else "횡보"

        # 거래량 판단
        volume_status = "평균 이상" if volume > volume_ma_5 * 1.2 else "평균 이하" if volume < volume_ma_5 * 0.8 else "평균 수준"

        # 변동성 판단
        volatility = "높음" if daily_range > 5 else "낮음" if daily_range < 2 else "보통"

        # 자연어 요약 생성
        summary = f"""종목: {ticker}

【가격 동향】
- 일중 수익률: {returns:.2f}%
- 종가 변화: 시가 대비 {close_open:+.2f}%
- 일중 변동폭: {daily_range:.2f}% ({volatility})

【이동평균 분석】
- 5일 이동평균: ${ma_5:.2f} (현재 가격: ${close:.2f})
  → 단기 추세: {trend_short} (가격/MA5 = {price_to_ma5:.3f})
- 20일 이동평균: ${ma_20:.2f}
  → 장기 추세: {trend_long} (가격/MA20 = {price_to_ma20:.3f})

【거래량 분석】
- 현재 거래량: {volume:,.0f}주 (로그 스케일: {volume_log:.2f})
- 5일 평균 거래량: {volume_ma_5:,.0f}주
- 거래 활동: {volume_status}

【기술적 신호】
- 단기/장기 추세 정렬: {'일치' if (price_to_ma5 > 1) == (price_to_ma20 > 1) else '불일치'}
- 골든크로스/데드크로스: {'골든크로스' if ma_5 > ma_20 else '데드크로스' if ma_5 < ma_20 else '중립'}
"""

        return summary.strip()


# ====================================================================
# DeepSeek-R1 주식 예측기 (Main Predictor Class)
# ====================================================================

class DeepSeekStockPredictor:
    """
    DeepSeek-R1 모델을 사용한 주식 트렌드 예측기.

    Chain-of-Thought 추론을 활용하여 기술적 지표를 분석하고
    30일 후 주가 변동을 예측합니다.
    """

    def __init__(self, model_name: str = DEEPSEEK_MODEL):
        """
        Args:
            model_name: vLLM 모델 이름 (기본: DeepSeek-R1-Distill-Qwen-7B)
        """
        self.model_name = model_name
        self.converter = FeatureToTextConverter()

        # OpenAI 동기/비동기 클라이언트 초기화 (vLLM 서버 연결)
        try:
            self.client = OpenAI(
                api_key=VLLM_API_KEY,
                base_url=VLLM_API_BASE
            )
            self.async_client = AsyncOpenAI(
                api_key=VLLM_API_KEY,
                base_url=VLLM_API_BASE
            )

            # vLLM 서버 연결 확인
            models = self.client.models.list()
            print(f"✅ vLLM 서버 연결 성공")
            print(f"   사용 가능한 모델: {[m.id for m in models.data]}")
        except Exception as e:
            print(f"❌ vLLM 연결 실패: {e}")
            print(f"   상세 오류: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            print(f"   vLLM 서버가 {VLLM_API_BASE}에서 실행 중인지 확인하세요")
            raise

        print(f"✅ DeepSeek-R1 예측기 초기화 완료 (모델: {model_name})")
        print(f"   동시 처리 요청 수: {CONCURRENT_REQUESTS}개 (예상 속도 향상: {CONCURRENT_REQUESTS}배)")

    def _create_system_prompt(self) -> str:
        """시스템 프롬프트를 생성합니다 (2025 Best Practices 적용)."""
        return """You are a professional quantitative analyst with 20 years of experience.

Based on the given technical indicators, predict whether the stock price will RISE (1) or FALL (0) after 30 trading days (~6 weeks).

【CRITICAL RULES】
1. You MUST choose either "상승" (RISE) or "하락" (FALL) - NO other options allowed
2. "횡보", "중립", "持平", "판단불가" or any neutral predictions are FORBIDDEN
3. If uncertain, follow the recent trend direction
4. Write your reasoning in <think> tags
5. Write your final answer in <answer> tags

【DECISION CRITERIA】
✅ BULLISH (상승) signals:
- Golden cross (5-day MA > 20-day MA)
- Price above both moving averages
- Volume increase with price rise
- Positive momentum

✅ BEARISH (하락) signals:
- Death cross (5-day MA < 20-day MA)
- Price below both moving averages
- Volume increase with price fall
- Negative momentum

【RESPONSE FORMAT】
<think>
- Current trend analysis
- Moving average cross check
- Volume pattern analysis
- Volatility assessment
- Final decision
</think>
<answer>
예측: [상승/하락] (MUST be one of these two)
신뢰도: [0-100]%
근거: [Key reasons in 1-2 sentences]
</answer>"""

    def _create_few_shot_examples(self) -> List[Dict[str, str]]:
        """Few-shot 학습을 위한 예제를 생성합니다 (10개: 5 상승, 5 하락)."""
        return [
            # 예시 1: 강한 상승 (골든크로스 + 거래량 증가)
            {
                "role": "user",
                "content": """종목: ticker_상승1
【가격 동향】일중 수익률: 2.5%, 종가 변화: +2.3%, 일중 변동폭: 3.1% (보통)
【이동평균 분석】5일MA: $120.50 (현재: $125.00) → 상승 (1.037), 20일MA: $115.00 → 상승 (1.087)
【거래량 분석】현재: 1,500,000주, 5일 평균: 1,000,000주 → 평균 이상
【기술적 신호】골든크로스, 추세 일치"""
            },
            {
                "role": "assistant",
                "content": """<think>
골든크로스 + 거래량 50% 증가 + 양봉 → 강한 매수세
</think>
<answer>
예측: 상승
신뢰도: 75%
근거: 골든크로스와 거래량 급증으로 상승 모멘텀 지속 예상
</answer>"""
            },

            # 예시 2: 강한 하락 (데드크로스 + 거래량 증가)
            {
                "role": "user",
                "content": """종목: ticker_하락1
【가격 동향】일중 수익률: -1.8%, 종가 변화: -2.0%, 일중 변동폭: 4.2% (높음)
【이동평균 분석】5일MA: $95.00 (현재: $92.00) → 하락 (0.968), 20일MA: $98.00 → 하락 (0.939)
【거래량 분석】현재: 2,000,000주, 5일 평균: 1,200,000주 → 평균 이상
【기술적 신호】데드크로스, 추세 일치"""
            },
            {
                "role": "assistant",
                "content": """<think>
데드크로스 + 거래량 증가 + 음봉 → 강한 매도세
</think>
<answer>
예측: 하락
신뢰도: 70%
근거: 데드크로스와 거래량 증가로 하락 추세 지속 예상
</answer>"""
            },

            # 예시 3: 상승 (가격이 이동평균 상회)
            {
                "role": "user",
                "content": """종목: ticker_상승2
【가격 동향】일중 수익률: 1.2%, 종가 변화: +1.5%, 일중 변동폭: 2.1% (낮음)
【이동평균 분석】5일MA: $48.50 (현재: $50.00) → 상승 (1.031), 20일MA: $47.00 → 상승 (1.064)
【거래량 분석】현재: 800,000주, 5일 평균: 750,000주 → 평균 수준
【기술적 신호】가격이 양쪽 MA 상회"""
            },
            {
                "role": "assistant",
                "content": """<think>
가격이 5일/20일MA 모두 상회 + 안정적 상승
</think>
<answer>
예측: 상승
신뢰도: 65%
근거: 이동평균 상회와 안정적인 상승세로 추세 지속 전망
</answer>"""
            },

            # 예시 4: 하락 (가격이 이동평균 하회)
            {
                "role": "user",
                "content": """종목: ticker_하락2
【가격 동향】일중 수익률: -0.8%, 종가 변화: -1.1%, 일중 변동폭: 2.5% (보통)
【이동평균 분석】5일MA: $78.00 (현재: $75.00) → 하락 (0.962), 20일MA: $80.00 → 하락 (0.938)
【거래량 분석】현재: 950,000주, 5일 평균: 1,000,000주 → 평균 이하
【기술적 신호】가격이 양쪽 MA 하회"""
            },
            {
                "role": "assistant",
                "content": """<think>
가격이 5일/20일MA 모두 하회 + 음봉
</think>
<answer>
예측: 하락
신뢰도: 62%
근거: 이동평균 하회로 약세 추세 지속 예상
</answer>"""
            },

            # 예시 5: 상승 (강한 모멘텀)
            {
                "role": "user",
                "content": """종목: ticker_상승3
【가격 동향】일중 수익률: 3.8%, 종가 변화: +3.5%, 일중 변동폭: 5.2% (높음)
【이동평균 분석】5일MA: $32.00 (현재: $35.00) → 상승 (1.094), 20일MA: $30.00 → 상승 (1.167)
【거래량 분석】현재: 2,500,000주, 5일 평균: 1,500,000주 → 평균 이상
【기술적 신호】강한 상승 모멘텀"""
            },
            {
                "role": "assistant",
                "content": """<think>
+3.8% 급등 + 거래량 67% 증가 → 돌파 시도
</think>
<answer>
예측: 상승
신뢰도: 80%
근거: 강한 모멘텀과 거래량 급증으로 상승 지속 가능성 높음
</answer>"""
            },

            # 예시 6: 하락 (급락)
            {
                "role": "user",
                "content": """종목: ticker_하락3
【가격 동향】일중 수익률: -2.9%, 종가 변화: -3.2%, 일중 변동폭: 5.8% (높음)
【이동평균 분석】5일MA: $65.00 (현재: $61.00) → 하락 (0.938), 20일MA: $68.00 → 하락 (0.897)
【거래량 분석】현재: 3,200,000주, 5일 평균: 1,800,000주 → 평균 이상
【기술적 신호】급락 + 고거래량"""
            },
            {
                "role": "assistant",
                "content": """<think>
-2.9% 급락 + 거래량 78% 증가 → 패닉 매도
</think>
<answer>
예측: 하락
신뢰도: 77%
근거: 급락과 거래량 폭증으로 추가 하락 압력 예상
</answer>"""
            },

            # 예시 7: 상승 (저거래량이지만 추세 명확)
            {
                "role": "user",
                "content": """종목: ticker_상승4
【가격 동향】일중 수익률: 0.9%, 종가 변화: +1.0%, 일중 변동폭: 1.5% (낮음)
【이동평균 분석】5일MA: $102.00 (현재: $104.00) → 상승 (1.020), 20일MA: $100.00 → 상승 (1.040)
【거래량 분석】현재: 400,000주, 5일 평균: 500,000주 → 평균 이하
【기술적 신호】안정적 상승세"""
            },
            {
                "role": "assistant",
                "content": """<think>
거래량 적지만 꾸준한 상승 + MA 상회 → 안정적
</think>
<answer>
예측: 상승
신뢰도: 60%
근거: 낮은 변동성 속 꾸준한 상승세로 안정적 추세 유지 전망
</answer>"""
            },

            # 예시 8: 하락 (저거래량 하락)
            {
                "role": "user",
                "content": """종목: ticker_하락4
【가격 동향】일중 수익률: -0.5%, 종가 변화: -0.7%, 일중 변동폭: 1.8% (낮음)
【이동평균 분석】5일MA: $55.00 (현재: $53.50) → 하락 (0.973), 20일MA: $56.50 → 하락 (0.947)
【거래량 분석】현재: 350,000주, 5일 평균: 450,000주 → 평균 이하
【기술적 신호】약한 하락세"""
            },
            {
                "role": "assistant",
                "content": """<think>
거래량 감소 + 지속적 하락 → 매수세 부족
</think>
<answer>
예측: 하락
신뢰도: 58%
근거: 매수세 부족과 이동평균 하회로 약세 지속 예상
</answer>"""
            },

            # 예시 9: 상승 (회복 신호)
            {
                "role": "user",
                "content": """종목: ticker_상승5
【가격 동향】일중 수익률: 1.8%, 종가 변화: +2.1%, 일중 변동폭: 3.5% (보통)
【이동평균 분석】5일MA: $88.00 (현재: $90.00) → 상승 (1.023), 20일MA: $85.00 → 상승 (1.059)
【거래량 분석】현재: 1,800,000주, 5일 평균: 1,300,000주 → 평균 이상
【기술적 신호】5일MA가 20일MA 상향 돌파 직전"""
            },
            {
                "role": "assistant",
                "content": """<think>
골든크로스 임박 + 거래량 증가 → 추세 전환
</think>
<answer>
예측: 상승
신뢰도: 72%
근거: 골든크로스 임박과 거래량 증가로 상승 전환 기대
</answer>"""
            },

            # 예시 10: 하락 (추세 전환 신호)
            {
                "role": "user",
                "content": """종목: ticker_하락5
【가격 동향】일중 수익률: -1.3%, 종가 변화: -1.5%, 일중 변동폭: 3.2% (보통)
【이동평균 분석】5일MA: $42.00 (현재: $41.00) → 하락 (0.976), 20일MA: $44.00 → 하락 (0.932)
【거래량 분석】현재: 1,600,000주, 5일 평균: 1,100,000주 → 평균 이상
【기술적 신호】5일MA가 20일MA 하향 돌파"""
            },
            {
                "role": "assistant",
                "content": """<think>
데드크로스 발생 + 거래량 증가 → 하락 전환
</think>
<answer>
예측: 하락
신뢰도: 68%
근거: 데드크로스 발생과 거래량 증가로 하락 추세 전환 예상
</answer>"""
            }
        ]

    def predict(self, features: Dict[str, float], ticker: str) -> Tuple[int, float, str]:
        """
        단일 종목에 대한 예측을 수행합니다.

        Args:
            features: 16개 기술적 지표 딕셔너리
            ticker: 종목 코드

        Returns:
            (prediction, confidence, reasoning)
            - prediction: 0 (하락) or 1 (상승)
            - confidence: 0.0 ~ 1.0
            - reasoning: DeepSeek의 추론 과정
        """
        # 1. 특징을 자연어로 변환
        ticker_summary = self.converter.convert(features, ticker)

        # 2. 메시지 구성 (Few-shot + User query)
        messages = [
            {"role": "system", "content": self._create_system_prompt()}
        ]

        # Few-shot 예제 추가
        messages.extend(self._create_few_shot_examples())

        # 실제 질문 추가
        messages.append({
            "role": "user",
            "content": ticker_summary
        })

        # 3. vLLM을 통한 DeepSeek-R1 추론 호출
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=DEEPSEEK_TEMPERATURE,
                max_tokens=DEEPSEEK_MAX_TOKENS,
            )

            response_text = response.choices[0].message.content

        except Exception as e:
            print(f"❌ 예측 실패 ({ticker}): {e}")
            # 실패 시 중립 예측 반환
            return 1, 0.5, f"Error: {str(e)}"

        # 4. 응답 파싱
        prediction, confidence, reasoning = self._parse_response(response_text)

        return prediction, confidence, reasoning

    def _parse_response(self, response_text: str) -> Tuple[int, float, str]:
        """
        DeepSeek-R1의 응답을 파싱하여 구조화된 예측을 추출합니다.

        Args:
            response_text: DeepSeek-R1의 원본 응답

        Returns:
            (prediction, confidence, reasoning)
        """
        # <think> 태그에서 추론 과정 추출
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)
        reasoning = think_match.group(1).strip() if think_match else ""

        # <answer> 태그에서 최종 답변 추출
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else response_text

        # 예측 추출 (상승/하락 키워드) - 강화된 파싱
        prediction = None

        # 하락 키워드 체크 (우선순위 높음)
        if re.search(r'(예측|prediction)[\s:]*하락|하락\s*예측', answer, re.IGNORECASE):
            prediction = 0
        elif re.search(r'하락|fall|down|bearish|decline|decrease', answer, re.IGNORECASE):
            prediction = 0

        # 상승 키워드 체크
        elif re.search(r'(예측|prediction)[\s:]*상승|상승\s*예측', answer, re.IGNORECASE):
            prediction = 1
        elif re.search(r'상승|rise|up|bullish|increase|rally', answer, re.IGNORECASE):
            prediction = 1

        # 중립/횡보 키워드 처리 (금지되었지만 혹시 나올 경우 대비)
        if prediction is None or re.search(r'횡보|중립|持平|판단불가|neutral|sideways|flat|hold', answer, re.IGNORECASE):
            # 최근 추론 과정에서 힌트 찾기
            if re.search(r'상승|rise|up|bullish|매수|골든크로스', response_text, re.IGNORECASE):
                prediction = 1
            elif re.search(r'하락|fall|down|bearish|매도|데드크로스', response_text, re.IGNORECASE):
                prediction = 0
            else:
                # 완전히 불확실한 경우 기본값
                prediction = 1  # 기본값: 상승

        # 신뢰도 추출 (0-100% → 0.0-1.0)
        confidence = 0.5  # 기본값
        conf_match = re.search(r'신뢰도:?\s*(\d+)\s*%?', answer, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1)) / 100.0
        else:
            # confidence 영어 패턴도 검색
            conf_match = re.search(r'confidence:?\s*(\d+)\s*%?', answer, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1)) / 100.0

        # 추론 과정 포함한 전체 reasoning
        full_reasoning = f"【추론 과정】\n{reasoning}\n\n【최종 답변】\n{answer}"

        return prediction, confidence, full_reasoning

    async def predict_async(self, features: Dict[str, float], ticker: str) -> Tuple[int, float, str]:
        """
        비동기 방식으로 단일 종목 예측을 수행합니다.

        Args:
            features: 16개 기술적 지표 딕셔너리
            ticker: 종목 코드

        Returns:
            (prediction, confidence, reasoning)
        """
        # 1. 특징을 자연어로 변환
        ticker_summary = self.converter.convert(features, ticker)

        # 2. 메시지 구성
        messages = [
            {"role": "system", "content": self._create_system_prompt()}
        ]
        messages.extend(self._create_few_shot_examples())
        messages.append({
            "role": "user",
            "content": ticker_summary
        })

        # 3. 비동기 vLLM을 통한 DeepSeek-R1 추론 호출
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=DEEPSEEK_TEMPERATURE,
                max_tokens=DEEPSEEK_MAX_TOKENS,
            )

            response_text = response.choices[0].message.content

            # DEBUG: 첫 번째 응답만 로깅 (디버깅용)
            if not hasattr(self, '_debug_logged'):
                print(f"\n🔍 DEBUG - Raw Response for {ticker}:")
                print(f"Response length: {len(response_text)} chars")
                print(f"Finish reason: {response.choices[0].finish_reason}")
                print(f"First 500 chars: {response_text[:500]}")
                self._debug_logged = True

        except Exception as e:
            print(f"❌ Async prediction error for {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return 1, 0.5, f"Error: {str(e)}"

        # 4. 응답 파싱
        prediction, confidence, reasoning = self._parse_response(response_text)
        return prediction, confidence, reasoning

    def batch_predict(self, features_list: List[Dict], tickers: List[str]) -> pd.DataFrame:
        """
        여러 종목에 대한 배치 예측을 수행합니다 (비동기 병렬 처리).

        Args:
            features_list: 특징 딕셔너리 리스트
            tickers: 종목 코드 리스트

        Returns:
            예측 결과를 담은 DataFrame (ticker, prediction, confidence, reasoning)
        """
        print(f"\n🤖 DeepSeek-R1으로 {len(tickers)}개 종목 예측 중...")
        print(f"   병렬 처리: {CONCURRENT_REQUESTS}개 동시 실행")

        # asyncio를 사용하여 배치 처리
        results = asyncio.run(self._batch_predict_async(features_list, tickers))

        return pd.DataFrame(results)

    async def _batch_predict_async(self, features_list: List[Dict], tickers: List[str]) -> List[Dict]:
        """
        비동기 배치 예측의 내부 구현.

        동시에 CONCURRENT_REQUESTS개씩 처리하여 속도를 크게 향상시킵니다.
        """
        results = []
        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

        async def predict_with_semaphore(features, ticker):
            """세마포어를 사용한 동시 실행 제한"""
            try:
                async with semaphore:
                    prediction, confidence, reasoning = await self.predict_async(features, ticker)
                    return {
                        'ticker': ticker,
                        'prediction': prediction,
                        'confidence': confidence,
                        'reasoning': reasoning
                    }
            except Exception as e:
                print(f"\n❌ {ticker} 예측 실패: {e}")
                import traceback
                traceback.print_exc()
                # 실패 시에도 기본값 반환
                return {
                    'ticker': ticker,
                    'prediction': 1,
                    'confidence': 0.5,
                    'reasoning': f"Error: {str(e)}"
                }

        # 모든 예측 작업을 태스크로 생성
        tasks = [
            predict_with_semaphore(features, ticker)
            for features, ticker in zip(features_list, tickers)
        ]

        # tqdm을 사용한 프로그레스바와 함께 실행
        results = []
        with tqdm(total=len(tasks), desc="예측 진행") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)

        print(f"\n✅ 총 {len(results)}개 예측 완료")
        return results


# ====================================================================
# 데이터 준비 함수
# ====================================================================

def prepare_ticker_features(train_df: pd.DataFrame, ticker: str) -> Dict[str, float]:
    """
    특정 종목의 최근 데이터로부터 특징을 추출합니다.

    Args:
        train_df: 전체 학습 데이터
        ticker: 종목 코드

    Returns:
        16개 특징을 담은 딕셔너리
    """
    # 해당 종목 데이터 필터링
    ticker_data = train_df[train_df['Ticker'] == ticker].sort_values('Date')

    if len(ticker_data) == 0:
        # 데이터 없는 경우 빈 딕셔너리 반환
        return {}

    # 기본 특징 생성
    features_df = create_features(ticker_data)

    # 이동평균 추가
    for window in [5, 10, 20]:
        features_df[f'ma_{window}'] = ticker_data['Close'].rolling(
            window=window, min_periods=1
        ).mean()
        features_df[f'volume_ma_{window}'] = ticker_data['Volume'].rolling(
            window=window, min_periods=1
        ).mean()

    # 가격/이동평균 비율
    features_df['price_to_ma5'] = ticker_data['Close'] / features_df['ma_5']
    features_df['price_to_ma20'] = ticker_data['Close'] / features_df['ma_20']

    # Close 가격 추가 (텍스트 변환용)
    features_df['close'] = ticker_data['Close'].values

    # 가장 최근 행 추출
    latest_features = features_df.iloc[-1].to_dict()

    return latest_features


def save_features_cache(features_list: List[Dict], tickers: List[str], cache_path: str):
    """
    특징 추출 결과를 parquet 파일로 저장합니다.

    Args:
        features_list: 특징 딕셔너리 리스트
        tickers: 종목 코드 리스트
        cache_path: 저장할 파일 경로
    """
    # DEBUG: 데이터 확인
    print(f"   저장할 데이터: {len(features_list)}개 딕셔너리")
    if len(features_list) > 0:
        print(f"   첫 번째 딕셔너리 키 개수: {len(features_list[0])}개")
        print(f"   샘플 키: {list(features_list[0].keys())[:5]}")

    # 딕셔너리 리스트를 DataFrame으로 변환
    features_df = pd.DataFrame(features_list)
    print(f"   DataFrame shape: {features_df.shape}")
    print(f"   DataFrame 컬럼: {features_df.columns.tolist()[:10]}")

    features_df['ticker'] = tickers

    # parquet으로 저장
    features_df.to_parquet(cache_path, index=False, compression='snappy')
    print(f"✅ 특징 캐시 저장 완료: {cache_path}")
    print(f"   크기: {os.path.getsize(cache_path) / 1024 / 1024:.2f} MB")


def load_features_cache(cache_path: str) -> Tuple[List[Dict], List[str]]:
    """
    저장된 특징 추출 결과를 로드합니다.

    Args:
        cache_path: 캐시 파일 경로

    Returns:
        (features_list, tickers)
    """
    features_df = pd.read_parquet(cache_path)

    # ticker 컬럼 분리
    tickers = features_df['ticker'].tolist()
    features_df = features_df.drop(columns=['ticker'])

    # DataFrame을 딕셔너리 리스트로 변환
    features_list = features_df.to_dict('records')

    print(f"✅ 특징 캐시 로드 완료: {cache_path}")
    print(f"   종목 수: {len(tickers)}개")

    return features_list, tickers


def prepare_test_features(test_df: pd.DataFrame, train_df: pd.DataFrame, use_cache: bool = True) -> Tuple[List[Dict], List[str]]:
    """
    테스트 세트의 모든 종목에 대한 특징을 준비합니다.

    Args:
        test_df: 테스트 데이터 (ID, Date 포함)
        train_df: 학습 데이터 (과거 OHLCV 데이터)
        use_cache: 캐시 사용 여부 (기본: True)

    Returns:
        (features_list, tickers)
        - features_list: 특징 딕셔너리 리스트
        - tickers: 종목 코드 리스트
    """
    # 캐시 파일 경로
    cache_path = os.path.join(OUTPUT_DIR, 'test_features_cache.parquet')

    # 캐시 사용 및 캐시 파일이 존재하는 경우
    if use_cache and os.path.exists(cache_path):
        print("\n📦 캐시된 특징 데이터를 로드합니다...")
        try:
            return load_features_cache(cache_path)
        except Exception as e:
            print(f"⚠️  캐시 로드 실패: {e}")
            print("   특징을 새로 추출합니다...")

    # 캐시가 없거나 사용하지 않는 경우 새로 추출
    print("\n📊 테스트 세트 특징 준비 중...")

    features_list = []
    tickers = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="특징 추출"):
        # ID가 이미 ticker임 (예: ticker_1, ticker_10)
        ticker = row['ID']
        tickers.append(ticker)

        # 특징 추출
        features = prepare_ticker_features(train_df, ticker)
        features_list.append(features)

    # 캐시 저장
    if use_cache:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_features_cache(features_list, tickers, cache_path)

    return features_list, tickers


# ====================================================================
# 메인 실행 파이프라인
# ====================================================================

def main():
    """
    DeepSeek-R1 기반 주식 예측 파이프라인의 메인 함수.

    실행 흐름:
        1. 데이터 로드
        2. 테스트 세트 특징 준비
        3. DeepSeek-R1 예측 수행
        4. 제출 파일 생성
        5. 추론 과정 저장 (선택)
    """
    print("="*70)
    print("🚀 DeepSeek-R1 주식 트렌드 예측 시작")
    print("="*70)

    start_time = datetime.now()

    # ====================================================================
    # 1. 데이터 로드
    # ====================================================================
    train_df, test_df, sample_submission = load_data()

    # ====================================================================
    # 2. 출력 디렉토리 생성
    # ====================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ====================================================================
    # 3. DeepSeek 예측기 초기화
    # ====================================================================
    predictor = DeepSeekStockPredictor(model_name=DEEPSEEK_MODEL)

    # ====================================================================
    # 4. 테스트 세트 특징 준비
    # ====================================================================
    features_list, tickers = prepare_test_features(test_df, train_df)

    # ====================================================================
    # 5. 배치 예측 수행
    # ====================================================================
    predictions_df = predictor.batch_predict(features_list, tickers)

    # 예측 결과 확인
    if predictions_df.empty or len(predictions_df) == 0:
        print("❌ 예측 결과가 비어있습니다!")
        return None, None

    print(f"\n📊 예측 완료: {len(predictions_df)}개")
    print(f"   상승 예측: {(predictions_df['prediction'] == 1).sum()}개")
    print(f"   하락 예측: {(predictions_df['prediction'] == 0).sum()}개")
    print(f"   평균 신뢰도: {predictions_df['confidence'].mean():.3f}")

    # ====================================================================
    # 6. 제출 파일 생성
    # ====================================================================
    submission = sample_submission.copy()
    submission['Pred'] = predictions_df['prediction'].values

    submission_path = os.path.join(OUTPUT_DIR, 'submission_deepseek.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\n✅ 제출 파일 저장: {submission_path}")

    # ====================================================================
    # 7. 추론 과정 저장 (선택)
    # ====================================================================
    if SAVE_REASONING:
        reasoning_path = os.path.join(OUTPUT_DIR, 'deepseek_reasoning.json')
        reasoning_data = predictions_df[['ticker', 'prediction', 'confidence', 'reasoning']].to_dict('records')

        with open(reasoning_path, 'w', encoding='utf-8') as f:
            json.dump(reasoning_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 추론 과정 저장: {reasoning_path}")

    # ====================================================================
    # 8. 통계 출력
    # ====================================================================
    print("\n" + "="*70)
    print("📊 예측 결과 통계")
    print("="*70)

    total = len(submission)
    rises = (submission['Pred'] == 1).sum()
    falls = (submission['Pred'] == 0).sum()

    print(f"총 예측 수: {total}")
    print(f"상승 예측 (1): {rises}개 ({rises/total:.1%})")
    print(f"하락 예측 (0): {falls}개 ({falls/total:.1%})")

    avg_confidence = predictions_df['confidence'].mean()
    print(f"평균 신뢰도: {avg_confidence:.1%}")

    # 실행 시간
    elapsed = datetime.now() - start_time
    print(f"\n⏱️  총 실행 시간: {elapsed}")
    print(f"   평균 예측 시간: {elapsed.total_seconds() / total:.2f}초/종목")

    # ====================================================================
    # 9. 경고 및 다음 단계 안내
    # ====================================================================
    print("\n" + "="*70)
    print("✅ DeepSeek-R1 예측 완료!")
    print("="*70)

    if rises / total < 0.1 or rises / total > 0.9:
        print("\n⚠️  경고: 예측이 매우 불균형합니다!")
        print("   - 프롬프트 튜닝이 필요할 수 있습니다")
        print("   - Few-shot 예제를 조정하세요")

    print("\n📌 다음 단계:")
    print(f"   1. 제출: {submission_path}")
    print(f"   2. 추론 검토: {reasoning_path if SAVE_REASONING else 'N/A'}")
    print("   3. baseline.py와 성능 비교")
    print("   4. 프롬프트 최적화 (temperature, few-shot 조정)")

    return predictions_df, submission


# ====================================================================
# 스크립트 진입점
# ====================================================================

if __name__ == "__main__":
    """
    스크립트 직접 실행 시 메인 파이프라인을 실행합니다.

    사용법:
        python deepseek_predictor.py

    사전 요구사항:
        1. Ollama 설치 및 실행 (ollama serve)
        2. DeepSeek-R1 모델 설치 (ollama pull deepseek-r1:14b)
        3. 데이터 파일 존재 (data/train.csv, data/test.csv)

    예상 실행 시간:
        - 특징 추출: ~5분 (5,000 종목)
        - DeepSeek 예측: ~40-60분 (0.5초/종목)
        - 총: ~45-65분
    """
    try:
        predictions_df, submission = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
