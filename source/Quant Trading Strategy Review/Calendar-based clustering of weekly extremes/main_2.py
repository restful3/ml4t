"""
논문 기반 요일 의존적 MSGARCH 구현
Paper: Calendar-based clustering of weekly extremes
"""
from collections import defaultdict
from datetime import timedelta
import math
from typing import Dict, Optional, Tuple
import numpy as np
from QuantConnect import DataNormalizationMode, Resolution
from QuantConnect.Algorithm import QCAlgorithm
from QuantConnect.Indicators import MovingAverageType

class WeeklyExtremeTracker:
    def __init__(self, expected_days: int = 5) -> None:
        self.expected_days = expected_days
        self.current_week: Optional[Tuple[int, int]] = None
        self.daily_records = []
        self.high_counts = defaultdict(int)
        self.low_counts = defaultdict(int)
        self.sample_weeks = 0
        self.skipped_weeks = 0
        self.week_history = []
        self.returns_history = []

    def add_bar(self, bar_time, high, low, close, prev_close) -> Optional[dict]:
        week_key = bar_time.isocalendar()[:2]
        info = None
        if self.current_week is None:
            self.current_week = week_key
        elif week_key != self.current_week:
            info = self._finalize_week()
            self.current_week = week_key
        self.daily_records.append((bar_time, high, low))
        if prev_close > 0:
            ret = math.log(close / prev_close)
            weekday = bar_time.weekday()
            self.returns_history.append((bar_time, ret, weekday))
        return info

    def finalize(self) -> Optional[dict]:
        return self._finalize_week()

    def _finalize_week(self) -> Optional[dict]:
        if not self.daily_records:
            self.current_week = None
            return None
        if len(self.daily_records) != self.expected_days:
            self.skipped_weeks += 1
            self.daily_records = []
            self.current_week = None
            return None
        self.daily_records.sort(key=lambda item: item[0])
        high_record = max(self.daily_records, key=lambda item: item[1])
        low_record = min(self.daily_records, key=lambda item: item[2])
        high_day = high_record[0].weekday()
        low_day = low_record[0].weekday()
        self.high_counts[high_day] += 1
        self.low_counts[low_day] += 1
        self.sample_weeks += 1
        info = {"week_key": self.current_week, "high_day": high_day, "low_day": low_day,
                "start": self.daily_records[0][0], "end": self.daily_records[-1][0]}
        self.week_history.append((self.current_week, high_day, low_day))
        self.daily_records = []
        self.current_week = None
        return info

    def get_distribution(self) -> dict:
        total = self.sample_weeks
        if total == 0:
            return {"high_probs": np.ones(5) / 5, "low_probs": np.ones(5) / 5}
        high_probs = np.array([self.high_counts.get(d, 0) for d in range(5)]) / total
        low_probs = np.array([self.low_counts.get(d, 0) for d in range(5)]) / total
        return {"high_probs": high_probs, "low_probs": low_probs}

    def get_returns_and_weekdays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.returns_history:
            return np.array([]), np.array([])
        returns = np.array([r[1] for r in self.returns_history])
        weekdays = np.array([r[2] for r in self.returns_history], dtype=int)
        return returns, weekdays

class DayDependentMSGARCH:
    def __init__(self, K: int = 3, max_iter: int = 50, tol: float = 1e-4) -> None:
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.transition_probs = np.ones((5, K, K)) / K
        self.mu = np.zeros((K, 5))
        self.alpha = np.full((K, 5), 0.01)
        self.beta = np.full((K, 5), 0.1)
        self.gamma = np.full((K, 5), 0.8)
        self.pi_0 = np.ones(K) / K
        self.is_fitted = False
        self.log_likelihood_history = []

    def fit(self, returns: np.ndarray, weekdays: np.ndarray) -> None:
        if len(returns) < 100:
            raise ValueError("학습 데이터 부족")
        T = len(returns)
        self._initialize_parameters(returns, weekdays)
        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            gamma, xi = self._e_step(returns, weekdays)
            self._m_step(returns, weekdays, gamma, xi)
            log_likelihood = self._compute_log_likelihood(returns, weekdays)
            self.log_likelihood_history.append(log_likelihood)
            if abs(log_likelihood - prev_ll) < self.tol:
                break
            prev_ll = log_likelihood
        self.is_fitted = True

    def _initialize_parameters(self, returns: np.ndarray, weekdays: np.ndarray) -> None:
        volatility = np.abs(returns)
        quantiles = np.percentile(volatility, np.linspace(0, 100, self.K + 1))
        for i in range(self.K):
            mask = (volatility >= quantiles[i]) & (volatility < quantiles[i + 1])
            if mask.sum() == 0:
                continue
            for d in range(5):
                day_mask = (weekdays == d) & mask
                if day_mask.sum() > 0:
                    self.mu[i, d] = returns[day_mask].mean()

    def _e_step(self, returns: np.ndarray, weekdays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(returns)
        K = self.K
        alpha = np.zeros((T, K))
        scaling_factors = np.zeros(T)
        for i in range(K):
            alpha[0, i] = self.pi_0[i] * self._emission_prob(returns[0], weekdays[0], i, 0)
        scaling_factors[0] = alpha[0].sum()
        if scaling_factors[0] > 0:
            alpha[0] /= scaling_factors[0]
        for t in range(1, T):
            d = weekdays[t]
            for j in range(K):
                alpha[t, j] = sum(alpha[t - 1, i] * self.transition_probs[d, i, j]
                                  for i in range(K)) * self._emission_prob(returns[t], d, j, t)
            scaling_factors[t] = alpha[t].sum()
            if scaling_factors[t] > 0:
                alpha[t] /= scaling_factors[t]
        beta = np.zeros((T, K))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            d_next = weekdays[t + 1]
            for i in range(K):
                beta[t, i] = sum(self.transition_probs[d_next, i, j] *
                                 self._emission_prob(returns[t + 1], d_next, j, t + 1) *
                                 beta[t + 1, j] for j in range(K))
            if scaling_factors[t + 1] > 0:
                beta[t] /= scaling_factors[t + 1]
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma = np.divide(gamma, gamma_sum, where=gamma_sum > 0)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            d_next = weekdays[t + 1]
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = (alpha[t, i] * self.transition_probs[d_next, i, j] *
                                   self._emission_prob(returns[t + 1], d_next, j, t + 1) *
                                   beta[t + 1, j])
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum
        return gamma, xi

    def _emission_prob(self, ret: float, weekday: int, state: int, t: int) -> float:
        mean = self.mu[state, weekday]
        denom = 1.0 - self.beta[state, weekday] - self.gamma[state, weekday]
        if denom <= 0:
            denom = 0.01
        variance = self.alpha[state, weekday] / denom
        variance = max(variance, 1e-6)
        return np.exp(-0.5 * ((ret - mean) ** 2) / variance) / np.sqrt(2 * np.pi * variance)

    def _m_step(self, returns: np.ndarray, weekdays: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        T = len(returns)
        K = self.K
        self.pi_0 = gamma[0]
        for d in range(5):
            for i in range(K):
                for j in range(K):
                    numerator = sum(xi[t, i, j] for t in range(T - 1) if weekdays[t + 1] == d)
                    denominator = sum(gamma[t, i] for t in range(T - 1) if weekdays[t + 1] == d)
                    if denominator > 0:
                        self.transition_probs[d, i, j] = numerator / denominator
                    else:
                        self.transition_probs[d, i, j] = 1.0 / K
            for i in range(K):
                row_sum = self.transition_probs[d, i].sum()
                if row_sum > 0:
                    self.transition_probs[d, i] /= row_sum
        for i in range(K):
            for d in range(5):
                mask = (weekdays == d)
                weighted_sum = (gamma[mask, i] * returns[mask]).sum()
                weight_total = gamma[mask, i].sum()
                if weight_total > 0:
                    self.mu[i, d] = weighted_sum / weight_total
        for i in range(K):
            for d in range(5):
                mask = (weekdays == d)
                weighted_var = (gamma[mask, i] * (returns[mask] - self.mu[i, d]) ** 2).sum()
                weight_total = gamma[mask, i].sum()
                if weight_total > 0:
                    variance = weighted_var / weight_total
                    self.alpha[i, d] = max(0.01, variance * 0.1)
                    self.beta[i, d] = min(0.15, variance * 0.5)
                    self.gamma[i, d] = min(0.84, 1.0 - self.beta[i, d] - 0.01)

    def _compute_log_likelihood(self, returns: np.ndarray, weekdays: np.ndarray) -> float:
        T = len(returns)
        K = self.K
        alpha = np.zeros((T, K))
        log_likelihood = 0.0
        for i in range(K):
            alpha[0, i] = self.pi_0[i] * self._emission_prob(returns[0], weekdays[0], i, 0)
        scaling = alpha[0].sum()
        if scaling > 0:
            log_likelihood += np.log(scaling)
            alpha[0] /= scaling
        for t in range(1, T):
            d = weekdays[t]
            for j in range(K):
                alpha[t, j] = sum(alpha[t - 1, i] * self.transition_probs[d, i, j]
                                  for i in range(K)) * self._emission_prob(returns[t], d, j, t)
            scaling = alpha[t].sum()
            if scaling > 0:
                log_likelihood += np.log(scaling)
                alpha[t] /= scaling
        return log_likelihood

    def simulate_weekly_paths(self, n_simulations: int = 2000, n_weeks: int = 500, initial_price: float = 100.0) -> dict:
        if not self.is_fitted:
            raise RuntimeError("모형 미학습")
        high_counts = np.zeros(5, dtype=int)
        low_counts = np.zeros(5, dtype=int)
        for sim in range(n_simulations):
            price = initial_price
            for week in range(n_weeks):
                weekly_prices = []
                if week == 0:
                    state = np.random.choice(self.K, p=self.pi_0)
                else:
                    state = np.random.choice(self.K, p=self.transition_probs[4, state])
                for weekday in range(5):
                    if weekday > 0:
                        state = np.random.choice(self.K, p=self.transition_probs[weekday, state])
                    mean = self.mu[state, weekday]
                    denom = 1.0 - self.beta[state, weekday] - self.gamma[state, weekday]
                    denom = max(denom, 0.01)
                    variance = self.alpha[state, weekday] / denom
                    variance = max(variance, 1e-6)
                    ret = np.random.normal(mean, np.sqrt(variance))
                    price *= np.exp(ret)
                    weekly_prices.append((weekday, price))
                high_day = max(weekly_prices, key=lambda x: x[1])[0]
                low_day = min(weekly_prices, key=lambda x: x[1])[0]
                high_counts[high_day] += 1
                low_counts[low_day] += 1
        total_weeks = n_simulations * n_weeks
        return {"high_probs": high_counts / total_weeks, "low_probs": low_counts / total_weeks}

class ModelValidator:
    @staticmethod
    def kl_divergence(observed: np.ndarray, expected: np.ndarray) -> float:
        kl = 0.0
        for o, e in zip(observed, expected):
            if o > 0 and e > 0:
                kl += o * np.log(o / e)
        return kl

    @staticmethod
    def g_test(observed_counts: np.ndarray, total: int) -> Tuple[float, Optional[float]]:
        if total == 0:
            return 0.0, None
        expected = total / len(observed_counts)
        if expected == 0:
            return 0.0, None
        g_stat = 0.0
        for observed in observed_counts:
            if observed > 0:
                g_stat += observed * np.log(observed / expected)
        g_stat *= 2.0
        try:
            from scipy.stats import chi2
            p_value = float(1.0 - chi2.cdf(g_stat, df=len(observed_counts) - 1))
        except:
            p_value = None
        return g_stat, p_value

    @staticmethod
    def validate_in_sample(observed_dist: dict, simulated_dist: dict) -> dict:
        obs_high = observed_dist["high_probs"]
        obs_low = observed_dist["low_probs"]
        sim_high = simulated_dist["high_probs"]
        sim_low = simulated_dist["low_probs"]
        kl_high = ModelValidator.kl_divergence(obs_high, sim_high)
        kl_low = ModelValidator.kl_divergence(obs_low, sim_low)
        scale = 1000
        obs_high_counts = (obs_high * scale).astype(int)
        obs_low_counts = (obs_low * scale).astype(int)
        g_high, p_high = ModelValidator.g_test(obs_high_counts, scale)
        g_low, p_low = ModelValidator.g_test(obs_low_counts, scale)
        return {"KL_high": kl_high, "KL_low": kl_low, "G_high": g_high, "G_low": g_low,
                "p_high": p_high, "p_low": p_low}

    @staticmethod
    def validate_out_of_sample(returns: np.ndarray, weekdays: np.ndarray, K: int, split_ratio: float = 0.8) -> dict:
        T = len(returns)
        split_idx = int(T * split_ratio)
        train_returns = returns[:split_idx]
        train_weekdays = weekdays[:split_idx]
        test_returns = returns[split_idx:]
        test_weekdays = weekdays[split_idx:]
        model = DayDependentMSGARCH(K=K, max_iter=30)
        model.fit(train_returns, train_weekdays)
        test_high_counts = np.zeros(5, dtype=int)
        test_low_counts = np.zeros(5, dtype=int)
        test_weeks = 0
        for i in range(0, len(test_returns) - 4, 5):
            week_returns = test_returns[i:i+5]
            week_weekdays = test_weekdays[i:i+5]
            if len(week_returns) == 5 and len(set(week_weekdays)) == 5:
                prices = 100 * np.exp(np.cumsum(week_returns))
                high_day = np.argmax(prices)
                low_day = np.argmin(prices)
                test_high_counts[high_day] += 1
                test_low_counts[low_day] += 1
                test_weeks += 1
        if test_weeks == 0:
            return {"error": "테스트 데이터 부족"}
        obs_high = test_high_counts / test_weeks
        obs_low = test_low_counts / test_weeks
        sim_dist = model.simulate_weekly_paths(n_simulations=1000, n_weeks=test_weeks)
        kl_high = ModelValidator.kl_divergence(obs_high, sim_dist["high_probs"])
        kl_low = ModelValidator.kl_divergence(obs_low, sim_dist["low_probs"])
        g_high, p_high = ModelValidator.g_test(test_high_counts, test_weeks)
        g_low, p_low = ModelValidator.g_test(test_low_counts, test_weeks)
        return {"KL_high": kl_high, "KL_low": kl_low, "G_high": g_high, "G_low": g_low,
                "p_high": p_high, "p_low": p_low, "test_weeks": test_weeks}

class PaperBasedStrategy(QCAlgorithm):
    def Initialize(self) -> None:
        self.SetStartDate(2005, 1, 3)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100_000)
        self.SetWarmUp(timedelta(days=100))
        self.SetSecurityInitializer(lambda security: security.SetDataNormalizationMode(DataNormalizationMode.Raw))
        equity = self.AddEquity("SPY", Resolution.Daily)
        self.symbol = equity.Symbol
        self.atr = self.ATR(self.symbol, 10, MovingAverageType.Simple, Resolution.Daily)
        self.tracker = WeeklyExtremeTracker(expected_days=5)
        self.prev_close = None
        self.training_phase = True
        self.validation_phase = False
        self.trading_phase = False
        self.min_training_weeks = 50
        self.msgarch_models = {}
        self.best_K = None
        self.best_model = None
        self.current_plan = None
        self.week_state = {"week_key": None, "entered": False}
        self.day_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        self.Schedule.On(self.DateRules.WeekEnd(self.symbol), self.TimeRules.AfterMarketClose(self.symbol, 1),
                         self._weekly_update)

    def OnData(self, slice) -> None:
        if self.symbol not in slice.Bars:
            return
        bar = slice.Bars[self.symbol]
        if self.prev_close is not None:
            self.tracker.add_bar(bar.EndTime, bar.High, bar.Low, bar.Close, self.prev_close)
        self.prev_close = bar.Close
        if self.trading_phase and not self.IsWarmingUp:
            self._handle_trading()

    def _weekly_update(self) -> None:
        if self.IsWarmingUp:
            return
        if self.training_phase:
            if self.tracker.sample_weeks % 10 == 0:
                self.Log(f"[DATA] {self.tracker.sample_weeks}/{self.min_training_weeks} weeks")
            if self.tracker.sample_weeks >= self.min_training_weeks:
                self._run_training()
                self.training_phase = False
                self.validation_phase = True
                self.Log(f"[TRAINING DONE] {self.tracker.sample_weeks} weeks")
        elif self.validation_phase:
            self._run_validation()
            self.validation_phase = False
            if self.best_model is not None:
                self.trading_phase = True
                self.Log("[VALIDATION PASSED]")
            else:
                self.Log("[VALIDATION FAILED]")
        elif self.trading_phase:
            self._update_trading_plan()

    def _run_training(self) -> None:
        returns, weekdays = self.tracker.get_returns_and_weekdays()
        if len(returns) < 100:
            self.Log("[TRAINING] 데이터 부족")
            return
        self.Log(f"[TRAINING] {len(returns)} returns...")
        best_kl = np.inf
        for K in [2, 3, 4, 5]:
            try:
                self.Log(f"[TRAINING] K={K}...")
                model = DayDependentMSGARCH(K=K, max_iter=30)
                model.fit(returns, weekdays)
                self.msgarch_models[K] = model
                obs_dist = self.tracker.get_distribution()
                sim_dist = model.simulate_weekly_paths(n_simulations=200, n_weeks=self.tracker.sample_weeks)
                kl_high = ModelValidator.kl_divergence(obs_dist["high_probs"], sim_dist["high_probs"])
                kl_low = ModelValidator.kl_divergence(obs_dist["low_probs"], sim_dist["low_probs"])
                avg_kl = (kl_high + kl_low) / 2
                self.Log(f"[TRAINING] K={K}, KL={avg_kl:.4f}")
                if avg_kl < best_kl:
                    best_kl = avg_kl
                    self.best_K = K
                    self.best_model = model
            except Exception as e:
                self.Log(f"[TRAINING] K={K} 실패: {str(e)[:50]}")

    def _run_validation(self) -> None:
        if self.best_model is None:
            return
        try:
            self.Log(f"[VALIDATION] In-sample...")
            obs_dist = self.tracker.get_distribution()
            sim_dist = self.best_model.simulate_weekly_paths(n_simulations=2000, n_weeks=self.tracker.sample_weeks)
            in_sample = ModelValidator.validate_in_sample(obs_dist, sim_dist)
            self.Log(f"[IN] K={self.best_K}")
            self.Log(f"  H: KL={in_sample['KL_high']:.3f}, G={in_sample['G_high']:.2f}, p={in_sample['p_high']:.3f}")
            self.Log(f"  L: KL={in_sample['KL_low']:.3f}, G={in_sample['G_low']:.2f}, p={in_sample['p_low']:.3f}")
            self.Log(f"[VALIDATION] Out-of-sample...")
            returns, weekdays = self.tracker.get_returns_and_weekdays()
            oos = ModelValidator.validate_out_of_sample(returns, weekdays, self.best_K, split_ratio=0.8)
            if "error" not in oos:
                self.Log(f"[OUT] weeks={oos['test_weeks']}")
                self.Log(f"  H: KL={oos['KL_high']:.3f}, G={oos['G_high']:.2f}, p={oos['p_high']:.3f}")
                self.Log(f"  L: KL={oos['KL_low']:.3f}, G={oos['G_low']:.2f}, p={oos['p_low']:.3f}")
                if oos.get('p_high') and oos.get('p_low'):
                    if oos['p_high'] < 0.05 or oos['p_low'] < 0.05:
                        self.Log("[VALIDATION FAILED] p < 0.05")
                        self.best_model = None
            else:
                self.Log(f"[OUT] {oos['error']}")
        except Exception as e:
            self.Log(f"[VALIDATION] 실패: {str(e)[:50]}")
            self.best_model = None

    def _update_trading_plan(self) -> None:
        if self.best_model is None:
            return
        try:
            sim_dist = self.best_model.simulate_weekly_paths(n_simulations=100, n_weeks=10)
            entry_day = int(np.argmax(sim_dist["low_probs"]))
            exit_day = int(np.argmax(sim_dist["high_probs"]))
            entry_prob = sim_dist["low_probs"][entry_day]
            edge = entry_prob - 0.2
            if edge < 0.08:
                self.current_plan = None
                return
            if exit_day <= entry_day:
                exit_day = min(4, entry_day + 1)
            self.current_plan = {"entry_day": entry_day, "exit_day": exit_day, "entry_prob": entry_prob, "edge": edge}
            self.Log(f"[PLAN] Entry={self.day_names[entry_day]} (p={entry_prob:.2f}), Exit={self.day_names[exit_day]}")
        except Exception as e:
            self.current_plan = None

    def _handle_trading(self) -> None:
        if self.current_plan is None:
            return
        weekday = self.Time.weekday()
        if weekday > 4:
            return
        week_key = self.Time.isocalendar()[:2]
        if self.week_state["week_key"] != week_key:
            self.week_state = {"week_key": week_key, "entered": False}
            if self.Portfolio[self.symbol].Invested:
                self.Liquidate(self.symbol)
        entry_day = self.current_plan["entry_day"]
        exit_day = self.current_plan["exit_day"]
        holdings = self.Portfolio[self.symbol].Quantity
        if holdings > 0 and (weekday == exit_day or weekday == 4):
            self.Liquidate(self.symbol)
            self.week_state["entered"] = False
            return
        if not self.week_state["entered"] and weekday == entry_day and not self.Portfolio[self.symbol].Invested:
            if not self.atr.IsReady:
                return
            quantity = self.CalculateOrderQuantity(self.symbol, 0.8)
            if quantity > 0:
                self.MarketOrder(self.symbol, quantity)
                price = self.Securities[self.symbol].Price
                stop_distance = self.atr.Current.Value * 1.5
                stop_price = max(0.01, price - stop_distance)
                self.StopMarketOrder(self.symbol, -quantity, stop_price)
                self.week_state["entered"] = True

    def OnEndOfAlgorithm(self) -> None:
        if self.Portfolio[self.symbol].Invested:
            self.Liquidate(self.symbol)
        self.Log(f"[FINAL] weeks={self.tracker.sample_weeks}, K={self.best_K}")
        obs_dist = self.tracker.get_distribution()
        self.Log(f"[FINAL] High: {obs_dist['high_probs']}")
        self.Log(f"[FINAL] Low: {obs_dist['low_probs']}")
