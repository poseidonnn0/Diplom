"""Модели прогноза. У всех один интерфейс: fit(data) и predict_many(data, origins, horizon).

origin — номер последнего известного шага; прогноз строится на шаги origin+1 .. origin+horizon.
Модель не имеет права заглядывать в data.y дальше origin.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .dataset import SeriesData

# Горизонты, на которых учится бустинг. Остальные он интерполирует по признаку «горизонт».
TRAIN_HORIZONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96)


def weekly_profile(data: SeriesData) -> np.ndarray:
    """Среднее значение для каждого дня недели и шага суток: матрица 7 x spd."""
    sums = np.zeros((7, data.spd))
    hits = np.zeros((7, data.spd))
    np.add.at(sums, (data.dow, data.slot), data.y)
    np.add.at(hits, (data.dow, data.slot), 1)
    slot_mean = sums.sum(axis=0) / np.maximum(hits.sum(axis=0), 1)
    return np.where(hits > 0, sums / np.maximum(hits, 1), slot_mean[None, :])


class Forecaster:
    name = "base"
    title = "Модель"

    def fit(self, data: SeriesData) -> "Forecaster":
        return self

    def predict_many(self, data: SeriesData, origins: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def predict(self, data: SeriesData, origin: int, horizon: int) -> np.ndarray:
        return self.predict_many(data, np.array([origin]), horizon)[0]


class SeasonalNaive(Forecaster):
    """Сезонная наивная модель: «будет как неделю назад в это же время»."""

    name = "seasonal_naive"
    title = "Сезонная наивная"

    def predict_many(self, data: SeriesData, origins: np.ndarray, horizon: int) -> np.ndarray:
        steps = np.arange(1, horizon + 1)
        targets = origins[:, None] + steps[None, :]
        out = np.empty(targets.shape)
        week, day = 7 * data.spd, data.spd
        for row, origin in enumerate(origins):
            for col, target in enumerate(targets[row]):
                if target - week >= 0 and target - week <= origin:
                    out[row, col] = data.y[target - week]
                elif target - day >= 0 and target - day <= origin:
                    out[row, col] = data.y[target - day]
                else:
                    out[row, col] = data.y[origin]
        return out


class WeeklyProfile(Forecaster):
    """Типовой профиль: среднее за прошлые недели для этого дня недели и времени.

    По сути это аналитика бакалаврской версии (средние по часам и дням), превращённая в прогноз.
    """

    name = "weekly_profile"
    title = "Типовой профиль недели"

    def __init__(self) -> None:
        self.profile: Optional[np.ndarray] = None

    def fit(self, data: SeriesData) -> "WeeklyProfile":
        self.profile = weekly_profile(data)
        return self

    def predict_many(self, data: SeriesData, origins: np.ndarray, horizon: int) -> np.ndarray:
        slot, dow = data.target_calendar(origins, np.arange(1, horizon + 1))
        return self.profile[dow, slot]


class GradientBoosting(Forecaster):
    """Градиентный бустинг с пуассоновской функцией потерь (целевая величина — число людей).

    Одна модель на все горизонты: горизонт подаётся как признак, вместе с последними
    значениями ряда, значениями сутки и неделю назад и типовым профилем на целевой момент.
    """

    name = "gradient_boosting"
    title = "Градиентный бустинг"
    feature_names = [
        "horizon", "slot_sin", "slot_cos", "dow", "weekend",
        "last", "lag1", "lag2", "lag3", "mean_hour", "mean_day", "trend_hour",
        "profile_target", "profile_origin", "deviation", "lag_day", "lag_week",
        "ratio_hour", "ratio_day",
    ]

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.06, random_state: int = 0):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.profile: Optional[np.ndarray] = None
        self.model = None

    def _features(self, data: SeriesData, origins: np.ndarray, horizons: np.ndarray) -> np.ndarray:
        y, spd = data.y, data.spd
        n, m = len(origins), len(horizons)
        slot, dow = data.target_calendar(origins, horizons)
        targets = origins[:, None] + horizons[None, :]

        def lagged(offset: int) -> np.ndarray:
            idx = origins - offset
            return np.where(idx >= 0, y[np.maximum(idx, 0)], np.nan)

        def window_mean(size: int) -> np.ndarray:
            csum = np.concatenate([[0.0], np.cumsum(y)])
            start = np.maximum(origins + 1 - size, 0)
            return (csum[origins + 1] - csum[start]) / (origins + 1 - start)

        def seasonal(period: int) -> np.ndarray:
            idx = targets - period
            known = (idx >= 0) & (idx <= origins[:, None])
            return np.where(known, y[np.clip(idx, 0, len(y) - 1)], np.nan)

        last = lagged(0)
        profile_origin = self.profile[data.dow[origins], data.slot[origins]]
        # насколько сегодняшний день загруженнее типового: за последний час и с начала суток (в логарифмах)
        csum = np.concatenate([[0.0], np.cumsum(y)])
        typical = self.profile[data.dow, data.slot]
        csum_typical = np.concatenate([[0.0], np.cumsum(typical)])
        hour_start = np.maximum(origins + 1 - 4, 0)
        day_start = np.maximum(origins - data.slot[origins], 0)

        def log_ratio(start: np.ndarray) -> np.ndarray:
            observed = csum[origins + 1] - csum[start]
            expected = csum_typical[origins + 1] - csum_typical[start]
            return np.log((observed + 1.0) / (expected + 1.0))

        ratio_hour, ratio_day = log_ratio(hour_start), log_ratio(day_start)
        per_origin = np.column_stack([
            last, lagged(1), lagged(2), lagged(3),
            window_mean(4), window_mean(spd), last - lagged(4),
            profile_origin, last - profile_origin,
        ])
        angle = 2 * np.pi * slot / spd
        columns = [
            np.broadcast_to(horizons[None, :], (n, m)),
            np.sin(angle), np.cos(angle), dow, (dow >= 5).astype(float),
            *[np.broadcast_to(per_origin[:, k : k + 1], (n, m)) for k in range(7)],
            self.profile[dow, slot],
            np.broadcast_to(per_origin[:, 7:8], (n, m)),
            np.broadcast_to(per_origin[:, 8:9], (n, m)),
            seasonal(spd), seasonal(7 * spd),
            np.broadcast_to(ratio_hour[:, None], (n, m)),
            np.broadcast_to(ratio_day[:, None], (n, m)),
        ]
        return np.stack([np.asarray(c, dtype=float).reshape(-1) for c in columns], axis=1)

    def fit(self, data: SeriesData) -> "GradientBoosting":
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.profile = weekly_profile(data)
        horizons = np.array([h for h in TRAIN_HORIZONS if h < len(data) - 8])
        origins = np.arange(4, len(data) - int(horizons.max()))
        features = self._features(data, origins, horizons)
        target = data.y[origins[:, None] + horizons[None, :]].reshape(-1)
        self.model = HistGradientBoostingRegressor(
            loss="poisson",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            categorical_features=[self.feature_names.index("dow")],
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_state,
        )
        self.model.fit(features, np.maximum(target, 0.0))
        return self

    def predict_many(self, data: SeriesData, origins: np.ndarray, horizon: int) -> np.ndarray:
        horizons = np.arange(1, horizon + 1)
        features = self._features(data, np.asarray(origins), horizons)
        return self.model.predict(features).reshape(len(origins), horizon)


class HorizonBlend(Forecaster):
    """Комбинация бустинга и типового профиля с весом, зависящим от горизонта.

    На ближних горизонтах точнее бустинг (он видит, что происходит прямо сейчас),
    на дальних — профиль (он не подстраивается под случайности вчерашнего дня).
    Вес бустинга для каждого горизонта подбирается на последних днях обучающего участка.
    """

    name = "horizon_blend"
    title = "Комбинация по горизонту"

    def __init__(self, validation_days: int = 7, stride: int = 4):
        self.validation_days = validation_days
        self.stride = stride
        self.profile_model = WeeklyProfile()
        self.boosting = GradientBoosting()
        self.weights: Optional[np.ndarray] = None

    def fit(self, data: SeriesData) -> "HorizonBlend":
        horizon = max(TRAIN_HORIZONS)
        validation = min(self.validation_days * data.spd, len(data) // 4)
        split = len(data) - validation
        grid = np.linspace(0.0, 1.0, 11)
        if validation > horizon + data.spd and split > 3 * data.spd:
            inner = data.head(split)
            profile, boosting = WeeklyProfile().fit(inner), GradientBoosting().fit(inner)
            origins = np.arange(split - 1, len(data) - horizon, self.stride)
            actual = data.y[origins[:, None] + np.arange(1, horizon + 1)[None, :]]
            p = profile.predict_many(data, origins, horizon)
            g = boosting.predict_many(data, origins, horizon)
            errors = np.stack([np.mean(np.abs(w * g + (1 - w) * p - actual), axis=0) for w in grid])
            best = grid[np.argmin(errors, axis=0)]
            padded = np.pad(best, 4, mode="edge")
            self.weights = np.convolve(padded, np.ones(9) / 9, mode="valid")  # сглаживание по горизонту
        else:
            self.weights = np.linspace(1.0, 0.0, horizon)  # данных мало: плавный переход от бустинга к профилю
        self.profile_model.fit(data)
        self.boosting.fit(data)
        return self

    def predict_many(self, data: SeriesData, origins: np.ndarray, horizon: int) -> np.ndarray:
        origins = np.asarray(origins)
        weights = self.weights[:horizon]
        if len(weights) < horizon:
            weights = np.pad(weights, (0, horizon - len(weights)), mode="edge")
        p = self.profile_model.predict_many(data, origins, horizon)
        g = self.boosting.predict_many(data, origins, horizon)
        return weights[None, :] * g + (1 - weights[None, :]) * p


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


class RecurrentNetwork(Forecaster):
    """Рекуррентная нейросеть (GRU): читает последние двое суток и выдаёт прогноз на все шаги сразу.

    Кодировщик получает ряд и календарь прошлого, декодер — состояние кодировщика
    и календарь будущего (время суток, выходной, типовой профиль).
    """

    name = "gru"
    title = "Нейросеть GRU"

    def __init__(self, horizon: int = 96, hidden: int = 64, epochs: int = 15, batch_size: int = 128,
                 learning_rate: float = 2e-3, window_days: int = 2, seed: int = 0):
        self.horizon = horizon
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.window_days = window_days
        self.seed = seed
        self.profile: Optional[np.ndarray] = None
        self.scale = 1.0
        self.state: Optional[dict] = None
        self._net = None

    # --- сеть ---
    def _build(self):
        import torch
        from torch import nn

        hidden = self.hidden

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.GRU(input_size=5, hidden_size=hidden, batch_first=True)
                self.decoder = nn.Sequential(nn.Linear(hidden + 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))

            def forward(self, past, future):
                _, state = self.encoder(past)
                state = state[-1].unsqueeze(1).expand(-1, future.shape[1], -1)
                out = self.decoder(torch.cat([state, future], dim=-1)).squeeze(-1)
                # сеть предсказывает поправку к типовому профилю, итог не может быть отрицательным
                return torch.relu(out + future[..., 3])

        return Net()

    def _window(self, data: SeriesData) -> int:
        return self.window_days * data.spd

    def _tensors(self, data: SeriesData, origins: np.ndarray, horizon: int):
        window = self._window(data)
        spd = data.spd
        past_idx = origins[:, None] - np.arange(window - 1, -1, -1)[None, :]
        valid = past_idx >= 0
        past_idx = np.clip(past_idx, 0, len(data) - 1)
        angle = 2 * np.pi * data.slot[past_idx] / spd
        past = np.stack([
            np.where(valid, data.y[past_idx], 0.0) / self.scale,
            np.sin(angle), np.cos(angle),
            (data.dow[past_idx] >= 5).astype(float),
            self.profile[data.dow[past_idx], data.slot[past_idx]] / self.scale,
        ], axis=-1)
        slot, dow = data.target_calendar(origins, np.arange(1, horizon + 1))
        angle = 2 * np.pi * slot / spd
        future = np.stack([
            np.sin(angle), np.cos(angle), (dow >= 5).astype(float), self.profile[dow, slot] / self.scale,
        ], axis=-1)
        return past.astype(np.float32), future.astype(np.float32)

    def fit(self, data: SeriesData) -> "RecurrentNetwork":
        import torch

        torch.manual_seed(self.seed)
        self.profile = weekly_profile(data)
        self.scale = float(max(np.percentile(data.y, 95), 1.0))
        horizon = min(self.horizon, len(data) - self._window(data) - 1)
        origins = np.arange(self._window(data) - 1, len(data) - horizon, 2)  # через шаг: вдвое быстрее без потери качества
        past, future = self._tensors(data, origins, horizon)
        target = (data.y[origins[:, None] + np.arange(1, horizon + 1)[None, :]] / self.scale).astype(np.float32)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = self._build().to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
        past_t, future_t, target_t = (torch.from_numpy(a) for a in (past, future, target))
        generator = torch.Generator().manual_seed(self.seed)
        net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(origins), generator=generator)
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                prediction = net(past_t[batch].to(device), future_t[batch].to(device))
                loss = loss_fn(prediction, target_t[batch].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
        self.state = {key: value.detach().cpu() for key, value in net.state_dict().items()}
        self._net = net.cpu().eval()
        return self

    def _network(self):
        if self._net is None:
            self._net = self._build()
            self._net.load_state_dict(self.state)
            self._net.eval()
        return self._net

    def predict_many(self, data: SeriesData, origins: np.ndarray, horizon: int) -> np.ndarray:
        import torch

        past, future = self._tensors(data, np.asarray(origins), horizon)
        net = self._network()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(past), 512):
                chunk = net(torch.from_numpy(past[start : start + 512]), torch.from_numpy(future[start : start + 512]))
                chunks.append(chunk.numpy())
        return np.concatenate(chunks) * self.scale

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_net"] = None  # сеть восстанавливается из весов при первой необходимости
        return state


def candidates(horizon: int = 96, include_neural: bool = True) -> Dict[str, Callable[[], Forecaster]]:
    """Модели, которые участвуют в сравнении. Нейросеть добавляется, если установлен torch."""
    registry: Dict[str, Callable[[], Forecaster]] = {
        SeasonalNaive.name: SeasonalNaive,
        WeeklyProfile.name: WeeklyProfile,
        GradientBoosting.name: GradientBoosting,
        HorizonBlend.name: HorizonBlend,
    }
    if include_neural and torch_available():
        registry[RecurrentNetwork.name] = lambda: RecurrentNetwork(horizon=horizon)
    return registry


def model_titles() -> Dict[str, str]:
    classes: List[type] = [SeasonalNaive, WeeklyProfile, GradientBoosting, HorizonBlend, RecurrentNetwork]
    return {cls.name: cls.title for cls in classes}
