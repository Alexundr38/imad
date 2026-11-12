"""Нечёткая кластеризация c-средних суточных рядов нейтронного монитора INVK (Инувик, вариант 3).

Данные: одноминутные ряды станции INVK из базы NMDB (www.nmdb.eu), канал с поправкой
на эффективность (RCORR_E), значения — скорость счёта в имп./с. Признаки считаются
в процентах от среднего за сутки, как в эталонной работе. Метод c-средних реализован
самостоятельно по алгоритму функции fcm из MatLab.
"""
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATION = "INVK"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "invk"
DATA_DIR.mkdir(parents=True, exist_ok=True)
URL = "https://www.nmdb.eu/nest/draw_graph.php"
ROW = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):\d{2};\s*(-?[0-9.eE+-]+)")
FORMAT = "%Y-%m-%d"

CALM = """2018-06-11 2018-06-12 2018-06-15 2018-06-16 2018-06-29 2018-07-01 2018-07-02 2018-09-20
2018-10-17 2018-10-18 2018-10-19 2018-10-24 2018-10-27 2018-10-28 2018-10-29 2018-10-30 2018-11-22
2018-11-23 2018-12-12 2018-12-13 2018-12-14 2018-12-15 2018-12-24 2018-01-03 2018-01-05 2018-01-06
2018-01-07 2018-01-10 2018-01-11 2018-01-16 2018-01-17 2018-02-03 2018-02-11 2018-02-13 2018-03-12
2018-03-13 2018-04-03 2018-04-06 2018-04-16 2018-04-17 2018-05-01 2018-05-25 2018-05-25 2018-06-09
2018-06-10""".split()
STORM = """2018-12-28 2018-01-01 2018-01-13 2018-04-12 2018-03-18 2018-04-20 2018-05-05 2018-05-06
2018-05-07 2018-06-01 2018-06-18 2018-06-26 2018-07-05 2018-08-20 2018-09-10 2018-09-11 2018-09-14
2018-09-22 2018-10-08 2018-10-10 2018-10-19 2018-11-05 2018-12-08 2018-12-26 2018-12-29 2019-01-05
2019-01-25 2019-02-01 2019-02-28 2019-03-01 2019-03-17 2019-05-14 2019-08-05 2019-08-31 2019-09-01
2019-10-25 2020-03-04 2020-03-05 2020-06-08 2020-09-28 2020-10-25 2021-03-01 2021-05-12 2021-10-12
2021-11-04""".split()

# Признаки в процентах от среднесуточного счёта (по данным NMDB в имп./с).
FEATURES = ["СКО, %", "Амплитуда, %", "Размах часовых средних, %", "Изменение за сутки, %"]
FINAL = [2, 3]
NC, M_EXP, MAX_ITER, MIN_IMPROVE, N_STARTS, SEED = 2, 2.0, 100, 1e-5, 20, 42
DAILY_FILE = DATA_DIR / f"{STATION}_daily.csv"


def fetch_day(day: pd.Timestamp) -> str:
    """POST-запрос минутных данных INVK за сутки; возвращает страницу NMDB с ascii-блоком."""
    data = [
        ("formchk", "1"), ("stations[]", STATION), ("tabchoice", "revori"),
        ("dtype", "corr_for_efficiency"), ("yunits", "0"), ("date_choice", "bydate"),
        ("start_day", str(day.day)), ("start_month", str(day.month)), ("start_year", str(day.year)),
        ("start_hour", "0"), ("start_min", "0"),
        ("end_day", str(day.day)), ("end_month", str(day.month)), ("end_year", str(day.year)),
        ("end_hour", "23"), ("end_min", "59"), ("output", "ascii"),
    ]
    body = urlencode(data).encode("utf-8")
    last = None
    for attempt in range(5):
        try:
            with urlopen(URL, data=body, timeout=180) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - транзиентные сетевые сбои
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"не удалось получить данные {day}: {last}")


def load_day(day: pd.Timestamp) -> tuple[np.ndarray, int]:
    """Минутный ряд суток на сетке 1440 отсчётов; пропуски заменяются медианой суток."""
    path = DATA_DIR / f"{STATION}_{day:%Y-%m-%d}.txt"
    if not path.exists():
        path.write_text(fetch_day(day), encoding="utf-8")
        time.sleep(0.3)
    x = np.full(1440, np.nan)
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if (m := ROW.match(line)) and f"{m[1]}-{m[2]}-{m[3]}" == f"{day:%Y-%m-%d}":
            x[int(m[4]) * 60 + int(m[5])] = float(m[6])
    n_miss = int(np.isnan(x).sum())
    if n_miss < 1440:
        x[np.isnan(x)] = np.nanmedian(x)
    return x, n_miss


def day_stats(x: np.ndarray, n_miss: int) -> dict:
    """Сводка суток: среднее, СКО, min, max и 24 среднечасовых значения (в %)."""
    if n_miss == 1440:
        return {"n_miss": n_miss}
    stats = {"n_miss": n_miss, "mean": x.mean(), "std": x.std(ddof=1), "min": x.min(), "max": x.max()}
    stats.update({f"h{h}": v for h, v in enumerate(x.reshape(24, 60).mean(axis=1))})
    return stats


def daily_table(days: list[str]) -> pd.DataFrame:
    """Сводки по всем датам: из минутных файлов, если они есть, иначе из кэша INVK_daily.csv."""
    cache = pd.read_csv(DAILY_FILE, sep=";", index_col="date") if DAILY_FILE.exists() else pd.DataFrame()
    rows = {}
    for d in days:
        if (DATA_DIR / f"{STATION}_{d}.txt").exists() or d not in cache.index:
            rows[d] = day_stats(*load_day(pd.Timestamp(d)))
        else:
            rows[d] = cache.loc[d].dropna().to_dict()
    return pd.DataFrame.from_dict(rows, orient="index")


def features(s: pd.Series) -> list[float]:
    """Признаки в процентах от среднего за сутки: так они не зависят от уровня счёта прибора."""
    h = s[[f"h{i}" for i in range(24)]].to_numpy(dtype=float)
    return [
        s["std"] / s["mean"] * 100,
        (s["max"] - s["min"]) / s["mean"] * 100,
        (h.max() - h.min()) / s["mean"] * 100,
        (h[-3:].mean() - h[:3].mean()) / s["mean"] * 100,
    ]


def fcm(data: np.ndarray, nc: int, m: float, max_iter: int, min_improve: float, rng) -> tuple:
    """Нечёткие c-средние, аналог fcm из MatLab: строки data - наблюдения, U размера nc x N."""
    u = rng.random((nc, len(data)))
    u /= u.sum(axis=0)
    history = []
    for _ in range(max_iter):
        um = u**m
        centers = um @ data / um.sum(axis=1, keepdims=True)
        dist = np.fmax(np.linalg.norm(data[None, :, :] - centers[:, None, :], axis=2), 1e-12)
        history.append(float((um * dist**2).sum()))
        u_new = dist ** (-2 / (m - 1))
        u_new /= u_new.sum(axis=0)
        change = np.abs(u_new - u).max()
        u = u_new
        if change < min_improve:
            break
    return centers, u, np.array(history)


def best_fcm(data: np.ndarray, seed: int, m: float = M_EXP) -> tuple:
    """Несколько запусков из случайных начальных U, выбирается разбиение с минимальной J."""
    rng = np.random.default_rng(seed)
    runs = [fcm(data, NC, m, MAX_ITER, MIN_IMPROVE, rng) for _ in range(N_STARTS)]
    return min(runs, key=lambda r: r[2][-1])


def match(y: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Номера кластеров произвольны: выбирается соответствие кластер -> класс с большей долей совпадений."""
    return cl if np.mean(cl == y) >= np.mean((1 - cl) == y) else 1 - cl


def fpc(u: np.ndarray) -> float:
    """Коэффициент разбиения: 1 - чёткое разбиение, 1/Nc - полностью размытое."""
    return float((u**2).sum() / u.shape[1])


def synthetic_check() -> None:
    """Проверка реализации на двух заведомо разделимых группах точек."""
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(5, 1, (60, 2))])
    y = np.repeat([0, 1], 60)
    _, u, hist = best_fcm(x, 1)
    pred = match(y, u.argmax(axis=0))
    print(f"Синтетика: точность {np.mean(pred == y):.3f}, FPC {fpc(u):.3f}, итераций {len(hist)}")


def main() -> None:
    calm, storm = set(CALM), set(STORM)
    both = calm & storm
    print(f"Спокойных в таблице: {len(CALM)}, без повторов: {len(calm)}; бурь: {len(storm)}; "
          f"в обоих классах: {sorted(both)}")
    stats = daily_table(sorted(calm | storm))
    empty = sorted(stats.index[stats["n_miss"] == 1440])
    print(f"Нет данных за сутки: {empty}")
    table = pd.DataFrame([(d, 1) for d in sorted(calm - both)] + [(d, 2) for d in sorted(storm - both)],
                         columns=["date", "class"])
    table = table[~table["date"].isin(empty)].reset_index(drop=True)
    stats = stats.loc[table["date"]]
    n_miss = stats["n_miss"].astype(int).to_numpy()
    print(f"Суток: {len(table)} (спокойных {sum(table['class'] == 1)}, бурь {sum(table['class'] == 2)}); "
          f"пропущено минут: {n_miss.sum()} ({n_miss.sum() / (1440 * len(table)):.3%}), "
          f"суток с пропусками: {np.sum(n_miss > 0)}")

    data = np.array([features(s) for _, s in stats.iterrows()])
    feat = pd.DataFrame(data, columns=FEATURES)
    feat.insert(0, "date", table["date"])
    feat.insert(1, "class", table["class"])
    print("\nПервые строки матрицы data:")
    print(feat.head().to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nСредние признаков по классам:")
    print(feat.groupby("class")[FEATURES].mean().to_string(float_format=lambda v: f"{v:.3f}"))

    y = table["class"].to_numpy() - 1
    synthetic_check()

    print("\n=== Основной прогон: все сутки, все признаки ===")
    run(data, y, list(range(len(FEATURES))))
    print("\nДоля совпадений для наборов признаков:")
    all_cols = list(range(len(FEATURES)))
    combos = [[0], [1], [2], [3], [0, 1], [2, 3], [0, 2, 3], [1, 2, 3], all_cols]
    for cols in combos:
        acc = run(data, y, cols, verbose=False)[0]
        print(f"  {', '.join(FEATURES[c] for c in cols):70s} {acc:.3f}")
    print("\n=== Итоговый прогон: признаки по часовым средним ===")
    acc, pred, u, hist, centers = run(data, y, FINAL)
    print("Устойчивость к m:")
    for m in (1.5, 2.0, 2.5, 3.0):
        print(f"  m = {m}: {run(data, y, FINAL, m=m, verbose=False)[0]:.3f}")

    feat.assign(кластер=pred + 1, u1=u[0], u2=u[1]).to_csv(ROOT / "clusters.csv", sep=";", index=False,
                                                           float_format="%.4f")
    plots(table, data, y, pred, u, hist, centers)


def run(data: np.ndarray, y: np.ndarray, cols: list[int], m: float = M_EXP, verbose: bool = True) -> tuple:
    """Стандартизация выбранных признаков, FCM, сопоставление кластеров с классами."""
    x = data[:, cols]
    mu, sd = x.mean(axis=0), x.std(axis=0)
    centers, u, hist = best_fcm((x - mu) / sd, SEED, m)
    pred = match(y, u.argmax(axis=0))
    if not np.array_equal(pred, u.argmax(axis=0)):
        u, centers = u[::-1], centers[::-1]
    acc = float(np.mean(pred == y))
    if verbose:
        print(f"Суток: {len(y)}, Nc = {NC}, m = {m}, итераций {len(hist)}, J = {hist[-1]:.3f}, FPC = {fpc(u):.3f}")
        print("Центры кластеров в исходных единицах:")
        print(pd.DataFrame(centers * sd + mu, columns=[FEATURES[c] for c in cols],
                           index=["кластер 1", "кластер 2"]).to_string(float_format=lambda v: f"{v:.3f}"))
        print(pd.crosstab(pd.Series(y + 1, name="класс"), pd.Series(pred + 1, name="кластер")).to_string())
        tp, fp = np.sum((pred == 1) & (y == 1)), np.sum((pred == 1) & (y == 0))
        tp_c = np.sum((pred == 0) & (y == 0))
        fp_c = np.sum((pred == 0) & (y == 1))
        fn_c = np.sum((pred == 1) & (y == 0))
        print(f"Доля совпадений: {acc:.3f}; чувствительность (бури): {tp / np.sum(y == 1):.3f}; "
              f"специфичность: {1 - fp / np.sum(y == 0):.3f}; "
              f"recall (спокойные): {tp_c / (tp_c + fn_c):.3f}; "
              f"precision (спокойные): {tp_c / (tp_c + fp_c):.3f}; "
              f"суток с принадлежностью 0.35-0.65: {int(np.sum(np.abs(u[0] - 0.5) < 0.15))}")
    return acc, pred, u, hist, centers * sd + mu


def plots(table, data, y, pred, u, hist, centers) -> None:
    calm_c, storm_c, med_c = "#2A9D8F", "#E76F51", "#F4A261"
    colors = np.array([calm_c, storm_c])
    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D0D6DC",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
    })
    t = np.arange(1440) / 60

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, d, title in zip(axes, ("2018-06-11", "2018-04-20"),
                            ("сутки в спокойном состоянии", "сутки с магнитной бурей")):
        raw = np.full(1440, np.nan)
        for line in (DATA_DIR / f"{STATION}_{d}.txt").read_text().splitlines():
            if m := ROW.match(line):
                raw[int(m[4]) * 60 + int(m[5])] = float(m[6])
        filled, n_miss = load_day(pd.Timestamp(d))
        gap = np.isnan(raw)
        ax.plot(t, filled, lw=0.4, color="#9AA5B1", alpha=0.7, label="одноминутные отсчёты")
        ax.plot(t[gap], filled[gap], ".", ms=3, color=med_c, label=f"восстановлено медианой ({n_miss})")
        ax.plot(np.arange(24) + 0.5, filled.reshape(24, 60).mean(axis=1), color="#24324A", lw=2,
                label="часовые средние")
        ax.set_title(f"{pd.Timestamp(d):%d.%m.%Y}, {title}")
        ax.set_xlabel("Время, ч UTC")
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("Скорость счёта, имп./с")
    fig.tight_layout()
    fig.savefig(ROOT / "fig1_days.png", dpi=200)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(16, 3.8))
    for k, ax in enumerate(axes):
        bp = ax.boxplot([data[y == 0, k], data[y == 1, k]], tick_labels=["класс 1", "класс 2"],
                        patch_artist=True)
        for j, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[j])
            patch.set_alpha(0.6)
        for el in ("whiskers", "caps", "medians"):
            plt.setp(bp[el], color="#24324A")
        ax.set_title(FEATURES[k], fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "fig2_features.png", dpi=200)

    xs, yk = data[:, FINAL], y
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for ax, lab, title in zip(axes, (yk, pred),
                              ("Разметка исходных данных", "Результат кластеризации (максимум принадлежности)")):
        ax.scatter(xs[:, 0], xs[:, 1], c=[colors[v] for v in lab], s=30, alpha=0.85,
                   edgecolors="white", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel(FEATURES[FINAL[0]])
    axes[1].scatter(centers[:, 0], centers[:, 1], marker="X", s=180, c=colors, edgecolors="#24324A",
                    linewidths=0.8, label="центры")
    axes[1].legend(fontsize=9)
    axes[0].set_ylabel(FEATURES[FINAL[1]])
    axes[0].scatter([], [], c=calm_c, label="спокойные сутки (кластер 1)")
    axes[0].scatter([], [], c=storm_c, label="возмущённые сутки (кластер 2)")
    axes[0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "fig3_clusters.png", dpi=200)

    order = np.argsort(yk, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].bar(np.arange(len(yk)), u[1, order], color=[colors[v] for v in yk[order]], width=0.9)
    axes[0].axhline(0.5, color="#24324A", ls="--", lw=0.9)
    axes[0].axvline(np.sum(yk == 0) - 0.5, color="#24324A", lw=1)
    axes[0].set_title("Степень принадлежности наблюдений кластеру 2 (слева — спокойные сутки, справа — возмущённые)")
    axes[0].set_xlabel("Индекс суток")
    axes[0].set_ylabel("μ₂")
    axes[1].plot(np.arange(1, len(hist) + 1), hist, marker="o", ms=3, color=storm_c)
    axes[1].set_title("Значение целевой функции F")
    axes[1].set_xlabel("Номер итерации")
    fig.tight_layout()
    fig.savefig(ROOT / "fig4_membership.png", dpi=200)

    fig, axes = plt.subplots(2, (len(FEATURES) + 1) // 2, figsize=(14, 7))
    for k, ax in enumerate(axes.flat):
        if k >= len(FEATURES):
            ax.axis("off")
            continue
        bins = np.histogram_bin_edges(np.concatenate([data[pred == 0, k], data[pred == 1, k]]), bins=20)
        ax.hist(data[pred == 0, k], bins=bins, alpha=0.55, color=calm_c, edgecolor="white")
        ax.hist(data[pred == 1, k], bins=bins, alpha=0.55, color=storm_c, edgecolor="white")
        ax.set_title(FEATURES[k], fontsize=10)
    fig.legend(handles=[plt.Line2D([0], [0], color=calm_c, label="кластер 1"),
                        plt.Line2D([0], [0], color=storm_c, label="кластер 2")],
               loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Гистограммы признаков в разрезе кластеров")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(ROOT / "fig5_hist_pred.png", dpi=200)


if __name__ == "__main__":
    main()