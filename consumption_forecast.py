"""
consumption_forecast.py
=========================
Прогноз споживання на 24-48 год наперед: LightGBM (перенавчається щоразу при
запуску - "постійне навчання на нових даних") + шар адаптивної корекції зсуву
на основі нещодавніх фактичних відхилень (коли з'являться дані в
actuals_consumption).

Адаптовано під хмарне середовище (GitHub Actions):
- жодних хардкод-шляхів до локального ПК - усі шляхи/координати беруться
  з environment variables з розумними дефолтами (working dir = корінь репо);
- дані споживання читаються з SQLite-файлу (SQL_SOURCE_PATH). Наповнення
  цього файлу даними з віддаленого ПК через workflow_dispatch/
  repository_dispatch - окреме питання, тут лише споживач цих даних;
  функцію load_historical_data() і треба буде міняти, коли зміниться
  механізм доставки/формат джерела;
- погода: історична фактична - Open-Meteo Archive API (для навчання),
  прогнозна - Open-Meteo Forecast API з past_days (для майбутніх годин і
  для "мосту" в останні дні, куди Archive API ще не встиг дотягнутися -
  у нього є затримка ~5-6 днів);
- дані споживання живуть у фіксованій (без DST) таймзоні DATA_TIMEZONE =
  'Etc/GMT-2' - перевірено на реальних даних, що лічильник не переходить
  на літній/зимовий час (є окремі рядки і на 03:00, і на 04:00 в день
  переходу). Погода запитується в цивільній TIMEZONE (Open-Meteo іншої не
  розуміє), але одразу конвертується в DATA_TIMEZONE - інакше reindex/join
  між споживанням і погодою під час DST-переходів падає на дублікатах;
- усі HTTP-запити - з ретраями, весь main() - з try/except і
  sys.exit(1) при фатальній помилці, щоб GitHub Actions чітко бачив
  падіння запуску в статусі workflow, а не мовчки "зелений" крок з
  порожнім результатом.

Години з малою кількістю зчитувань (n_readings < LOW_READING_THRESHOLD)
лишаються в даних (нічого не видаляється), але позначаються як ненадійні -
ціль (Споживання) для них ставиться NaN, тому train_model() автоматично
виключає їх з навчання через dropna, не чіпаючи самі рядки.
"""

import os
import sys
import time
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import holidays

# ==============================================================================
# 1. 📂 КОНФІГУРАЦІЯ
# ==============================================================================
# Шляхи - відносні до кореня репозиторію (саме там опиняється working dir
# у GitHub Actions checkout), або перевизначаються env-змінними, якщо
# файли лежать деінде. Локально можна так само задати ці env-змінні,
# або просто тримати data/ поруч зі скриптом.
SQL_SOURCE_PATH = os.environ.get('SQL_SOURCE_PATH', 'data/hourly_consumption.db')
DB_PATH = os.environ.get('FORECAST_DB_PATH', 'data/forecasts.db')

LATITUDE = float(os.environ.get('SITE_LATITUDE', '48.8475'))
LONGITUDE = float(os.environ.get('SITE_LONGITUDE', '24.6894'))
# Цивільний час - для запитів до Open-Meteo (він очікує IANA-таймзону з
# переходами на літній/зимовий час, як і будь-яка погодна станція).
TIMEZONE = 'Europe/Kyiv'
# А ось дані споживання - НЕ цивільний час: перевірено на реальних даних,
# що 29.03.2026 (день переходу на літній час) в лічильнику є і 03:00, і
# 04:00 - тобто пристрій веде облік фіксованим зсувом UTC+2 і DST не
# застосовує. Якщо localізувати цей індекс у Europe/Kyiv, нібито-неіснуюча
# 03:00 навесні "зсувається" в 04:00, де вже є реальний рядок - і
# reindex() падає на дублікаті. Тому для споживання - окрема, фіксована
# (без DST) таймзона; для погоди - лишається цивільна.
DATA_TIMEZONE = 'Etc/GMT-2'  # УВАГА: назва інвертована по POSIX-угоді, це і є UTC+2
OPEN_METEO_MODELS = ['icon_seamless', 'gfs_seamless', 'ecmwf_ifs025']

FORECAST_HOURS = 36
# Запас понад FORECAST_HOURS - щоб прогноз погоди точно покривав future_index,
# навіть якщо той почався не рівно "зараз" (дані споживання можуть відставати).
WEATHER_FORECAST_BUFFER_HOURS = FORECAST_HOURS + 12
# На скільки днів "углиб" тягнути Forecast API (past_days), щоб перекрити
# затримку Archive API. На практиці затримка Archive API ближче до 5-6 днів
# (не 1-2, як спершу закладалось) - 3 дні лишали ~63-годинну дірку в перших
# тестових прогонах. 7 - з запасом.
WEATHER_PAST_DAYS_BRIDGE = 7

# Лаги, безпечні для будь-якого горизонту в межах FORECAST_HOURS (36 год):
# lag_24h навмисно не використовується - для годин 25-36 наперед його ще
# не існує на момент прогнозу.
LAG_HOURS = [48, 168]

VALIDATION_DAYS = 21
MODEL_VERSION = 'lightgbm_v2_weather'

LOW_READING_THRESHOLD = 30  # менше цієї к-сті зчитувань в годині - вважаємо ненадійним

# Наскільки "застарілими" можуть бути дані споживання, перш ніж це стає
# проблемою. WARNING - просто голосно попереджає в лозі (Actions), FAIL -
# зупиняє запуск з помилкою (exit code 1), щоб CI показав явний "провалений"
# статус замість тихого повторення того самого прогнозу щогодини.
# Підберіть під реальну частоту доставки даних з remote ПК - якщо там
# дані приходять раз на годину, поріг варто тримати вужчим.
DATA_STALENESS_WARNING_HOURS = float(os.environ.get('DATA_STALENESS_WARNING_HOURS', '6'))
DATA_STALENESS_FAIL_HOURS = float(os.environ.get('DATA_STALENESS_FAIL_HOURS', '48'))

HTTP_RETRIES = 3
HTTP_BACKOFF_SECONDS = 5


# ==============================================================================
# 2. 🌐 ДОПОМІЖНЕ: HTTP з ретраями, локалізація часових індексів
# ==============================================================================
def _http_get_json(url, params, retries=HTTP_RETRIES, backoff=HTTP_BACKOFF_SECONDS):
    """GET з ретраями - в CI/GitHub Actions поодинокий мережевий збій
    не повинен валити весь запуск."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_exc = exc
            print(f"⚠️ Спроба {attempt}/{retries} запиту до {url} невдала: {exc}")
            if attempt < retries:
                time.sleep(backoff)
    raise last_exc


def _localize_to_tz(index, tz):
    """tz-naive -> tz-aware. Без цього пошук лагів/погоди для майбутніх годин
    мовчки провалиться через конфлікт naive/aware timestamps. Фіксовані
    зсуви (як DATA_TIMEZONE) ніколи не мають неіснуючих/неоднозначних
    годин, тому nonexistent/ambiguous реально спрацьовують лише для
    цивільної TIMEZONE (погода)."""
    if index.tz is not None:
        return index
    try:
        return index.tz_localize(tz, nonexistent='shift_forward', ambiguous='infer')
    except Exception:
        localized = index.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
        return localized[~localized.isna()]


# ==============================================================================
# 3. 📂 ІСТОРИЧНІ ДАНІ СПОЖИВАННЯ (SQL)
# ==============================================================================
def load_historical_data():
    """Читає погодинне споживання з SQLite-джерела."""
    with sqlite3.connect(SQL_SOURCE_PATH) as conn:
        df = pd.read_sql(
            "SELECT hour_start, consumption_kw, n_readings FROM hourly_consumption", conn
        )

    df['Дата'] = pd.to_datetime(df['hour_start'], dayfirst=True)
    df = df.rename(columns={'consumption_kw': 'Споживання'})
    df = df.sort_values('Дата').set_index('Дата')
    df.index = _localize_to_tz(df.index, DATA_TIMEZONE)

    full_range = pd.date_range(df.index.min(), df.index.max(), freq='h', tz=DATA_TIMEZONE)
    missing = full_range.difference(df.index)
    if len(missing) > 0:
        print(f"⚠️ У джерелі бракує {len(missing)} годин (повністю відсутні рядки) "
              f"в діапазоні {full_range.min()} - {full_range.max()}.")

    # Ненадійні години НЕ видаляються з таблиці - лишаються в df, лише ціль
    # (Споживання) стає NaN, щоб пізніший dropna() в train_model() природно
    # виключив їх з навчання, не чіпаючи сам рядок і n_readings.
    unreliable = df['n_readings'] < LOW_READING_THRESHOLD
    if unreliable.any():
        print(f"⚠️ {unreliable.sum()} годин з малою кількістю зчитувань (<{LOW_READING_THRESHOLD}) "
              f"- позначено ненадійними, лишаються в даних, але не увійдуть в навчання.")
        df.loc[unreliable, 'Споживання'] = np.nan

    return df[['Споживання', 'n_readings']]


# ==============================================================================
# 4. 🌡️ ПОГОДА - ІСТОРИЧНА (Open-Meteo Archive API, для навчання)
# ==============================================================================
def fetch_historical_weather(start, end):
    """Фактична температура для навчання. Archive API відстає на 1-2 дні
    від реального часу - найсвіжіший "хвіст" перекриває fetch_temperature_forecast()
    через past_days."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date': end.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m',
        'timezone': TIMEZONE,
    }
    data = _http_get_json(url, params)
    hourly = data['hourly']
    idx = pd.to_datetime(hourly['time'])
    temp = pd.Series(hourly['temperature_2m'], index=idx, name='Температура_C')
    # Open-Meteo повертає час у цивільній TIMEZONE (з DST) - конвертуємо в
    # DATA_TIMEZONE (фіксований UTC+2), бо саме в ній живе індекс споживання,
    # і всі подальші join/reindex мають звірятись по одній сітці годин.
    temp.index = _localize_to_tz(temp.index, TIMEZONE).tz_convert(DATA_TIMEZONE)
    return temp


# ==============================================================================
# 5. 🌡️ ПОГОДА - ПРОГНОЗ + МІСТОК В НЕДАВНЄ МИНУЛЕ (Open-Meteo Forecast API)
# ==============================================================================
def fetch_temperature_forecast(past_days=WEATHER_PAST_DAYS_BRIDGE,
                                forecast_hours=WEATHER_FORECAST_BUFFER_HOURS,
                                models=OPEN_METEO_MODELS):
    """Ансамбль моделей Open-Meteo. past_days перекриває затримку Archive API,
    forecast_hours - вперед для самого прогнозу споживання."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        'latitude': LATITUDE,
        'longitude': LONGITUDE,
        'hourly': 'temperature_2m',
        'models': ','.join(models),
        'timezone': TIMEZONE,
        'past_days': past_days,
        'forecast_hours': forecast_hours,
    }
    data = _http_get_json(url, params)
    hourly = data['hourly']
    time_index = pd.to_datetime(hourly['time'])

    temp_cols = []
    for model in models:
        key = f"temperature_2m_{model}"
        if key in hourly:
            temp_cols.append(pd.Series(hourly[key], index=time_index))
        elif 'temperature_2m' in hourly:
            temp_cols.append(pd.Series(hourly['temperature_2m'], index=time_index))

    temp_series = pd.concat(temp_cols, axis=1).mean(axis=1, skipna=True)
    temp_series.name = 'Температура_C'
    # Та сама причина, що й у fetch_historical_weather - переводимо з
    # цивільної TIMEZONE у фіксовану DATA_TIMEZONE перед будь-яким reindex.
    temp_series.index = _localize_to_tz(temp_series.index, TIMEZONE).tz_convert(DATA_TIMEZONE)
    return temp_series


def build_weather_series(hist_start, hist_end):
    """Об'єднує архівну (навчання) і forecast+past_days (місток + майбутнє)
    температуру в один ряд. Там, де діапазони перекриваються, пріоритет -
    у даних з Forecast API (свіжіші й покривають "хвіст", який Archive
    ще не встиг оновити)."""
    archive_end = hist_end - pd.Timedelta(days=WEATHER_PAST_DAYS_BRIDGE)
    temp_archive = fetch_historical_weather(hist_start, archive_end) if archive_end > hist_start else pd.Series(dtype=float)
    temp_recent_and_future = fetch_temperature_forecast()

    combined = pd.concat([temp_archive, temp_recent_and_future])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    return combined


# ==============================================================================
# 6. 🧩 ОЗНАКИ
# ==============================================================================
def engineer_features(df, ukr_holidays):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df['dow'] >= 5).astype(int)
    df['is_holiday'] = df.index.normalize().isin(ukr_holidays).astype(int)

    for lag in LAG_HOURS:
        df[f'lag_{lag}h'] = df['Споживання'].shift(lag)

    return df


FEATURE_COLS = ['hour', 'dow', 'month', 'is_weekend', 'is_holiday', 'Температура_C'] + \
               [f'lag_{lag}h' for lag in LAG_HOURS]
CATEGORICAL_COLS = ['hour', 'dow', 'month']


# ==============================================================================
# 7. 🧠 НАВЧАННЯ МОДЕЛІ
# ==============================================================================
def train_model(df_features):
    # Ненадійні (n_readings замалі) і рядки без погоди самі відсіюються тут
    # через NaN у цілі/ознаках - рядки в df_features лишаються, у навчання
    # просто не потрапляють.
    train_ready = df_features.dropna(subset=['Споживання'] + FEATURE_COLS)

    cutoff = train_ready.index.max() - pd.Timedelta(days=VALIDATION_DAYS)
    train_part = train_ready[train_ready.index <= cutoff]
    valid_part = train_ready[train_ready.index > cutoff]

    train_set = lgb.Dataset(
        train_part[FEATURE_COLS], label=train_part['Споживання'],
        categorical_feature=CATEGORICAL_COLS
    )
    valid_set = lgb.Dataset(
        valid_part[FEATURE_COLS], label=valid_part['Споживання'],
        categorical_feature=CATEGORICAL_COLS, reference=train_set
    )

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
    }

    model = lgb.train(
        params, train_set, num_boost_round=1000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    valid_pred = model.predict(valid_part[FEATURE_COLS])
    mae = np.mean(np.abs(valid_part['Споживання'] - valid_pred))

    nonzero_mask = valid_part['Споживання'] != 0
    if nonzero_mask.any():
        mape = np.mean(np.abs(
            (valid_part['Споживання'][nonzero_mask] - valid_pred[nonzero_mask.values]) /
            valid_part['Споживання'][nonzero_mask]
        )) * 100
    else:
        mape = float('nan')

    print(f"📊 Перевірка на останніх {VALIDATION_DAYS} днях: MAE = {mae:.1f} кВт, MAPE = {mape:.1f}%")

    return model


# ==============================================================================
# 8. 🔮 ПОБУДОВА ОЗНАК ДЛЯ МАЙБУТНІХ ГОДИН
# ==============================================================================
def build_future_index(reference_time):
    """FORECAST_HOURS год наперед від наступної повної години після
    reference_time. Навмисно НЕ прив'язано до останньої точки в df_hist -
    reference_time має бути реальний поточний час (див. main()), інакше
    при застарілих даних (збій доставки з remote ПК) прогноз мовчки
    "втікає" в минуле відносно реального "зараз", замість передбачення
    на актуальний час з fallback-значеннями там, де свіжих даних бракує."""
    start = (reference_time + pd.Timedelta(hours=1)).floor('h')
    return pd.date_range(start=start, periods=FORECAST_HOURS, freq='h', tz=DATA_TIMEZONE)


def build_future_features(df_hist, future_index, weather_series, ukr_holidays):
    future_df = pd.DataFrame(index=future_index)
    future_df['hour'] = future_df.index.hour
    future_df['dow'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    future_df['is_weekend'] = (future_df['dow'] >= 5).astype(int)
    future_df['is_holiday'] = future_df.index.normalize().isin(ukr_holidays).astype(int)
    future_df['Температура_C'] = weather_series.reindex(future_index).values

    consumption_hist = df_hist['Споживання']
    for lag in LAG_HOURS:
        lookup_times = future_df.index - pd.Timedelta(hours=lag)
        future_df[f'lag_{lag}h'] = consumption_hist.reindex(lookup_times).values

    return future_df


# ==============================================================================
# 9. ⚖️ АДАПТИВНА КОРЕКЦІЯ ЗСУВУ
# ==============================================================================
def compute_recent_bias(db_path, lookback_days=14, ewma_span=7):
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS actuals_consumption (
                datetime TEXT NOT NULL,
                actual_kw REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consumption_forecast (
                run_timestamp TEXT NOT NULL,
                target_datetime TEXT NOT NULL,
                horizon_hours REAL NOT NULL,
                model_version TEXT NOT NULL,
                predicted_kw REAL NOT NULL
            )
        """)
        conn.commit()

        forecasts = pd.read_sql("SELECT target_datetime, predicted_kw FROM consumption_forecast", conn)
        actuals = pd.read_sql("SELECT datetime, actual_kw FROM actuals_consumption", conn)

    if forecasts.empty or actuals.empty:
        return 0.0

    # Порівнюємо як реальні datetime (UTC), а не як сирі рядки - інакше
    # найменша різниця в форматі ISO-рядка (наявність секунд, офсет тощо)
    # між тим, що пише log_consumption_forecast(), і тим, що пише окремий
    # скрипт-завантажувач actuals_consumption, мовчки зіб'є весь join,
    # і корекція завжди буде виглядати як "даних ще немає".
    forecasts['target_datetime'] = pd.to_datetime(forecasts['target_datetime'], utc=True, errors='coerce')
    actuals['datetime'] = pd.to_datetime(actuals['datetime'], utc=True, errors='coerce')
    forecasts = forecasts.dropna(subset=['target_datetime'])
    actuals = actuals.dropna(subset=['datetime'])

    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
    forecasts = forecasts[forecasts['target_datetime'] >= cutoff]

    matched = forecasts.merge(actuals, left_on='target_datetime', right_on='datetime', how='inner')
    if matched.empty:
        return 0.0

    matched = matched.sort_values('target_datetime')
    errors = matched['actual_kw'] - matched['predicted_kw']
    bias = errors.ewm(span=ewma_span).mean().iloc[-1]
    return float(bias)


# ==============================================================================
# 10. 💾 ЛОГУВАННЯ ПРОГНОЗУ
# ==============================================================================
def log_consumption_forecast(predictions, db_path=DB_PATH, model_version=MODEL_VERSION):
    run_timestamp = datetime.now(timezone.utc)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consumption_forecast (
                run_timestamp TEXT NOT NULL,
                target_datetime TEXT NOT NULL,
                horizon_hours REAL NOT NULL,
                model_version TEXT NOT NULL,
                predicted_kw REAL NOT NULL
            )
        """)

        rows = []
        for target_dt, predicted_kw in predictions.items():
            target_dt_utc = target_dt.tz_convert('UTC') if target_dt.tzinfo else target_dt
            horizon_hours = (target_dt_utc - run_timestamp).total_seconds() / 3600
            # Пишемо target_datetime в UTC - узгоджено з тим, як його потім
            # парсить compute_recent_bias(). Якщо колись зробите окремий
            # завантажувач actuals_consumption - пишіть туди datetime теж у UTC.
            rows.append((run_timestamp.isoformat(), target_dt_utc.isoformat(),
                         round(horizon_hours, 2), model_version, float(predicted_kw)))

        conn.executemany("""
            INSERT INTO consumption_forecast
            (run_timestamp, target_datetime, horizon_hours, model_version, predicted_kw)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

    return len(rows)


# ==============================================================================
# 11. 🚀 ОСНОВНИЙ ЗАПУСК
# ==============================================================================
def main():
    print("📂 Завантажуємо історичні дані з SQL...")
    df_hist = load_historical_data()

    now = pd.Timestamp.now(tz=DATA_TIMEZONE)
    data_age_hours = (now - df_hist.index.max()).total_seconds() / 3600
    if data_age_hours > DATA_STALENESS_FAIL_HOURS:
        raise RuntimeError(
            f"Дані застаріли на {data_age_hours:.1f} год (поріг: {DATA_STALENESS_FAIL_HOURS} год). "
            f"Останній рядок у джерелі: {df_hist.index.max()}, зараз: {now}. "
            f"Схоже, доставка даних з remote ПК не працює - зупиняю запуск, "
            f"щоб не публікувати прогноз, побудований на явно неактуальних даних."
        )
    elif data_age_hours > DATA_STALENESS_WARNING_HOURS:
        print(f"⚠️ Дані застаріли на {data_age_hours:.1f} год (останній рядок: {df_hist.index.max()}). "
              f"Прогноз все одно рахується від реального поточного часу - лаги там, де свіжих "
              f"даних бракує, підстрахує fallback-медіана по годині доби, але точність нижче звичної.")

    years = range(df_hist.index.year.min(), df_hist.index.year.max() + 2)
    ukr_holidays = holidays.Ukraine(years=years)

    print("🌡️ Отримуємо погоду (архів + прогноз) з Open-Meteo...")
    weather_series = build_weather_series(df_hist.index.min(), df_hist.index.max())
    df_hist = df_hist.join(weather_series.rename('Температура_C'))
    missing_weather = df_hist['Температура_C'].isna().sum()
    if missing_weather:
        print(f"⚠️ {missing_weather} годин без температури (поза покриттям Archive/Forecast API) "
              f"- ці рядки не увійдуть у навчання.")

    print("🧠 Готуємо ознаки й навчаємо модель...")
    df_features = engineer_features(df_hist, ukr_holidays)
    model = train_model(df_features)

    future_index = build_future_index(now)
    future_features = build_future_features(df_hist, future_index, weather_series, ukr_holidays)

    for lag in LAG_HOURS:
        col = f'lag_{lag}h'
        if future_features[col].isna().any():
            fallback = df_hist.groupby(df_hist.index.hour)['Споживання'].median()
            missing_mask = future_features[col].isna()
            future_features.loc[missing_mask, col] = future_features.loc[missing_mask, 'hour'].map(fallback)

    if future_features['Температура_C'].isna().any():
        # Не мало б статись при WEATHER_FORECAST_BUFFER_HOURS > FORECAST_HOURS,
        # але про всяк випадок - краще підставити середнє, ніж впасти на predict().
        fallback_temp = weather_series.mean()
        print("⚠️ Немає прогнозу температури для частини future_index - підставляємо середнє.")
        future_features['Температура_C'] = future_features['Температура_C'].fillna(fallback_temp)

    raw_predictions = pd.Series(
        model.predict(future_features[FEATURE_COLS]),
        index=future_features.index
    )

    print("⚖️ Перевіряємо, чи потрібна адаптивна корекція за нещодавніми фактами...")
    bias = compute_recent_bias(DB_PATH)
    if bias != 0.0:
        print(f"   → Виявлено систематичне відхилення {bias:+.1f} кВт, коригуємо прогноз.")
    else:
        print("   → Фактичних даних для порівняння ще немає, корекція = 0.")

    corrected_predictions = (raw_predictions + bias).clip(lower=0)

    print(f"💾 Записуємо прогноз в базу: {DB_PATH}")
    n_rows = log_consumption_forecast(corrected_predictions)
    print(f"✅ Записано {n_rows} рядків.")

    print()
    print("Прогноз споживання (кВт), перші 10 годин:")
    print(corrected_predictions.head(10))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Явний ненульовий exit code - щоб GitHub Actions показав запуск
        # як провалений, а не "зелений" крок з обірваним логом.
        print(f"❌ Запуск провалився: {exc}", file=sys.stderr)
        raise
