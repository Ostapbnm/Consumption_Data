"""
ingest_dispatch_payload.py
============================
ТИМЧАСОВА заглушка ingestion-кроку для repository_dispatch. Очікує, що
GitHub Actions передасть payload через env-змінну PAYLOAD (JSON-рядок),
сформований remote-ПК при виклику GitHub API:

    POST /repos/{owner}/{repo}/dispatches
    {
      "event_type": "new-consumption-data",
      "client_payload": {
        "rows": [
          {"hour_start": "2026-08-20 15:00:00", "consumption_kw": 123.4, "n_readings": 60},
          ...
        ]
      }
    }

Формат client_payload ще не узгоджено остаточно ("поки про це не думаємо") -
коли вирішите, який вигляд він матиме на боці remote-ПК, підправте парсинг
нижче під нього. Що вже зроблено правильно: upsert по hour_start
(INSERT OR REPLACE), щоб повторний dispatch з тим самим hour_start (напр.
повторна спроба після мережевого збою) не плодив дублікати, а тихо
перезаписував рядок.

⚠️ Для роботи upsert потрібно, щоб hour_start був PRIMARY KEY (або мав
UNIQUE-індекс) у таблиці hourly_consumption. CREATE TABLE IF NOT EXISTS
нижче додає це лише для щойно створеної таблиці - якщо ви кладете в репо
вже готовий data/hourly_consumption.db (наприклад, наявний
test_sql_source.db), перевірте/додайте цей індекс туди вручну один раз:

    CREATE UNIQUE INDEX IF NOT EXISTS idx_hour_start ON hourly_consumption(hour_start);
"""
import os
import json
import sqlite3

SQL_SOURCE_PATH = os.environ.get('SQL_SOURCE_PATH', 'data/hourly_consumption.db')


def main():
    payload_raw = os.environ.get('PAYLOAD', '{}')
    payload = json.loads(payload_raw)
    rows = payload.get('rows', [])

    if not rows:
        print("ℹ️ У client_payload немає rows - нічого вносити, форекаст запуститься на тому, що вже є в БД.")
        return

    os.makedirs(os.path.dirname(SQL_SOURCE_PATH) or '.', exist_ok=True)
    conn = sqlite3.connect(SQL_SOURCE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hourly_consumption (
            hour_start TEXT PRIMARY KEY,
            consumption_kw REAL NOT NULL,
            n_readings INTEGER NOT NULL
        )
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO hourly_consumption (hour_start, consumption_kw, n_readings) VALUES (?, ?, ?)",
        [(r['hour_start'], r['consumption_kw'], r['n_readings']) for r in rows]
    )
    conn.commit()
    conn.close()
    print(f"✅ Внесено/оновлено {len(rows)} рядків у {SQL_SOURCE_PATH}.")


if __name__ == '__main__':
    main()
