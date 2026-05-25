# Sprint 2: ML Model Improvement with MLflow

## Общее описание проекта

Проект направлен на улучшение модели машинного обучения для прогнозирования рыночной стоимости недвижимости через отслеживание экспериментов в MLflow и последовательную оптимизацию модели.

**Основная задача**: Улучшить качество предсказательной модели через систематический анализ данных, инженерию признаков и оптимизацию параметров модели.

**Бакет S3 для хранения артефактов**: `s3-student-mle-20260317-efc01cb482-freetrack`

## Цели Проекта
1. **Развернуть MLflow** — установить и настроить инфраструктуру для отслеживания экспериментов
2. **Провести EDA** — провести глубокий исследовательский анализ данных
3. **Генерировать признаки** — создать новые информативные признаки
4. **Отобрать признаки** — выбрать наиболее значимые для модели
5. **Оптимизировать параметры** — подобрать гиперпараметры двумя методами

## Быстрый Старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/demetrioprotasov/mle-project-sprint-2-v001
cd mle-project-sprint-2-v001

# 2. Установите зависимости
pip install -r requirements.txt

# 3. Подготовьте .env файл
cp .env.example .env
# Отредактируйте .env с вашими credentials

# 4. Запустите MLflow
sh mlflow_server/run_mlflow_server.sh

# 5. Откройте Jupyter Notebook
jupyter notebook project_template_sprint_2.ipynb

# 6. Откройте MLflow Web UI
# http://localhost:5000
```
**Необходимые переменные в .env:**
```env
# PostgreSQL для MLflow Backend Store
DB_DESTINATION_HOST
DB_DESTINATION_PORT
DB_DESTINATION_NAME
DB_DESTINATION_USER
DB_DESTINATION_PASSWORD

# AWS S3 для артефактов
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
S3_BUCKET_NAME

# MLflow конфигурация
MLFLOW_S3_ENDPOINT_URL
MLFLOW_TRACKING_URI
```

## Этап 1: Разворачивание MLflow и регистрация модели

### Описание
На этом этапе была произведена подготовка инфраструктуры MLflow для отслеживания экспериментов и регистрации моделей.

#### Запуск MLflow сервера (PostgreSQL + S3):
```bash
bash  mlflow_server/run_mlflow_server.sh
```

### Конфигурация MLflow

**Tracking Server URL**: `http://http://localhost:5000`

**Backend Store**: PostgreSQL база данных с метаданными экспериментов
- `DB_HOST`: rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net
- `DB_PORT`: 6432
- `DB_NAME`: playground_mle_20260317_efc01cb482

### Регистрация базовой модели

**Experiment ID**: `[EXPERIMENT_ID]`
**Experiment Name**: `baseline_model`

В этом эксперименте была зарегистрирована исходная модель RandomForestRegressor с параметрами по умолчанию для использования как базовой версии.

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("baseline_model")

with mlflow.start_run(run_name="baseline_run"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)
    mlflow.log_metric("baseline_rmse", baseline_rmse)
    mlflow.log_metric("baseline_r2", baseline_r2)
    mlflow.sklearn.log_model(baseline_model, "model")
```

---

### Этап 2: Проведение исследовательского анализа данных (EDA)

**Описание**: Анализ структуры датасета `users_churn`, выявление закономерностей, распределений целевой переменной и корреляций признаков.

**Ключевые шаги**:
- Анализ структуры данных (размер, типы, пропуски)
- Исследование распределений числовых признаков (monthly_charges, total_charges и др.)
- Анализ категориальных признаков (internet_service, payment_method, tech_support и др.)
- Построение матрицы корреляций и выявление сильных зависимостей с целевой переменной `target`
- Проверка на выбросы и дисбаланс классов

**MLflow**:
- **Experiment Name**: `eda`
- **Run Name**: `eda_overview`
- **Артефакты**: correlation_matrix.png, target_distribution.png, numeric_distributions.png, categorical_analysis.png

---

### Этап 3: Генерация признаков и обучение модели

**Описание**: Создание новых информативных признаков из исходных данных для улучшения качества классификации.

**Типы генерируемых признаков**:
- **Бинаризация категорий**: преобразование категориальных переменных (internet_service, payment_method и др.)
- **Полиномиальные признаки**: взаимодействия между числовыми признаками (monthly_charges × tenure и т.д.)
- **Агрегированные признаки**: отношение total_charges к monthly_charges
- **Статистические трансформации**: стандартизация числовых признаков
- **Группировка**: создание категорий на основе числовых признаков

**Базовая модель обучения**:
- **Модель**: CatBoostClassifier
- **Валидация**: train-test split (80-20%) или cross-validation

**MLflow**:
- **Experiment Name**: `feature_engineering`
- **Run Name**: `feature_generation`
- **Логирование**: количество сгенерированных признаков, их названия, параметры модели, метрики качества
- **Артефакты**: feature_importance.png, model_performance.png

---

### Этап 4: Отбор признаков и обучение новой версии модели

**Описание**: Отбор наиболее важных признаков для снижения размерности модели и предотвращения переобучения.

**Методология отбора**:
1. **Анализ корреляций**: удаление высококоррелирующих признаков (коррелирующих между собой)
2. **Методы отбора признаков**:
   - SelectKBest с критерием chi2 (для классификации)
   - Анализ важности признаков из CatBoost
   - Исключение низкодисперсионных признаков

**Процесс**:
- Определить количество признаков для отбора (k)
- Обучить модель на отобранных признаках
- Оценить влияние на качество (accuracy, precision, recall, AUC-ROC)

**MLflow**:
- **Experiment Name**: `feature_selection`
- **Run Name**: `feature_selection`
- **Логирование**: список отобранных признаков, их количество, метрики модели
- **Артефакты**: selected_features.json, feature_importance_selected.png

---

### Этап 5: Подбор гиперпараметров и обучение финальной модели

**Описание**: Оптимизация гиперпараметров CatBoostClassifier двумя методами для достижения максимального качества классификации.

#### Метод 1: Grid Search

**Пространство гиперпараметров для перебора**:
- `iterations`: [100, 200, 300]
- `depth`: [4, 6, 8]
- `learning_rate`: [0.01, 0.05, 0.1]
- `l2_leaf_reg`: [1, 5, 10]

**Процесс**:
- Кросс-валидация для оценки каждой комбинации
- Выбор лучшей комбинации по метрике качества (accuracy или AUC-ROC)
- Логирование всех протестированных параметров

#### Метод 2: Random Search

**Пространство гиперпараметров для случайного поиска**:
- `iterations`: [50, 100, 150, 200, 300]
- `depth`: [3, 4, 5, 6, 7, 8]
- `learning_rate`: [0.001, 0.01, 0.05, 0.1]
- `l2_leaf_reg`: [1, 3, 5, 10, 20]
- `border_count`: [32, 64, 128]

**Процесс**:
- Количество итераций для случайного поиска (например, 20-30)
- Кросс-валидация для каждого набора параметров
- Сравнение результатов с Grid Search

#### Выбор лучшей модели

Сравнение результатов обоих методов по метрикам качества и выбор наиболее эффективного подхода.

**MLflow**:
- **Experiment Name**: `hyperparameter_tuning`
- **Run Name**: `hyperparameter_tuning`
- **Логирование**: 
  - Параметры Grid Search и Random Search
  - Результаты кросс-валидации для каждого метода
  - Финальные гиперпараметры лучшей модели
- **Артефакты**: grid_search_results.json, random_search_results.json, model_comparison.png, confusion_matrix.png

---

## Финальные результаты

### Финальная модель
```python
CatBoostClassifier(
    iterations=[значение],
    depth=[значение],
    learning_rate=[значение],
    l2_leaf_reg=[значение],
    random_state=42,
    verbose=False
)
```

### Метрики на тестовой выборке

| Метрика | Значение | Описание |
|---------|----------|---------|
| **Accuracy** | [значение] | Общая точность классификации |
| **Precision** | [значение] | Доля правильно предсказанных положительных случаев |
| **Recall** | [значение] | Доля найденных положительных случаев |
| **F1-Score** | [значение] | Гармоническое среднее Precision и Recall |
| **AUC-ROC** | [значение] | Площадь под кривой ROC |

### Матрица ошибок (Confusion Matrix)
```
                  Предсказано класс 0  |  Предсказано класс 1
Реальный класс 0  |        [TN]         |        [FP]
Реальный класс 1  |        [FN]         |        [TP]
```

### Прогресс улучшения
| Этап | Метрика | Значение | Примечание |
|------|---------|----------|-----------|
| Baseline (исходная модель) | Accuracy | [значение] | — |
| + EDA & Генерация признаков | Accuracy | [значение] | [улучшение]% |
| + Отбор признаков | Accuracy | [значение] | [стабилизация] |
| + Подбор гиперпараметров | **Accuracy** | **[значение]** | **[финальное улучшение]%** |

