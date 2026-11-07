# Wine Quality — демо-приложение сомелье (ДЗ-3, часть 4)
#
# Модель здесь НЕ обучается: приложение загружает итоговый пайплайн из ноутбука (разделы 3.5–3.6:
# преобразования признаков + LogisticRegression), порог и тестовую выборку (model.joblib, config.json, test.csv).
#
# 4.5. Ответ на вопрос о кэшировании:
# - Для модели используется st.cache_resource: модель — единственный «ресурс» (тяжёлый неизменяемый
#   объект, общий для всех пользователей и всех перезапусков скрипта). st.cache_resource не хеширует
#   содержимое, а просто держит один экземпляр объекта на весь процесс: он создаётся один раз.
# - Для таблиц (test.csv, предвычисленные вероятности) используется st.cache_data: это данные,
#   зависящие от аргументов; кэш хранит результат по хешу аргументов (тот же вызов — тот же ответ,
#   изменились аргументы — пересчёт), при чтении возвращается копия.
# - Если вызвать joblib.load без декоратора, модель будет перечитываться с диска при каждом
#   действии пользователя (каждое движение слайдера перезапускает app.py сверху вниз) —
#   лишние десятки-сотни миллисекунд на каждый сдвиг.
# - Путь к файлам строится от папки app.py: Path(__file__).parent, а не от текущей директории.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import metrics

BASE = Path(__file__).parent


def add_log1p(Xd, cols=()):
    """log1p-преобразование части признаков — та же функция, что в ноутбуке (3.5).

    FunctionTransformer при сохранении модели держит только ссылку на имя функции
    (__main__.add_log1p), поэтому в приложении должна быть ровно такая же функция.
    """
    Xd = Xd.copy()
    for c in cols:
        Xd[c] = np.log1p(Xd[c])
    return Xd


@st.cache_resource
def load_model():
    """Модель — единственный ресурс на весь процесс (см. комментарий про 4.5 выше)."""
    return joblib.load(BASE / 'model.joblib')


@st.cache_data
def load_config():
    return json.loads((BASE / 'config.json').read_text())


@st.cache_data
def load_test():
    return pd.read_csv(BASE / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    """Вероятности класса 1 на тестовой выборке: считаются один раз (аргумент _model не хешируется)."""
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='Wine Quality — сомелье', page_icon='🍷', layout='wide')

model = load_model()
DEFAULT_THR = float(load_config()['threshold'])
FEATURES = list(model.feature_names_in_)
NUM_FEATURES = [c for c in FEATURES if c != 'is_red']

# --- сайдбар: порог (по умолчанию — из config.json; общий для вкладок 2 и 3) ---
st.sidebar.header('Порог модели')
thr = st.sidebar.slider('Порог «хорошего» вина', 0.0, 1.0, DEFAULT_THR, 0.01,
                        key='threshold_slider', format='%.2f')
st.sidebar.caption(f'Вино считается «хорошим», если P(хорошее) ≥ {thr:.2f}.\n\n'
                   f'Порог по умолчанию ({DEFAULT_THR:.3f}) подобран на валидации в ноутбуке:\n'
                   f'максимальный precision при recall ≥ 0.7.')

test_df = load_test()
y_test = test_df['target']
proba = test_proba(model, test_df[FEATURES])

tab1, tab2, tab3, tab4 = st.tabs(
    ['🍷 Одно вино', '📊 Модельные метрики', '📦 Пакетное предсказание', '🧠 Что выучила модель'])

# ============================ 4.1. Оценка одного вина ============================
with tab1:
    st.header('Оценка одного вина')
    st.caption('Диапазоны и значения по умолчанию — минимум/максимум/медиана по тестовой выборке.')

    cols = st.columns(3)
    values = {}
    for i, col_name in enumerate(NUM_FEATURES):
        lo, hi = float(test_df[col_name].min()), float(test_df[col_name].max())
        med = float(test_df[col_name].median())
        with cols[i % 3]:
            values[col_name] = st.number_input(
                col_name.replace('_', ' '), min_value=lo, max_value=hi, value=med,
                step=(hi - lo) / 200 if hi > lo else 0.1, format='%.3f', key=f'num_{col_name}')

    is_red = st.radio('Тип вина', ['белое (is_red = 0)', 'красное (is_red = 1)'],
                      horizontal=True, key='is_red') == 'красное (is_red = 1)'

    row = pd.DataFrame([{**values, 'is_red': 1.0 if is_red else 0.0}])[FEATURES]
    p1 = float(model.predict_proba(row)[0, 1])

    c1, c2, c3 = st.columns(3)
    c1.metric('Вероятность «хорошего»', f'{p1:.1%}',
              delta='выше порога' if p1 >= thr else 'ниже порога')
    c2.metric(f'Решение при пороге {thr:.2f}', '✅ в подборку' if p1 >= thr else '❌ не в подборку')
    c3.metric('Тип вина', 'красное' if is_red else 'белое')
    st.progress(min(p1, 1.0), text=f'P(хорошее) = {p1:.3f} (порог {thr:.2f})')

# ==================== 4.2. Интерактивная визуализация метрик ====================
with tab2:
    st.header('Модельные метрики на тестовой выборке')
    st.caption(f'Тест: {len(test_df)} вин, доля «хороших» {(y_test == 1).mean():.1%}. '
               f'Всё пересчитывается при движении порога в сайдбаре.')

    pred = (proba >= thr).astype(int)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Precision', f'{metrics.precision_score(y_test, pred, zero_division=0):.3f}')
    m2.metric('Recall', f'{metrics.recall_score(y_test, pred, zero_division=0):.3f}')
    m3.metric('F1', f'{metrics.f1_score(y_test, pred, zero_division=0):.3f}')
    m4.metric('Доля вин в подборке', f'{pred.mean():.1%}')

    col_cm, col_curves = st.columns([1, 1.4])
    with col_cm:
        st.subheader('Матрица ошибок')
        fig_cm, ax = plt.subplots(figsize=(3.8, 3))
        metrics.ConfusionMatrixDisplay.from_predictions(
            y_test, pred, display_labels=['не хорошее', 'хорошее'], cmap='Blues',
            ax=ax, colorbar=False)
        st.pyplot(fig_cm)

    with col_curves:
        st.subheader('ROC- и PR-кривые (точка — текущий порог)')
        fpr, tpr, th_roc = metrics.roc_curve(y_test, proba)
        i_roc = int(np.argmin(np.abs(th_roc - thr)))
        pre, rec, th_pr = metrics.precision_recall_curve(y_test, proba)
        i_pr = int(np.argmin(np.abs(th_pr - thr)))

        fig_roc, ax1 = plt.subplots(figsize=(4.4, 3.2))
        ax1.plot(fpr, tpr)
        ax1.plot(fpr[i_roc], tpr[i_roc], 'ro')
        ax1.set(xlabel='FPR', ylabel='TPR', title=f'ROC-AUC = {metrics.roc_auc_score(y_test, proba):.3f}')

        fig_pr, ax2 = plt.subplots(figsize=(4.4, 3.2))
        ax2.plot(rec, pre)
        ax2.plot(rec[i_pr], pre[i_pr], 'ro')
        prev = (y_test == 1).mean()
        ax2.axhline(prev, ls='--', lw=1, c='gray')
        ax2.text(0.5, prev + 0.015, f'случайная модель = {prev:.2f}',
                 ha='center', va='bottom', fontsize=8, color='gray')
        ax2.set(xlabel='recall', ylabel='precision',
                title=f'PR-AUC = {metrics.average_precision_score(y_test, proba):.3f}')

        cc1, cc2 = st.columns(2)
        cc1.pyplot(fig_roc)
        cc2.pyplot(fig_pr)

    st.info(
        f'**Подсказка сомелье.** При текущем пороге {thr:.2f} в подборку попадает {pred.mean():.1%} всех вин, '
        f'из них «хороших» — {metrics.precision_score(y_test, pred, zero_division=0):.1%}. '
        f'**Порог больше** — модель требовательнее: подборка меньше и чище (precision растёт), '
        f'но больше хороших вин остаётся вне её (recall падает). '
        f'**Порог меньше** — наоборот: подборка шире и полнее (recall растёт), '
        f'но посредственных вин в ней становится больше (precision падает).'
    )

# ========================= 4.3. Пакетное предсказание =========================
with tab3:
    st.header('Пакетное предсказание')
    st.caption(f'Загрузите CSV со столбцами: {", ".join(FEATURES)}. '
               f'Используется порог из сайдбара: {thr:.2f}.')

    uploaded = st.file_uploader('CSV-файл с признаками вин (без столбца target)', type='csv')
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error(f'В файле не хватает столбцов: {", ".join(missing)}. '
                     f'Нужны признаки: {", ".join(FEATURES)}.')
        else:
            probs = model.predict_proba(batch[FEATURES])[:, 1]
            out = batch.copy()
            out['p_good'] = np.round(probs, 4)
            out['решение'] = np.where(probs >= thr, 'в подборку', 'не в подборку')
            st.success(f'Готово: {len(out)} вин, порог {thr:.2f}, '
                       f'в подборку: {(probs >= thr).sum()}.')
            st.dataframe(out)
            st.download_button('⬇️ Скачать предсказания (CSV)',
                               data=out.to_csv(index=False).encode('utf-8'),
                               file_name='predictions.csv', mime='text/csv', key='download_pred')

# ========================== 4.4. Что выучила модель ==========================
with tab4:
    st.header('Что выучила модель')
    st.caption('Коэффициенты итоговой модели: в её признаки входят и сами показатели, и их '
               'парные произведения (после преобразований пайплайна); столбик — вклад признака '
               'в log-odds «хорошего» вина.')

    coefs = model[-1].coef_[0]
    order = np.argsort(coefs)
    names = np.array([n.replace(' ', '×') for n in model[:-1].get_feature_names_out()])[order]
    vals = coefs[order]

    fig_c, ax = plt.subplots(figsize=(8, max(5, 0.16 * len(vals))))
    ax.barh(names, vals, color=['#c0392b' if v < 0 else '#27885f' for v in vals])
    ax.axvline(0, c='k', lw=0.8)
    ax.set(title='Коэффициенты итоговой модели',
           xlabel='вклад в log-odds «хорошего» вина')
    st.pyplot(fig_c)

    good3 = [n for n, v in zip(names[::-1], vals[::-1]) if v > 0][:3]
    bad3 = [n for n, v in zip(names, vals) if v < 0][:3]
    st.markdown(
        f'**За «хорошее» вино** сильнее всего голосуют: **{", ".join(good3)}** — чем они больше, '
        f'тем выше шанс хорошего отзыва.\n\n'
        f'**Против**: **{", ".join(bad3)}** — их рост понижает оценку модели.')
