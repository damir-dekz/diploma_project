import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.utils.multiclass import unique_labels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    auc,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
import scipy.stats
import scipy
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
from matplotlib.pyplot import figure
import plotly.figure_factory as ff
import datetime as dt
import functools
import operator
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def tests(
        type_model,
        type_block,
        options,
        tab2,
        train,
        test,
        df_raw,
        features_raw,
        data_for_dyn,
        format_date,
        test_or_oot,
        model_clf,
        development_date,
):
    if (
            "M1.1 Анализ репрезентативности выборки(-ок), использованной(-ых) на этапе разработки в разрезе факторов модели"
            in options
    ):

        try:
            psi_df, m1_1_color = m1_1(train, test, type_block, type_model, tab2)
        except:
            tab2.error(
                "**М1.1 АНАЛИЗ РЕПРЕЗЕНТАТИВНОСТИ ВЫБОРКИ (ЗНАЧИМЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ**"
            )
            pass

    if "M1.2 Актуальность данных" in options:
        try:
            m1_2_color, m1_2_df = m1_2(data_for_dyn, development_date, type_block, type_model, tab2)
        except:
            tab2.error(
                "**M1.2 АКТУАЛЬНОСТЬ ДАННЫХ (ЗНАЧИМЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ**"
            )
            pass

    if "M1.3 Анализ пропущенных значений" in options:
        try:
            a_empty = m1_3(df_raw, features_raw, tab2)
        except:
            tab2.error(
                "**М1.3 АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ.(ИНФОРМАТИВНЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ СЫРЫХ "
                "ДАННЫХ**"
            )
            pass

        try:
            calc_month, calc_quart, m1_3_color = m1_3_stability(df_raw, tab2)
        except:
            tab2.error(
                "**СТАБИЛЬНОСТЬ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ.(ИНФОРМАТИВНЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ СЫРЫХ ДАННЫХ**"
            )
            pass

    if type_block in ["Бизнес блок", "Розничный блок"] or (
            type_block == "Корпоративный блок" and type_model == "Антифрод модель"):
        if (
                "M1.4 Проверка глубины и качества данных, использованных в разработке модели"
                in options
        ):
            try:
                m1_4_glubina_color, glubina_month, data_dep = m1_4_glubina(data_for_dyn, type_block, type_model, tab2)
            except:
                tab2.error(
                    "**M1.4 ПРОВЕРКА ГЛУБИНЫ И КАЧЕСТВА ДАННЫХ, ИСПОЛЬЗОВАННЫХ В РАЗРАБОТКЕ МОДЕЛИ.(ИНФОРМАТИВНЫЙ ТЕСТ)"
                    "⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩЕЙ И ТЕСТОВОВЫХ ДАННЫХ**"
                )
                pass
    if type_block == "Корпоративный блок" and type_model == "Скоринговая модель":
        if (
                "M1.4 Анализ нормальности распределения факторов и проверка на наличие выбросов"
                in options
        ):
            try:
                m1_4_analyze(test, train, tab2)
            except:
                tab2.error(
                    "**М1.4 АНАЛИЗ НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ ФАКТОРОВ И ПРОВЕРКА НА НАЛИЧИЕ ВЫБРОСОВ (ЗНАЧИМЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
                )
                pass

        if (
                "M1.5 Проверка глубины и качества данных, использованных в разработке модели"
                in options
        ):
            try:
                data_dep, color, length = data_depth(
                    data_for_dyn, "ISSUE_DATE", "TARGET", type_block, type_model
                )
                if color == "Красный":
                    color_picker = ":red"
                elif color == "Желтый":
                    color_picker = ":orange"
                else:
                    color_picker = ":green"
                tab2.write(
                    "**М1.5 ПРОВЕРКА ГЛУБИНЫ И КАЧЕСТВА ДАННЫХ, ИСПОЛЬЗОВАННЫХ В РАЗРАБОТКЕ МОДЕЛИ.(ЗНАЧИМЫЙ ТЕСТ)-** "
                    + color_picker
                    + "[**{}**]".format(color)
                )
                tab2.write(
                    color_picker + f"[**Глубина данных составляет {length} месяцев.**]"
                )
                tab2.write(data_dep)
            except:
                tab2.error(
                    "**M1.5 ПРОВЕРКА ГЛУБИНЫ И КАЧЕСТВА ДАННЫХ, ИСПОЛЬЗОВАННЫХ В РАЗРАБОТКЕ МОДЕЛИ.(ЗНАЧИМЫЙ ТЕСТ)"
                    "⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩЕЙ И ТЕСТОВОВЫХ ДАННЫХ**"
                )
                pass

    if "M2.1 Эффективность ранжирования всей модели" in options:
        try:
            m2_1_color, roc_fig = m2_1_all_model(train, test, type_block, tab2)
        except:
            tab2.error(
                "**М2.1 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЕ ВСЕЙ МОДЕЛИ.(ЗНАЧИМЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
            )
            pass

    if type_block in ["Розничный блок", "Бизнес блок"] or (
            type_block == "Корпоративный блок" and type_model == "Антифрод модель"):
        if "M2.2 Использование процедуры бутстрепа" in options:
            try:
                m2_2_color, m2_2_df = m2_2_bootstrap(train, test, type_block, model_clf, tab2)
            except:
                tab2.error(
                    "**M2.2 Использование процедуры бутстрепа (ЗНАЧИМЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ "
                    "В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                )
                pass

        if "M2.3 Эффективность ранжирования с исключением отдельных факторов модели" in options:
            try:
                m2_3_color, m2_3_df = m2_3(model_clf, train, test, tab2)
            except:
                tab2.error(
                    "**М2.3 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ С ИСКЛЮЧЕНИЕМ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                )
                pass

        if "M2.4 Эффективность ранжирования отдельных факторов модели" in options:
            try:
                m2_4_color, e_r_o_f_TR, e_r_o_f_TS = m2_4(train, test, tab2)
            except:
                tab2.error(
                    "**М2.4 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ ДЛЯ ИНТЕПРЕТИРУЕМЫХ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
                )
                pass

        if "M2.5 Динамика коэффициента Джини" in options:
            try:
                gini_dinamic_month_plot, gini_dinamic_quart_plot = m2_5_dinamic_gini(data_for_dyn, tab2, format_date)
            except:
                tab2.error(
                    "**М2.5 ДИНАМИКА КОЭФФИЦИЕНТА ДЖИНИ. (ЗНАЧИМЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
                )
                pass
        if type_block == "Бизнес блок" or (
                type_block in ["Розничный блок", "Корпоративный блок"] and type_model == "Антифрод модель"):
            if "M2.6 Оценка производительности (общей эффективности) модели" in options:
                try:
                    m2_6_res = m2_6(train, test, tab2)
                except:
                    tab2.error(
                        "**M2.6 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ (ОБЩЕЙ ЭФФЕКТИВНОСТИ) МОДЕЛИ (ИНФОРМАТИВНЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                    )
                    pass

            if "M2.7 Оценка точности модели в определении истинно положительных результатов" in options:
                try:
                    precision_res, m2_7_color = m2_7(train, test, type_block, type_model, tab2)
                except:
                    tab2.error(
                        "**M2.7 ОЦЕНКА ТОЧНОСТИ МОДЕЛИ В ОПРЕДЕЛЕНИИ ИСТИННО ПОЛОЖИТЕЛЬНЫХ РЕЗУЛЬТАТОВ (ИНФОРМАТИВНЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                    )
                    pass

            if "M2.8 Оценка полноты (частоты) модели в определении истинно положительных результатов" in options:
                try:
                    m2_8_color, recall_res = m2_8(train, test, type_block, type_model, tab2)
                except:
                    tab2.error(
                        "**M2.8 ОЦЕНКА ПОЛНОТЫ (ЧАСТОТЫ) МОДЕЛИ В ОПРЕДЕЛЕНИИ ИСТИННО ПОЛОЖИТЕЛЬНЫХ РЕЗУЛЬТАТОВ (ИНФОРМАТИВНЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                    )
                    pass

            if "M2.9 Оценка среднего гармонического значения между точностью и полнотой модели" in options:
                try:
                    m2_9_color, f1_res = m2_9(train, test, type_block, type_model, tab2)
                except:
                    tab2.error(
                        "**M2.9 ОЦЕНКА СРЕДНЕГО ГАРМОНИЧЕСКОГО ЗНАЧЕНИЯ МЕЖДУ ТОЧНОСТЬЮ И ПОЛНОТОЙ МОДЕЛИ (ИНФОРМАТИВНЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                    )
                    pass

            if "M2.10 Оценка эффективности модели – матрица ошибок" in options:
                try:
                    m2_10_color, error_matrix_df = m2_10(train, test, tab2)
                except:
                    tab2.error(
                        "**M2.10 ОЦЕНКА ЭФФЕКТИВНОСТИ МОДЕЛИ – МАТРИЦА ОШИБОК (ИНФОРМАТИВНЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                    )
                    pass

            if "M2.11 Оценка качества вероятностных предсказаний" in options:
                try:
                   m2_11_color, mean_log_loss = m2_11(test, tab2)
                except:
                    tab2.error(
                        "**M2.11: ОЦЕНКА КАЧЕСТВА ВЕРОЯТНОСТНЫХ ПРЕДСКАЗАНИЙ (ЗНАЧИМЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**"
                    )
                    pass
    elif type_block == "Корпоративный блок" and type_model == "Скоринговая модель":
        if "M2.2 Эффективность ранжирования отдельных факторов модели" in options:
            try:
                train_ = train.copy()
                test_ = test.copy()
                if "ISSUE_DATE" in list(train.columns):
                    train_ = train.drop(columns="ISSUE_DATE")
                    test_ = test.drop(columns="ISSUE_DATE")
                list_features_for_erof = list(train.columns)
                list_features_for_erof.remove("TARGET")
                list_features_for_erof.remove("DECIL")
                list_features_for_erof.remove("SCORE")
                e_r_o_f_TR = effect_range_otdel(train_, list_features_for_erof, "TARGET")
                e_r_o_f_TS = effect_range_otdel(test_, list_features_for_erof, "TARGET")
                # tab2.write([x for x in e_r_o_f_TR.Gini.tolist() if abs(x)>1 and abs(x)<=5])
                # tab2.write(len([x for x in e_r_o_f_TS.Gini.tolist() if abs(x)<=1]) > 0)

                if (
                        len([x for x in e_r_o_f_TR.Gini.tolist() if abs(x) <= 1]) > 0
                        or len([x for x in e_r_o_f_TS.Gini.tolist() if abs(x) <= 1]) > 0
                ):
                    word_erof = "Красный"
                    color_picker_erof = ":red"
                elif (
                        len([x for x in e_r_o_f_TR.Gini.tolist() if (1 < abs(x) <= 5)]) > 0
                        or len([x for x in e_r_o_f_TS.Gini.tolist() if (1 < abs(x) <= 5)])
                        > 0
                ):
                    word_erof = "Желтый"
                    color_picker_erof = ":orange"
                else:
                    word_erof = "Зеленый"
                    color_picker_erof = ":green"
                tab2.write(
                    "**М2.2 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ ДЛЯ ИНТЕПРЕТИРУЕМЫХ)**"
                    + " **РЕЗУЛЬТАТ ТЕСТА -** "
                    + color_picker_erof
                    + "[**{}**]".format(word_erof)
                )

                col1, col2 = tab2.columns(2)
                col1.header("Train")
                e_r_o_f_TR = e_r_o_f_TR.style.applymap(
                    lambda x: (
                        "background-color: red"
                        if abs(x) <= 1
                        else (
                            "background-color: orange"
                            if 1 < abs(x) <= 5
                            else "background-color: green"
                        )
                    )
                )

                col1.dataframe(e_r_o_f_TR)

                col2.header("Test")
                e_r_o_f_TS = e_r_o_f_TS.style.applymap(
                    lambda x: (
                        "background-color: red"
                        if abs(x) <= 1
                        else (
                            "background-color: orange"
                            if 1 < abs(x) <= 5
                            else "background-color: green"
                        )
                    )
                )
                col2.dataframe(e_r_o_f_TS)

                # abs_rel_df = abs_rel(train, test, list_features_for_erof, 'TARGET')
                # tab2.dataframe(abs_rel_df)
            except:
                tab2.error(
                    "**М2.2 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ ДЛЯ "
                    "ИНТЕПРЕТИРУЕМЫХ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
                )
                pass

        if "M2.3 Динамика коэффициента Джини" in options:
            try:
                data_for_dyn.columns = [x.upper() for x in data_for_dyn.columns]
                model = pd.DataFrame()
                model["target_column"] = data_for_dyn.TARGET
                model["pred_column"] = data_for_dyn.SCORE
                model["date_column"] = pd.to_datetime(
                    data_for_dyn.ISSUE_DATE, format=format_date
                )
                model["model_id"] = 1
                model["sub_model"] = 1

                mounth_dyn = gini_dynamic(model, time_slice="month")
                quart_dyn = gini_dynamic(model, time_slice="quarter")

                data_res_dyn = mounth_dyn[mounth_dyn.value_type == "gini"]
                if (
                        len([x for x in data_res_dyn.value.to_list() if x <= 0.15])
                        / len(data_res_dyn)
                        >= 0.3
                ):

                    color_picker_MD = ":red"
                elif (
                        len([x for x in data_res_dyn.value.to_list() if x <= 0.15]) > 0
                ) or (
                        (
                                len([x for x in data_res_dyn.value.to_list() if x <= 0.35])
                                / len(data_res_dyn)
                        )
                        >= 0.3
                ):

                    color_picker_MD = ":orange"

                else:
                    color_picker_MD = ":green"

                tab2.write(
                    "**М2.3 ДИНАМИКА КОЭФФИЦИЕНТА ДЖИНИ (ПОМЕСЯЧНО). (ЗНАЧИМЫЙ ТЕСТ)**"
                    + " **РЕЗУЛЬТАТ ТЕСТА -** "
                    + color_picker_MD
                    + "[**{}**]".format(res_gini_dyn(mounth_dyn))
                )
                tab2.dataframe(mounth_dyn)
                tab2.pyplot(gini_dynamic_graph(model, time_slice="month"))

                data_res_dyn_Q = quart_dyn[quart_dyn.value_type == "gini"]
                if (
                        len([x for x in data_res_dyn_Q.value.to_list() if x <= 0.15])
                        / len(data_res_dyn_Q)
                        >= 0.3
                ):
                    color_picker_QD = ":red"
                elif (
                        len([x for x in data_res_dyn_Q.value.to_list() if x <= 0.15]) > 0
                ) or (
                        (
                                len([x for x in data_res_dyn_Q.value.to_list() if x <= 0.35])
                                / len(data_res_dyn_Q)
                        )
                        >= 0.3
                ):
                    color_picker_QD = ":orange"
                else:
                    color_picker_QD = ":green"

                tab2.write(
                    "**М2.3 ДИНАМИКА КОЭФФИЦИЕНТА ДЖИНИ (ПОКВАРТАЛЬНО). (ЗНАЧИМЫЙ ТЕСТ)**"
                    + " **РЕЗУЛЬТАТ ТЕСТА -** "
                    + color_picker_QD
                    + "[**{}**]".format(res_gini_dyn(quart_dyn))
                )

                tab2.dataframe(quart_dyn)
                tab2.pyplot(gini_dynamic_graph(model, time_slice="quarter"))

            except:
                tab2.error(
                    "**М2.3 ДИНАМИКА КОЭФФИЦИЕНТА ДЖИНИ. (ЗНАЧИМЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
                )
                pass

    # if 'M3.1 Анализ корректности дискретного преобразования факторов' in options:
    #     woe_count = train.apply(lambda col: col.astype(str).str.contains('woe').any()).sum()
    #     if woe_count > 0:
    #         try:
    #             m3_1(train, test, type_block, tab2)
    #         except:
    #             if type_block == 'Корпоративный блок':
    #                 tab2.error(
    #                     "**М3.1 АНАЛИЗ КОРРЕКТНОСТИ ДИСКРЕТНОГО ПРЕОБРАЗОВАНИЯ ФАКТОРОВ (ИНФОРМАТИВНЫЙ ТЕСТ) (НЕ ПРЕМЕНИМ В НЕИНТЕРПРЕТИРУЕМЫХ МОДЕЛЯХ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**")
    #             elif type_block == 'Бизнес блок' or type_block == 'Розничный блок':
    #                 tab2.error(
    #                     "**М3.1 АНАЛИЗ КОРРЕКТНОСТИ ДИСКРЕТНОГО ПРЕОБРАЗОВАНИЯ ФАКТОРОВ (ЗНАЧИМЫЙ ТЕСТ) (НЕ ПРЕМЕНИМ В НЕИНТЕРПРЕТИРУЕМЫХ МОДЕЛЯХ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**")
    #             pass
    #     else:
    #         tab2.error("M3.1 Преобразование Woe не применялось")
    #         m3_1_color = 'Не применим'
    #         m3_1_woe = 'Преобразование Woe не применялось'

    if "M3.2 Статистическая значимость весов факторов" in options:
        try:
            m3_2_color, m3_2_df = m3_2(test, tab2, test_or_oot)
        except:
            tab2.error(
                "**М3.2 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ ВЕСОВ ФАКТОРОВ (ЗНАЧИМЫЙ ТЕСТ. НЕ ПРЕМЕНИМ В НЕИНТЕРПРЕТИРУЕМЫХ МОДЕЛЯХ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
            )
            pass

    if "M3.3 Тест на наличие мультиколлинеарности" in options:
        try:
            m3_3_color, vif_df = m3_3(test, tab2)
        except:
            tab2.error(
                "**M3.3 ТЕСТ НА НАЛИЧИЕ МУЛЬТИКОЛЛИНЕАРНОСТИ (ЗНАЧИМЫЙ ТЕСТ) ⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ТЕСТОВЫХ ДАННЫХ !**"
            )
            pass

    if type_block == "Розничный блок" or type_block == "Бизнес блок" or (
            type_block == "Корпоративный блок" and type_model == "Антифрод модель"):
        if (
                "M4.1 Сравнение прогнозного и фактического TR[2] на уровне выборки"
                in options
        ):
            try:
                m4_1_color, m4_1_df = m4_1(test, tab2)
            except:
                tab2.error(
                    "**СРАВНЕНИЕ ПРОГНОЗНОГО И ФАКТИЧЕСКОГО TR[2] НА УРОВНЕ ВЫБОРКИ (ИНФОРМАТИВНЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!!**"
                )

    elif (type_block == "Корпоративный блок" and type_model == "Скоринговая модель"):
        if (
                "M4.1 Сравнение прогнозного и фактического TR[1] на уровне выборки"
                in options
        ):
            try:

                PR_TR = prediction_fact_compare(test.TARGET, test.SCORE)
                if (
                        PR_TR.P[0] > PR_TR.interval_green[0][0]
                        and PR_TR.P[0] < PR_TR.interval_green[0][1]
                ):
                    color_picker_PR_TR = ":green"
                    word_PR_TR = "Зеленый"
                elif (
                        PR_TR.P[0] > PR_TR.interval_yellow[0][0]
                        and PR_TR.P[0] < PR_TR.interval_yellow[0][1]
                ):
                    color_picker_PR_TR = ":yellow"
                    word_PR_TR = "Желтый"
                else:
                    color_picker_PR_TR = ":red"
                    word_PR_TR = "Красный"
                tab2.write(
                    "**M4.1 СРАВНЕНИЕ ПРОГНОЗНОГО И ФАКТИЧЕСКОГО TR[1] НА УРОВНЕ ВЫБОРКИ (ИНФОРМАТИВНЫЙ ТЕСТ)-** "
                    + color_picker_PR_TR
                    + "[**{}**]".format(word_PR_TR)
                )
                tab2.dataframe(PR_TR)

            except:
                tab2.error(
                    "**M4.1 СРАВНЕНИЕ ПРОГНОЗНОГО И ФАКТИЧЕСКОГО TR[1] НА УРОВНЕ ВЫБОРКИ (ИНФОРМАТИВНЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!!**"
                )

    if "M4.2 Точность калибровочной кривой (биномиальный тест)" in options:
        try:
            m4_2_color, binom_df = m4_2_binom(train, test, tab2)
        except:
            tab2.error(
                "**М4.2 ТОЧНОСТЬ КАЛИБРОВОЧНОЙ КРИВОЙ (БИНОМИАЛЬНЫЙ ТЕСТ). (ИНФОРМАТИВНЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
            )
            pass

    if (
            "M5.1 Сравнение эффективности ранжирования модели во время разработки и во время валидации."
            in options
    ):
        try:
            m5_1_color, m5_1_df = m5_1(train, test, tab2)
        except:
            tab2.error(
                "**М5.1 СРАВНЕНИЕ ЭФФЕКТИВНОСТИ РАНЖИРОВАНИЯ МОДЕЛИ ВО ВРЕМЯ РАЗРАБОТКИ И ВО ВРЕМЯ ВАЛИДАЦИИ. (ЗНАЧИМЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
            )
            pass

    if (
            "M5.2 Сравнение эффективности ранжирования отдельных факторов модели во время разработки и во время валидации."
            in options
    ):
        try:
            m5_2_color, m5_2_df = m5_2(train, test, type_block, tab2)
        except:
            tab2.error(
                "**М5.2 СРАВНЕНИЕ ЭФФЕКТИВНОСТИ РАНЖИРОВАНИЕ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ ВО ВРЕМЯ РАЗРАБОТКИ И ВО ВРЕМЯ ВАЛИДАЦИИ. (ИНФОРМАТИВНЫЙ ТЕСТ ДЛЯ ИНТЕРПРЕТИРУЕМЫХ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!**"
            )
            pass

    if "Коэффициент корреляции" in options:
        try:
            coef_corr(train, test, tab2)
        except:
            tab2.error(
                "**КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ ПИРСОНА. (ИНФОРМАТИВНЫЙ ТЕСТ)⚠️ПРОВЕРЬТЕ ДАННЫЕ В ПОЛЕ ЗАГРУЗКИ ОБУЧАЮЩИХ ИЛИ ТЕСТОВЫХ ДАННЫХ!!**"
            )
            pass
    # if 'Доверительный интервал для коэффициента Джини методом бутстрэпа' in options:
    #     try:
    #         if 'ISSUE_DATE' in list(train.columns):
    #             train = train.drop(columns="ISSUE_DATE")
    #             test = test.drop(columns="ISSUE_DATE")
    #         X_columns_for_boots = train.columns.tolist().copy()
    #         X_columns_for_boots.remove('TARGET')
    #         X_columns_for_boots.remove('DECIL')
    #         X_columns_for_boots.remove('SCORE')
    #         data_boots = pd.concat([train, test])
    #         data_boots.columns = [x.upper() for x in data_boots.columns]
    #         conf = conf_interval_for_bootstrap_test(data_boots, 100, 80, X_columns_for_boots, 'TARGET', model_clf)
    #         # tab2.write('**Доверительный интервал для коэффициента Джини методом бутстрэпа = {}**%'.format(
    #         #     tuple([i * 100 for i in conf])))
    #
    #         if (conf['MIN'][0] + conf['MAX'][0]) * 100 / 2 < 45:
    #             color_for_boots = ':red'
    #             word_for_boots = 'Красный'
    #         elif (conf['MIN'][0] + conf['MAX'][0]) * 100 / 2 < 50:
    #             color_for_boots = ':orange'
    #             word_for_boots = 'Желтый'
    #         else:
    #             color_for_boots = ':green'
    #             word_for_boots = 'Зеленый'
    #         tab2.write(
    #             '**М2.2 ДОВИРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ КОЭФФИЦИЕНТА ДЖИНИ МЕТОДОМ БУТСТРАПА  (ЗНАЧИМЫЙ ТЕСТ).**' + ' **РЕЗУЛЬТАТ ТЕСТА -** '
    #             + color_for_boots + '[**{}**]'.format(word_for_boots))
    #         tab2.dataframe(conf)
    #     except:
    #         tab2.write(
    #             '**М2.2 ДОВИРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ КОЭФФИЦИЕНТА ДЖИНИ МЕТОДОМ БУТСТРАПА.  (ЗНАЧИМЫЙ ТЕСТ)  ⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**')

    # if 'Эффективность ранжирования с исключением отдельных факторов модели' in options:
    #     try:
    #
    #         if 'ISSUE_DATE' in list(train.columns):
    #             train = train.drop(columns="ISSUE_DATE")
    #             test = test.drop(columns="ISSUE_DATE")
    #         train_gini_for_without_test = metrics.roc_auc_score(train['TARGET'], train['SCORE']) * 2 - 1
    #         test_gini_for_without_test = metrics.roc_auc_score(test['TARGET'], test['SCORE']) * 2 - 1
    #         X_columns_for_without = train.columns.tolist().copy()
    #         y_columns_for_without = train.TARGET
    #         X_columns_for_without.remove('TARGET')
    #         X_columns_for_without.remove('DECIL')
    #         X_columns_for_without.remove('SCORE')
    #         # X_columns_for_without.remove('ISSUE_DATE')
    #
    #         range_without_feat = range_effect_without_f(train, test, X_columns_for_without, 'TARGET', model_clf,
    #                                                     train_gini_for_without_test * 100,
    #                                                     test_gini_for_without_test * 100)[0]
    #         range_without_feat111 = range_effect_without_f(train, test, X_columns_for_without, 'TARGET', model_clf,
    #                                                        train_gini_for_without_test * 100,
    #                                                        test_gini_for_without_test * 100)[1]
    #         # tab2.write(train_gini_for_without_test)
    #         # tab2.write(test_gini_for_without_test)
    #
    #         #
    #         #
    #         if [True for i, j in zip(range_without_feat111['gini_train-new_gini_train'],
    #                                  range_without_feat111['gini_test-new_gini_test']) if i > 5 and j > 5]:
    #             color_picker_ts = ':red'
    #             word_gini = 'Красный'
    #         elif [True for i, j in zip(range_without_feat111['gini_train-new_gini_train'],
    #                                    range_without_feat111['gini_test-new_gini_test']) if i > 0 and j > 0]:
    #             color_picker_ts = ':orange'
    #             word_gini = 'Желтый'
    #         else:
    #             color_picker_ts = ':green'
    #             word_gini = 'Зеленый'
    #
    #         tab2.write(
    #             '**М2.3 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ С ИСКЛЮЧЕНИЕМ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ)**' + ' **РЕЗУЛЬТАТ ТЕСТА -** '
    #             + color_picker_ts + '[**{}**]'.format(word_gini))
    #
    #         tab2.dataframe(range_without_feat)
    #
    #         # tab2.pyplot(ROC(train.SCORE, train.TARGET, 'Train'), ROC(test.SCORE, test.TARGET, 'Test'))

    # except:
    #     tab2.write(
    #         "**М2.3 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ С ИСКЛЮЧЕНИЕМ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ) "
    #         "⚠️ПРОВЕРЬТЕ КОРРЕКТНОСТЬ ВВОДИМЫХ ДАННЫХ В ПОЛЕ ИМПОРТА БИБЛТОТЕК ИЛИ ГИПРЕПАРАМЕТРОВ МОДЕЛИ!**")
    st.session_state['test_started'] = True
    return \
        psi_df, \
            m1_1_color, \
            m1_2_color, \
            m1_2_df, \
            a_empty, \
            calc_month, \
            calc_quart, \
            m1_3_color, \
            m1_4_glubina_color, \
            glubina_month, \
            data_dep, \
            m2_1_color, \
            roc_fig, \
            m2_2_color, \
            m2_2_df, \
            m2_3_color, \
            m2_3_df, \
            m2_4_color, \
            e_r_o_f_TR, \
            e_r_o_f_TS, \
            gini_dinamic_month_plot, \
            gini_dinamic_quart_plot, \
            m2_6_res, \
            m2_7_color, \
            precision_res, \
            m2_8_color, \
            recall_res, \
            m2_9_color, \
            f1_res, \
            m2_10_color, \
            error_matrix_df, \
            m2_11_color, \
            mean_log_loss, \
            m3_2_color, \
            m3_2_df, \
            m3_3_color, \
            vif_df, \
            m4_1_color, \
            m4_1_df, \
            m4_2_color, \
            binom_df, \
            m5_1_color, \
            m5_1_df, \
            m5_2_color, \
            m5_2_df


def plot_woe_bars(train_enc, train_target, test_enc, test_target, target_name, column):
    sns.set(style="whitegrid", font_scale=1.5)
    names = ["train", "test"]
    samples = []
    for test_df, target in zip([train_enc, test_enc], [train_target, test_target]):
        test_df_copy = test_df.copy().round(3)
        test_df_copy[target_name] = target
        samples.append(test_df_copy)

    samples = [
        x[[target_name, column]]
        .groupby(column)[target_name]
        .agg(["mean", "count"])
        .reset_index()
        for x in samples
    ]

    for test_df in samples:
        test_df["count"] /= test_df["count"].sum()

        test_df.rename(
            {"count": "Freq", "mean": "DefaultRate", column: "WOE: " + column},
            inplace=True,
            axis=1,
        )

    total = pd.concat(samples, axis=0, ignore_index=True)
    order = total["WOE: " + column].drop_duplicates().sort_values().values
    order = pd.Series(np.arange(order.shape[0]), index=order)

    total["_sample_"] = np.concatenate(
        [[n] * x.shape[0] for (n, x) in zip(names, samples)]
    )

    plt.figure(figsize=(10, 10))
    ax = sns.barplot(
        x="WOE: " + column,
        hue="_sample_",
        y="Freq",
        data=total,
        palette=sns.color_palette("Accent", 7),
    )
    ax2 = ax.twinx()

    for test_df, name in zip(samples, names):
        test_df.set_index(test_df["WOE: " + column].map(order).values)[
            "DefaultRate"
        ].plot(ax=ax2, label=name, marker="x")
    ax2.legend(title="_sample_")

    plt.show()


def calculate_missing_percentage_by_month(data, date_col):
    """
    Подсчитывает долю пропусков в переменных по месяцам и строит линейный график.

    Параметры:
    dataframe (pd.DataFrame): Таблица данных с переменными и датами.

    Возвращает:
    None (строит график)
    """
    data[date_col] = pd.to_datetime(data[date_col].tolist())
    # Группировка данных по месяцам и подсчет доли пропусков
    data["date_col_m"] = data[date_col].astype("datetime64[M]")
    missing_percentages = data.groupby("date_col_m").apply(lambda x: x.isnull().mean())

    nan_col = data.isna().sum().sort_values(ascending=False).index
    nan_bigger_red, mean_all = [], []
    nan_bigger_yel, mean_top = [], []
    cols = missing_percentages.columns.tolist()
    cols.remove("date_col_m")
    new_df = pd.DataFrame({"features": cols})
    for i in cols:
        if (
                missing_percentages.sort_values(i, ascending=False)[i]
                        .head(int(len(missing_percentages) * 0.3))
                        .mean()
                - missing_percentages[i].mean()
                >= 0.2
        ):
            nan_bigger_red.append(i)
        if (
                missing_percentages.sort_values(i, ascending=False)[i]
                        .head(int(len(missing_percentages) * 0.3))
                        .mean()
                - missing_percentages[i].mean()
                >= 0.1
        ):
            nan_bigger_yel.append(i)

        mean_top.append(
            round(
                missing_percentages.sort_values(i, ascending=False)[i]
                .head(int(len(missing_percentages) * 0.3))
                .mean(),
                2,
            )
        )
        mean_all.append(round(missing_percentages[i].mean(), 2))

    new_df["mean_all_missed_date"] = mean_all
    new_df["mean_top_30%_missed_date"] = mean_top

    new_df["color"] = mean_all

    for i in range(len(new_df)):
        if new_df.features[i] in nan_bigger_red:
            new_df.color[i] = "red"
        elif new_df.features[i] in nan_bigger_yel:
            new_df.color[i] = "yellow"
        else:
            new_df.color[i] = "green"

    if "red" in new_df['color'].values:
        color_picker = "Красный"
    elif "yellow" in new_df['color'].values:
        color_picker = "Желтый"
    else:
        color_picker = "Зеленый"

    return (
        new_df.sort_values("color", ascending=False).style.applymap(
            lambda x: (
                "background-color: red"
                if x == "red"
                else (
                    "background-color: orange"
                    if x == "yellow"
                    else "background-color: green"
                )
            ),
            subset=["color"],
        ),
        missing_percentages,
        nan_col,
        color_picker,
    )


def calculate_missing_percentage_by_quart(data, date_col):
    """
    Подсчитывает долю пропусков в переменных по месяцам и строит линейный график.

    Параметры:
    dataframe (pd.DataFrame): Таблица данных с переменными и датами.

    Возвращает:
    None (строит график)
    """
    data[date_col] = pd.to_datetime(data[date_col].tolist())

    # Группировка данных по месяцам и подсчет доли пропусков
    data["date_col_q"] = pd.PeriodIndex(data[date_col], freq="Q")
    missing_percentages = data.groupby("date_col_q").apply(lambda x: x.isnull().mean())

    nan_col = data.isna().sum().sort_values(ascending=False).index
    nan_bigger_red, mean_all = [], []
    nan_bigger_yel, mean_top = [], []
    cols = missing_percentages.columns.tolist()
    cols.remove("date_col_q")
    new_df = pd.DataFrame({"features": cols})
    for i in cols:
        if (
                missing_percentages.sort_values(i, ascending=False)[i]
                        .head(int(len(missing_percentages) * 0.3))
                        .mean()
                - missing_percentages[i].mean()
                >= 0.2
        ):
            nan_bigger_red.append(i)
        if (
                missing_percentages.sort_values(i, ascending=False)[i]
                        .head(int(len(missing_percentages) * 0.3))
                        .mean()
                - missing_percentages[i].mean()
                >= 0.1
        ):
            nan_bigger_yel.append(i)

        mean_top.append(
            round(
                missing_percentages.sort_values(i, ascending=False)[i]
                .head(int(len(missing_percentages) * 0.3))
                .mean(),
                2,
            )
        )
        mean_all.append(round(missing_percentages[i].mean(), 2))

    new_df["mean_all_missed_date"] = mean_all
    new_df["mean_top_30%_missed_date"] = mean_top

    new_df["color"] = mean_all

    for i in range(len(new_df)):
        if new_df.features[i] in nan_bigger_red:
            new_df.color[i] = "red"
        elif new_df.features[i] in nan_bigger_yel:
            new_df.color[i] = "yellow"
        else:
            new_df.color[i] = "green"

    return (
        new_df.sort_values("color", ascending=False).style.applymap(
            lambda x: (
                "background-color: red"
                if x == "red"
                else (
                    "background-color: orange"
                    if x == "yellow"
                    else "background-color: green"
                )
            ),
            subset=["color"],
        ),
        missing_percentages,
        nan_col,
    )


def calc_nan_plot(a, b, file_name):
    # Create a figure object
    fig, ax = plt.subplots(figsize=(20, 10))

    # Loop through each item in b, excluding the last one
    for i in b[:-1]:
        a[i].plot(ax=ax, title="Nans")
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left")
        ax.set_xlabel("Отчётная дата")
        ax.set_ylabel("Доля пропусков")

    file_path = f"test_results/{file_name}"
    fig.savefig(file_path, bbox_inches='tight')
    # Return the figure object
    return fig


# Функция для проверки эффективности ранжирования без отдельных признаков


def range_effect_without_f(
        train, test, features_columns, target_column, model, gini_train, gini_test
):
    """
    Функция принимает на вход тестовую и обучающую выборку, наименования факторов, наименование целевой переменной, модель.
    Расчеты джини без отдельных факторов, разница между фактическим джини и расчитанным без фактора собираются в таблицу,
    строится график по тестовым и обучающим данным. Светофор выставляется по значениям разницы.

    !!!Необходимо создать копию модели для точных результатов!!!
    """
    featx = features_columns
    featy = target_column

    X_train_temp = pd.DataFrame(train, columns=featx)
    X_test_temp = pd.DataFrame(test, columns=featx)
    gini = pd.DataFrame(
        index=featx, columns=["Gini without feature train", "Gini without feature test"]
    )
    # Посмотрим gini для каждой переменной отдельно
    for feature in featx:
        s = 0
        estimator = model
        estimator.fit(
            train[list(filter(functools.partial(operator.ne, feature), X_train_temp))],
            train[featy],
        )
        try:
            lr_tr_pp = estimator.predict_proba(
                train[
                    list(filter(functools.partial(operator.ne, feature), X_train_temp))
                ]
            )[:, 1]
            lr_ts_pp = estimator.predict_proba(
                test[list(filter(functools.partial(operator.ne, feature), X_test_temp))]
            )[:, 1]
        except:
            lr_tr_pp = estimator.predict(
                train[
                    list(filter(functools.partial(operator.ne, feature), X_train_temp))
                ]
            )
            lr_ts_pp = estimator.predict(
                test[list(filter(functools.partial(operator.ne, feature), X_test_temp))]
            )
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(train[featy], lr_tr_pp)
        fpr_ts, tpr_ts, thresholds_ts = roc_curve(test[featy], lr_ts_pp)
        roc_auc_tr = auc(fpr_tr, tpr_tr)
        gini_tr_lr = 2 * roc_auc_tr - 1
        gini.loc[feature, "Gini without feature train"] = gini_tr_lr
        roc_auc_ts = auc(fpr_ts, tpr_ts)
        gini_ts_lr = 2 * roc_auc_ts - 1
        gini.loc[feature, "Gini without feature test"] = gini_ts_lr
    gini = gini.sort_values("Gini without feature train", ascending=False)

    gini["Gini without feature train"] = gini["Gini without feature train"] * 100
    gini["Gini without feature test"] = gini["Gini without feature test"] * 100
    gini["gini_train-new_gini_train"] = gini["Gini without feature train"] - gini_train
    gini["gini_test-new_gini_test"] = gini["Gini without feature test"] - gini_test

    X = gini.index
    train = gini["Gini without feature train"]
    test = gini["Gini without feature test"]

    X_axis = np.arange(len(X))

    # plt.figure(figsize=(10, 5))
    # plt.bar(X_axis - 0.2, train, 0.4, label='Train')
    # plt.bar(X_axis + 0.2, test, 0.4, label='Test')
    # plt.xticks(X_axis, X, rotation=90)
    # plt.xlabel("Features")
    # plt.ylabel("Gini")
    # plt.title("Gini without factors")
    # plt.legend()
    # plt.show()

    numeric_columns = ["gini_train-new_gini_train", "gini_test-new_gini_test"]

    def highlight(
            gini, col1="gini_train-new_gini_train", col2="gini_test-new_gini_test"
    ):
        ret = pd.DataFrame("", index=gini.index, columns=gini.columns)
        ret.loc[gini["gini_train-new_gini_train"] >= 5, "gini_train-new_gini_train"] = (
            "background-color: red"
        )
        ret.loc[
            (gini["gini_train-new_gini_train"] > 0)
            | (gini["gini_train-new_gini_train"] < 5),
            "gini_train-new_gini_train",
        ] = "background-color: yellow"
        ret.loc[gini["gini_train-new_gini_train"] <= 0, "gini_train-new_gini_train"] = (
            "background-color: green"
        )

        ret.loc[gini["gini_test-new_gini_test"] >= 5, "gini_test-new_gini_test"] = (
            "background-color: red"
        )
        ret.loc[
            (gini["gini_test-new_gini_test"] > 0)
            | (gini["gini_test-new_gini_test"] < 5),
            "gini_test-new_gini_test",
        ] = "background-color: yellow"
        ret.loc[gini["gini_test-new_gini_test"] <= 0, "gini_test-new_gini_test"] = (
            "background-color: green"
        )
        return ret

    gini111 = pd.DataFrame(gini)
    gini = gini.style.apply(
        highlight,
        col1="gini_train-new_gini_train",
        col2="gini_test-new_gini_test",
        axis=None,
    )

    return gini, gini111


def plot_diff_gini(df):
    X = df.index
    train = df["Gini train"]
    test = df["Gini validation"]

    X_axis = np.arange(len(X))
    fig, ax = plt.subplots(figsize=(10, 5))  # Создаем объект fig и ось ax
    ax.bar(X_axis - 0.2, train, 0.4, label="Train")
    ax.bar(X_axis + 0.2, test, 0.4, label="Test")
    ax.set_xticks(X_axis)
    ax.set_xticklabels(X, rotation=90)
    ax.set_xlabel("Features")
    ax.set_ylabel("Gini")
    ax.set_title("Gini many factors")
    ax.legend()
    fig.savefig("test_results/gini_many_factors", bbox_inches="tight")
    return fig  # Возвращаем объект fig


def plot_diff_gini_all(df):
    X = df.index
    train = df["Train"]
    test = df["Test"]

    X_axis = np.arange(len(X))
    plt.figure(figsize=(10, 5))
    plt.bar(X_axis - 0.2, train, 0.4, label="Train")
    plt.bar(X_axis + 0.2, test, 0.4, label="Test")
    plt.xticks(X_axis, X, rotation=90)
    plt.xlabel("features")
    plt.ylabel("Gini")
    plt.title("Gini many factors")
    plt.show()


# Функция сравнения прогнозного и фактического TR


def prediction_fact_compare(y_test, predictions):
    """
    Функция принимает на вход тестовые данные, трагет по тесту и по трейну и наименование столбца с прогнозными значениями.
    На выходе получаем таблицу с фактическим таргет рейтом, прогнозным и доверительным интервалом для тестовой выборки.
    """

    ts = round((y_test.mean()), 4)
    preds = round(np.mean(predictions), 4)

    conf_int_y = [0.01, 0.99]
    conf_int_g = [0.05, 0.95]
    interval_y = scipy.stats.binom.ppf(conf_int_y, len(y_test), ts) / len(y_test)
    interval_g = scipy.stats.binom.ppf(conf_int_g, len(y_test), ts) / len(y_test)
    interval_y[0] = round(interval_y[0], 4)
    interval_g[0] = round(interval_g[0], 4)

    interval_y[1] = round(interval_y[1], 4)
    interval_g[1] = round(interval_g[1], 4)

    table = pd.DataFrame(
        {
            "TR": [ts],
            "P": [preds],
            "interval_yellow": [interval_y],
            "interval_green": [interval_g],
        }
    )

    return table


# Функция для определения стат значимости


def wald(
        train: pd.DataFrame, test: pd.DataFrame, target_col: str, cols: list
) -> pd.DataFrame():
    x = train[cols]
    y = train[[target_col]]
    x = sm.add_constant(x)
    fit = sm.Logit(y, x).fit(
        disp=0,
        C=0.008,
        class_weight="balanced",
        max_iter=1000,
        penalty="l2",
        random_state=321,
        solver="liblinear",
    )
    result_summary = fit.summary()
    summary = pd.read_html(result_summary.tables[1].as_html(), header=0, index_col=0)[0]

    x1 = test[cols]
    y1 = test[[target_col]]
    x1 = sm.add_constant(x1)
    fit1 = sm.Logit(y1, x1).fit(
        disp=0,
        C=0.008,
        class_weight="balanced",
        max_iter=1000,
        penalty="l2",
        random_state=321,
        solver="liblinear",
    )
    result_summary1 = fit1.summary()
    summary1 = pd.read_html(result_summary1.tables[1].as_html(), header=0, index_col=0)[
        0
    ]

    return_df = pd.DataFrame(
        columns=[
            "Фактор",
            "Величина коэффициента для выборки разработки",
            "Величина коэффициента для выборки валидации",
            "p-value для выборки разработки",
            "p-value для выборки валидации",
            "Смена знака коэффициента",
        ]
    )
    return_df["Фактор"] = summary.index
    return_df["Величина коэффициента для выборки разработки"] = summary.coef.values
    return_df["Величина коэффициента для выборки валидации"] = summary1.coef.values
    return_df["p-value для выборки разработки"] = summary["P>|z|"].values
    return_df["p-value для выборки валидации"] = summary1["P>|z|"].values

    return_df["Смена знака коэффициента"] = [
        "Нет" if row1 * row2 > 0 else "Да"
        for row1, row2 in zip(
            return_df["Величина коэффициента для выборки разработки"],
            return_df["Величина коэффициента для выборки валидации"],
        )
    ]
    return_df["Результат"] = return_df["Смена знака коэффициента"]
    for i in range(len(return_df)):
        if (
                return_df["p-value для выборки разработки"][i] > 0.1
                or return_df["p-value для выборки валидации"][i] > 0.1
                or return_df["Смена знака коэффициента"][i] == "Да"
        ):
            return_df["Результат"][i] = "Красный"
        elif (
                return_df["p-value для выборки разработки"][i] > 0.05
                or return_df["p-value для выборки валидации"][i] > 0.05
        ):
            return_df["Результат"][i] = "Желтый"
        else:
            return_df["Результат"][i] = "Зеленый"

    if "Красный" in return_df["Результат"].tolist():
        color_picker_wald = ":red"
        word_wald = "Красный"
    elif (
            len([x for x in return_df["Результат"] if x == "Желтый"]) / len(return_df) > 0.1
    ):
        color_picker_wald = ":yellow"
        word_wald = "Желтый"
    else:
        color_picker_wald = ":green"
        word_wald = "Зеленый"

    def color(row):
        highlight1 = "background-color: green;"
        highlight2 = "background-color: yellow;"
        default = "background-color: red"

        for index in row.index:
            if 0.095 < row[index] < 0.10:
                return [highlight2]
            elif row[index] <= 0.095:
                return [highlight1]
            else:
                return [default]

    def color_col(col):
        highlight1 = "background-color: green;"
        highlight2 = "background-color: red;"
        if col["Смена знака коэффициента"] == "Да":
            return [highlight2]
        else:
            return [highlight1]

    return_df = return_df.sort_values(
        by=["p-value для выборки разработки"], ascending=False
    )
    return_df = return_df[return_df["Фактор"] != "const"]
    return_df.reset_index(drop=True, inplace=True)
    return (
        return_df.style.apply(color, subset=["p-value для выборки разработки"], axis=1)
        .apply(color, subset=["p-value для выборки валидации"], axis=1)
        .apply(color_col, subset=["Смена знака коэффициента"], axis=1)
        .format(
            {
                "p-value для выборки разработки": "{:.2%}",
                "p-value для выборки валидации": "{:.2%}",
            }
        ),
        color_picker_wald,
        word_wald,
    )


def wald_oot(
        train: pd.DataFrame, test: pd.DataFrame, target_col: str, cols: list
) -> pd.DataFrame():
    x = train[cols]
    y = train[[target_col]]
    x = sm.add_constant(x)
    fit = sm.Logit(y, x).fit(
        disp=0,
        C=0.008,
        class_weight="balanced",
        max_iter=1000,
        penalty="l2",
        random_state=321,
        solver="liblinear",
    )
    result_summary = fit.summary()
    summary = pd.read_html(result_summary.tables[1].as_html(), header=0, index_col=0)[0]

    x1 = test[cols]
    y1 = test[[target_col]]
    x1 = sm.add_constant(x1)
    fit1 = sm.Logit(y1, x1).fit(
        disp=0,
        C=0.008,
        class_weight="balanced",
        max_iter=1000,
        penalty="l2",
        random_state=321,
        solver="liblinear",
    )
    result_summary1 = fit1.summary()
    summary1 = pd.read_html(result_summary1.tables[1].as_html(), header=0, index_col=0)[
        0
    ]

    return_df = pd.DataFrame(
        columns=[
            "Фактор",
            "Величина коэффициента для выборки разработки",
            "Величина коэффициента для выборки валидации",
            "p-value для выборки разработки",
            "p-value для выборки валидации",
            "Смена знака коэффициента",
        ]
    )
    return_df["Фактор"] = summary.index
    return_df["Величина коэффициента для выборки разработки"] = summary.coef.values
    return_df["Величина коэффициента для выборки валидации"] = summary1.coef.values
    return_df["p-value для выборки разработки"] = summary["P>|z|"].values
    return_df["p-value для выборки валидации"] = summary1["P>|z|"].values

    return_df["Смена знака коэффициента"] = [
        "Нет" if row1 * row2 > 0 else "Да"
        for row1, row2 in zip(
            return_df["Величина коэффициента для выборки разработки"],
            return_df["Величина коэффициента для выборки валидации"],
        )
    ]
    return_df["Результат"] = return_df["Смена знака коэффициента"]
    for i in range(len(return_df)):
        if (
                return_df["p-value для выборки разработки"][i] > 0.1
                or return_df["p-value для выборки валидации"][i] > 0.1
                or return_df["Смена знака коэффициента"][i] == "Да"
        ):
            return_df["Результат"][i] = "Красный"
        elif (
                return_df["p-value для выборки разработки"][i] > 0.05
                or return_df["p-value для выборки валидации"][i] > 0.05
        ):
            return_df["Результат"][i] = "Желтый"
        else:
            return_df["Результат"][i] = "Зеленый"

    if (
            len([x for x in return_df["Результат"] if x == "Красный"]) / len(return_df)
            >= 0.2
    ):
        color_picker_wald = ":red"
        word_wald = "Красный"
    elif (
            len([x for x in return_df["Результат"] if x == "Красный"]) / len(return_df)
            < 0.2
            or len([x for x in return_df["Результат"] if x == "Желтый"]) / len(return_df)
            >= 0.2
    ):
        color_picker_wald = ":yellow"
        word_wald = "Желтый"
    else:
        color_picker_wald = ":green"
        word_wald = "Зеленый"

    def color(row):
        highlight1 = "background-color: green;"
        highlight2 = "background-color: yellow;"
        default = "background-color: red"

        for index in row.index:
            if 0.095 < row[index] < 0.10:
                return [highlight2]
            elif row[index] <= 0.095:
                return [highlight1]
            else:
                return [default]

    def color_col(col):
        highlight1 = "background-color: green;"
        highlight2 = "background-color: red;"
        if col["Смена знака коэффициента"] == "Да":
            return [highlight2]
        else:
            return [highlight1]

    return_df = return_df.sort_values(
        by=["p-value для выборки разработки"], ascending=False
    )
    return_df = return_df[return_df["Фактор"] != "const"]
    return_df.reset_index(drop=True, inplace=True)
    return (
        return_df.style.apply(color, subset=["p-value для выборки разработки"], axis=1)
        .apply(color, subset=["p-value для выборки валидации"], axis=1)
        .apply(color_col, subset=["Смена знака коэффициента"], axis=1)
        .format(
            {
                "p-value для выборки разработки": "{:.2%}",
                "p-value для выборки валидации": "{:.2%}",
            }
        ),
        color_picker_wald,
        word_wald,
    )


def effect_range_otdel(df, featx, featy):
    # df = df.dropna().reset_index(drop=True)
    lr = LogisticRegression()
    lr.fit(df[featx], df[featy])
    clf = lr
    X_train_temp = pd.DataFrame(df, columns=featx)
    gini = pd.DataFrame(index=featx, columns=["Gini"])
    # Посмотрим gini для каждой переменной отдельно
    for feature in featx:
        s = 0
        estimator = clf
        estimator.fit(df[feature].values.reshape(-1, 1), df[featy])
        lr_tr_pp = estimator.predict_proba(df[feature].values.reshape(-1, 1))[:, 1]
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(df[featy], lr_tr_pp)
        roc_auc_tr = auc(fpr_tr, tpr_tr)
        gini_tr_lr = 2 * roc_auc_tr - 1
        gini.loc[feature, "Gini"] = gini_tr_lr

    gini = gini.sort_values("Gini", ascending=False)
    gini.Gini = gini.Gini * 100

    gini = pd.DataFrame(gini)
    return gini


def abs_rel(tr, ts, featx, featy, type_block):
    trainn = effect_range_otdel(tr, featx, featy)
    test = effect_range_otdel(ts, featx, featy)
    trainn["Gini validation"] = test.Gini
    trainn["Абсолютное изменение"] = abs(trainn["Gini validation"] - trainn.Gini)
    trainn["Относительное изменение"] = (
                                                trainn["Абсолютное изменение"] / trainn.Gini
                                        ) * 100
    chto = []
    for i, j in zip(trainn["Абсолютное изменение"], trainn["Относительное изменение"]):
        if type_block == "Розничный блок" or type_block == "Бизнес блок":
            if i >= 10 and j >= 0.2:
                chto.append("red")
            elif i >= 5 and j >= 0.15:
                chto.append("yellow")
            else:
                chto.append("green")
        elif type_block == "Корпоративный блок":
            if i >= 10 and j >= 0.2:
                chto.append("red")
            elif i >= 5 and j >= 0.1:
                chto.append("yellow")
            else:
                chto.append("green")
    trainn["Оценка"] = chto
    trainn = trainn.rename(columns={"Gini": "Gini train"})
    return trainn


def gini_dynamic(
        df,
        sample_type="train",
        time_slice: str = "month",
        graph: bool = True,
        stat: bool = True,
):
    target_column = "target_column"
    prediction_column = "pred_column"
    date_column = "date_column"
    model_id = "model_id"
    sub_model = "sub_model"
    sample_type = sample_type

    # добавим ассигн
    if time_slice not in ["month", "quarter", "year"]:
        raise ValueError(
            'Переменная time_slice может принимать только два значения: "month", "quarter". Получено {} взамен'
        )

    data = df.copy(deep=True)
    a = []
    Gini = []
    conf_up = []
    conf_low = []
    count_15 = 0
    count_35 = 0
    length = []
    shapes = []
    sum_targets = []
    # сортируем датафрейм по датам, необходимо для последовательного результата
    data = data.sort_values([date_column])
    # Создание новой колонки Gini_1 = год+месяц/квартал
    if time_slice == "month":
        data.loc[data[date_column].dt.month > 9, "Gini_1"] = (
                data[date_column].dt.year.astype(str)
                + "_"
                + data[date_column].dt.month.astype(str)
        )
        data.loc[data[date_column].dt.month < 10, "Gini_1"] = (
                data[date_column].dt.year.astype(str)
                + "_0"
                + data[date_column].dt.month.astype(str)
        )
    elif time_slice == "quarter":
        data["Gini_1"] = (
                data[date_column].dt.year.astype(str)
                + "_"
                + data[date_column].dt.quarter.astype(str)
        )
    # Список значений Gini_1
    array_of_unique = data["Gini_1"].unique()
    array_of_unique = sorted(array_of_unique)
    for i, j in enumerate(array_of_unique):
        # Разбиение выборки на бакеты
        a.append(data[data["Gini_1"] == j])
        # Размеры каждой выборки
        shapes.append(a[i].shape[0])
        sum_targets.append(a[i][a[i][target_column] == 1].shape[0])  #
        try:
            AUC = roc_auc_score(a[i][target_column], y_score=a[i][prediction_column])
        except:
            AUC = 0
        Gini.append(2 * AUC - 1)
        n_a = a[i][a[i][target_column] == 1].shape[0]
        n_n = a[i][a[i][target_column] == 0].shape[0]
        Q1 = AUC / (2 - AUC)
        Q2 = 2 * (AUC ** 2) / (1 + AUC)
        try:
            SD = (
                         (
                                 (
                                         AUC * (1 - AUC)
                                         + (n_a - 1) * (Q1 - (AUC ** 2) + (n_n - 1) * (Q2 - (AUC) ** 2))
                                 )
                                 / (n_a * n_n)
                         )
                         ** 0.5
                 ) * 2
        except:
            SD = 0
        conf_up.append(Gini[i] + (SD * 1.96) / (a[i].shape[0] ** 0.5))
        conf_low.append(Gini[i] - (SD * 1.96) / (a[i].shape[0] ** 0.5))
        length.append(float(a[i].shape[0]))

    # График
    if graph:
        fig, ax2 = plt.subplots(1)
        fig.set_figheight(4)
        fig.set_figwidth(12)
        plt.xticks(range(1, len(array_of_unique) + 1), array_of_unique, rotation=90)

        # Наблюдения
        ax2.fill_between(range(1, len(length) + 1), length, color="mediumseagreen")
        ax2.set_ylabel("Observations number", fontsize=14)  # Ось Oy
        ax = ax2.twinx()
        # Gini
        ax.plot(range(1, len(Gini) + 1), Gini, lw=2, color="blue")
        for i in range(len(Gini)):
            ax.plot(
                i + 1, Gini[i], "ro", color="blue", markersize=5
            )  # Круг, значение p-value<0.1
            ax.plot(
                i + 1, conf_up[i], "ro", color="orange", markersize=5
            )  # Круг, значение p-value<0.1
            ax.plot(
                i + 1, conf_low[i], "ro", color="orange", markersize=5
            )  # Круг, значение p-value<0.1
        ax.plot(range(1, len(conf_up) + 1), conf_up, lw=2, color="orange")
        ax.plot(range(1, len(conf_up) + 1), conf_low, lw=2, color="orange")
        ax.plot(
            [0, len(Gini) + 1], [0.30] * 2, linestyle="dashed", color="red", lw=2
        )  # пунктир p-value<0.1
        ax.plot(
            [0, len(Gini) + 1], [0.40] * 2, linestyle="dashed", color="gold", lw=2
        )  # пунктир p-value<0.1
        # Параметры графика
        ax.set_xlim(0, len(Gini) + 1)  # Масштаб
        ax.set_ylim(0, max(Gini) + 0.1)  # Масштаб
        ax.set_xlabel("Time range slice", fontsize=14)  # Ось Oy
        plt.grid()  # Линейка на фоне
        ax.set_ylabel("Коэффицент Gini", fontsize=14)  # Ось Oy
        ax.set_title("Gini in dynamics")

    # Светофор
    for i in range(len(Gini)):
        if Gini[i] != -1:
            if Gini[i] < 0.30:
                count_15 += 1
            if Gini[i] < 0.40:
                count_35 += 1
    if count_15 > len(Gini) * 0.3:
        # print('Красный')
        ...
    elif count_15 > 0 or count_35 > len(Gini) * 0.3:
        # print('Желтый')
        ...
    else:
        # print('Зеленый')
        ...

    array_of_unique_of_unique = pd.to_datetime(
        list(map(lambda x: x.replace("_", "-") + "-01", array_of_unique))
    )
    return_df = pd.DataFrame(
        columns=[
            "id",
            "test_type",
            "sample_type",
            "report_date",
            "date",
            "value_type",
            "value",
        ]
    )
    local_df = pd.DataFrame(
        columns=[
            "id",
            "test_type",
            "sample_type",
            "report_date",
            "date",
            "value_type",
            "value",
        ]
    )
    local_df["date"] = array_of_unique
    local_df["value"] = shapes
    local_df["value_type"] = "size"
    return_df = return_df.append(local_df)

    local_df = pd.DataFrame(
        columns=[
            "id",
            "test_type",
            "sample_type",
            "report_date",
            "date",
            "value_type",
            "value",
        ]
    )
    local_df["date"] = array_of_unique
    local_df["value"] = Gini
    local_df["value_type"] = "gini"
    return_df = return_df.append(local_df)

    local_df = pd.DataFrame(
        columns=[
            "id",
            "test_type",
            "sample_type",
            "report_date",
            "date",
            "value_type",
            "value",
        ]
    )
    local_df["date"] = array_of_unique
    local_df["value"] = conf_up
    local_df["value_type"] = "yellow_upper"
    return_df = return_df.append(local_df)

    local_df = pd.DataFrame(
        columns=[
            "id",
            "test_type",
            "sample_type",
            "report_date",
            "date",
            "value_type",
            "value",
        ]
    )
    local_df["date"] = array_of_unique
    local_df["value"] = conf_low
    local_df["value_type"] = "yellow_lower"
    return_df = return_df.append(local_df)

    return_df["id"] = model_id
    return_df["test_type"] = "gini_dynamic"
    return_df["sample_type"] = sample_type
    return_df["report_date"] = dt.date.today()
    return_df["sub_model"] = 1
    return return_df


def res_gini_dyn(df):
    df = df[df.value_type == "gini"]
    if len([x for x in df.value.to_list() if x <= 0.15]) / len(df) >= 0.3:
        return "Красный"
    elif (
            len([x for x in df.value.to_list() if x <= 0.15]) > 0
            or (len([x for x in df.value.to_list() if x <= 0.35]) / len(df)) >= 0.3
    ):
        return "Желтый"
    else:
        return "Зеленый"


def gini_dynamic_graph(df, sample_type="train", time_slice: str = "month", graph: bool = True, stat: bool = True,
                       figname=None):
    target_column = "target_column"
    prediction_column = "pred_column"
    date_column = "date_column"
    model_id = "model_id"
    sub_model = "sub_model"
    sample_type = sample_type

    # добавим ассигн
    if time_slice not in ["month", "quarter", "year"]:
        raise ValueError(
            'Переменная time_slice может принимать только два значения: "month", "quarter". Получено {} взамен'
        )

    data = df.copy(deep=True)
    a = []
    Gini = []
    conf_up = []
    conf_low = []
    count_15 = 0
    count_35 = 0
    length = []
    shapes = []
    sum_targets = []
    # сортируем датафрейм по датам, необходимо для последовательного результата
    data = data.sort_values([date_column])
    # Создание новой колонки Gini_1 = год+месяц/квартал
    if time_slice == "month":
        data.loc[data[date_column].dt.month > 9, "Gini_1"] = (
                data[date_column].dt.year.astype(str)
                + "_"
                + data[date_column].dt.month.astype(str)
        )
        data.loc[data[date_column].dt.month < 10, "Gini_1"] = (
                data[date_column].dt.year.astype(str)
                + "_0"
                + data[date_column].dt.month.astype(str)
        )
    elif time_slice == "quarter":
        data["Gini_1"] = (
                data[date_column].dt.year.astype(str)
                + "_"
                + data[date_column].dt.quarter.astype(str)
        )
    # Список значений Gini_1
    array_of_unique = data["Gini_1"].unique()
    array_of_unique = sorted(array_of_unique)
    for i, j in enumerate(array_of_unique):
        # Разбиение выборки на бакеты
        a.append(data[data["Gini_1"] == j])
        # Размеры каждой выборки
        shapes.append(a[i].shape[0])
        sum_targets.append(a[i][a[i][target_column] == 1].shape[0])  #
        try:
            AUC = roc_auc_score(a[i][target_column], y_score=a[i][prediction_column])
        except:
            AUC = 0
        Gini.append(2 * AUC - 1)
        n_a = a[i][a[i][target_column] == 1].shape[0]
        n_n = a[i][a[i][target_column] == 0].shape[0]
        Q1 = AUC / (2 - AUC)
        Q2 = 2 * (AUC ** 2) / (1 + AUC)
        try:
            SD = (
                         (
                                 (
                                         AUC * (1 - AUC)
                                         + (n_a - 1) * (Q1 - (AUC ** 2) + (n_n - 1) * (Q2 - (AUC) ** 2))
                                 )
                                 / (n_a * n_n)
                         )
                         ** 0.5
                 ) * 2
        except:
            SD = 0
        conf_up.append(Gini[i] + (SD * 1.96) / (a[i].shape[0] ** 0.5))
        conf_low.append(Gini[i] - (SD * 1.96) / (a[i].shape[0] ** 0.5))
        length.append(float(a[i].shape[0]))

    # График
    if graph:
        fig, ax2 = plt.subplots(1)
        fig.set_figheight(4)
        fig.set_figwidth(12)
        plt.xticks(range(1, len(array_of_unique) + 1), array_of_unique, rotation=90)

        # Наблюдения
        ax2.fill_between(range(1, len(length) + 1), length, color="mediumseagreen")
        ax2.set_ylabel("Observations number", fontsize=14)  # Ось Oy
        ax = ax2.twinx()
        # Gini
        ax.plot(range(1, len(Gini) + 1), Gini, lw=2, color="blue")
        for i in range(len(Gini)):
            ax.plot(
                i + 1, Gini[i], "ro", color="blue", markersize=5
            )  # Круг, значение p-value<0.1
            ax.plot(
                i + 1, conf_up[i], "ro", color="orange", markersize=5
            )  # Круг, значение p-value<0.1
            ax.plot(
                i + 1, conf_low[i], "ro", color="orange", markersize=5
            )  # Круг, значение p-value<0.1
        ax.plot(range(1, len(conf_up) + 1), conf_up, lw=2, color="orange")
        ax.plot(range(1, len(conf_up) + 1), conf_low, lw=2, color="orange")
        ax.plot(
            [0, len(Gini) + 1], [0.30] * 2, linestyle="dashed", color="red", lw=2
        )  # пунктир p-value<0.1
        ax.plot(
            [0, len(Gini) + 1], [0.40] * 2, linestyle="dashed", color="gold", lw=2
        )  # пунктир p-value<0.1
        # Параметры графика
        ax.set_xlim(0, len(Gini) + 1)  # Масштаб
        ax.set_ylim(0, max(Gini) + 0.1)  # Масштаб
        ax.set_xlabel("Time range slice", fontsize=14)  # Ось Oy
        plt.grid()  # Линейка на фоне
        ax.set_ylabel("Коэффицент Gini", fontsize=14)  # Ось Oy
        ax.set_title("Gini in dynamics")
        fig.savefig(f"test_results/{figname}", bbox_inches='tight')


def res_actual(num):
    if num >= 25:
        return "Красный"
    elif 15 <= num < 25:
        return "Желтый"
    else:
        return "Зеленый"


def ROC(scores_labels, title="ROC Curve"):
    """
    Функция принимает на вход:
    - scores_labels: список кортежей, каждый из которых содержит (Score, Y, label),
      где 'Score' это массив с оценками вероятностей класса,
      'Y' это массив с фактическими метками классов,
      'label' это название набора данных для легенды (например, 'Train' или 'Test').
    - title: заголовок графика.

    Функция возвращает объект фигуры с графиком кривых ROC.
    """
    fig, ax = plt.subplots()

    # Отрисовка диагональной линии для справки
    ax.plot([0, 1], [0, 1], "k-.", label='Random (Gini=0)')

    for score, Y, label in scores_labels:
        # Вычисление значений для ROC-кривой
        fpr, tpr, _ = roc_curve(Y, score)
        gini_score = (roc_auc_score(Y, score) * 2 - 1) * 100
        ax.plot(
            fpr,
            tpr,
            label=f'{label}: Gini={gini_score:.3f}'
        )

    # Настройка графика
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.legend(loc='lower right')
    ax.grid(True)
    file_path = f"test_results/roc_curve"
    fig.savefig(file_path, bbox_inches='tight')
    return fig


def decils_plot(data, decil_f, target_f, name=None):
    # data = data.dropna().reset_index(drop=True)
    a = pd.crosstab(data[decil_f], data[target_f])
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    dec_uq = data[decil_f].dropna().unique()
    if (data[target_f] == -1).any():
        a["Rate"] = a[1] / (a[0] + a[1])
        ax.bar(
            range(int(min(dec_uq)), int(max(dec_uq)) + 1),
            a[0],
            color="orange",
            label="Good",
            edgecolor="black",
        )
        ax.bar(
            range(int(min(dec_uq)), int(max(dec_uq)) + 1),
            a[1],
            bottom=a[0].tolist(),
            color="green",
            label="Bad",
            edgecolor="black",
        )
        ax.bar(
            range(int(min(dec_uq)), int(max(dec_uq)) + 1),
            a[-1],
            bottom=(a[0] + a[1]).tolist(),
            color="grey",
            label="Undefined",
            edgecolor="black",
        )
    else:
        a["Rate"] = a[1] / (a[0] + a[1])
        ax.bar(
            range(int(min(dec_uq)), int(max(dec_uq)) + 1),
            a[0],
            color="orange",
            label="Good",
            edgecolor="black",
        )
        ax.bar(
            range(int(min(dec_uq)), int(max(dec_uq)) + 1),
            a[1],
            bottom=a[0],
            color="green",
            label="Bad",
            edgecolor="black",
        )

    ax2 = ax.twinx()
    ax2.plot(
        range(int(min(dec_uq)), int(max(dec_uq)) + 1),
        a["Rate"],
        lw=3,
        color="blue",
        marker="o",
    )

    if name is not None:
        ax.set_title(name, fontsize=32)
    fig.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


def analys_nan(df, feats):
    data = pd.DataFrame(
        {"Features": feats, "proportion": feats, "number_of_NAN": feats}
    )
    for i in range(len(data)):
        data.proportion[i] = (
                str(round(df[data.Features[i]].isna().sum() / len(df) * 100, 2)) + "%"
        )
        data.number_of_NAN[i] = df[data.Features[i]].isna().sum()
    return data


def binomi(
        train, valid, score, dec_f, target, conf_int=[[0.025, 0.975], [0.005, 0.995]]
):
    """
    функция принимает на вход обучающие данные, валидационные данные, предсказания,
    децили и целевой признак
    возвращает таблицу c резултатами биномиального теста
    """
    group = train.groupby(dec_f)
    group2 = valid.groupby(dec_f)

    stats = pd.DataFrame(
        columns=["Decil", "Left", "Right", "Expected DR", "All", "Defolt"]
    )
    # print(train[score])
    stats["Left"] = group[score].min().round(4)
    stats["Right"] = group[score].max().round(4)
    stats["All"] = group2[target].count()
    stats["Defolt"] = group2[target].sum()
    stats["Expected DR"] = (group[target].sum() / group[target].count()).round(4)
    stats["Realized DR"] = (group2[target].sum() / group2[target].count()).round(4)
    stats["Decil"] = list(stats.index.values.astype(str))
    for i in range(len(conf_int)):
        stats["{}".format(conf_int[i][0])] = round(
            scipy.stats.binom.ppf(conf_int[i][0], stats["All"], stats["Expected DR"])
            / stats["All"],
            4,
        )
        stats["{}".format(conf_int[i][1])] = round(
            scipy.stats.binom.ppf(conf_int[i][1], stats["All"], stats["Expected DR"])
            / stats["All"],
            4,
        )
    cols = list(stats.columns)

    stats["overestimation"], stats["underestimation"] = "", ""
    try:
        for i in range(len(stats)):
            if stats["Expected DR"][i] == 0 and stats["Realized DR"][i] == 0:
                stats["0.025"][i] = 0
                stats["0.975"][i] = 0
                stats["0.005"][i] = 0
                stats["0.995"][i] = 0
    except:
        pass
    for i in stats.index:
        if stats["Realized DR"][i] > stats[cols[10]][i]:
            stats.overestimation[i] = "red"
        elif (stats["Realized DR"][i] <= stats[cols[10]][i]) and (
                stats["Realized DR"][i] > stats[cols[8]][i]
        ):
            stats.overestimation[i] = "yellow"
        else:
            stats.overestimation[i] = "green"

    for i in stats.index:
        if stats["Realized DR"][i] < stats[cols[9]][i]:
            stats.underestimation[i] = "red"
        elif (stats["Realized DR"][i] >= stats[cols[9]][i]) and (
                stats["Realized DR"][i] < stats[cols[7]][i]
        ):
            stats.underestimation[i] = "yellow"
        else:
            stats.underestimation[i] = "green"

    stats1 = pd.DataFrame(stats)

    return [stats.to_string(index=False), stats, stats1]


def color_under(df):
    if list(df.underestimation).count("red") > 0:
        if (
                df.underestimation.value_counts()["red"]
                >= (df["Realized DR"].notnull().sum() * 20) / 100
        ):
            return "total underestimation is red"
        elif (
                df.underestimation.value_counts()["red"]
                <= (df["Realized DR"].notnull().sum() * 20) / 100
                and df.underestimation.value_counts()["red"] > 0
        ):
            return "total underestimation is yellow"
    if (
            list(df.underestimation).count("red") > 0
            and list(df.underestimation).count("yellow") > 0
    ):
        if (
                df.underestimation.value_counts()["red"]
                + df.underestimation.value_counts()["yellow"]
                >= (df["Realized DR"].notnull().sum() * 20) / 100
        ):
            return "total underestimation is yellow"
    if list(df.underestimation).count("yellow") > 0:
        if (
                df.underestimation.value_counts()["yellow"]
                >= (df["Realized DR"].notnull().sum() * 20) / 100
        ):
            return "total underestimation is yellow"
    else:
        return "total underestimation is green"


def color_over(df):
    if list(df.overestimation).count("red") > 0:
        if (
                df.overestimation.value_counts()["red"]
                >= (df["Realized DR"].notnull().sum() * 20) / 100
        ):
            return "total overestimation is red"
        elif (
                df.overestimation.value_counts()["red"]
                <= (df["Realized DR"].notnull().sum() * 20) / 100
                and df.overestimation.value_counts()["red"] > 0
        ):
            return "total overestimation is yellow"
    elif (
            list(df.overestimation).count("red") > 0
            and list(df.overestimation).count("yellow") > 0
    ):
        if (
                df.overestimation.value_counts()["red"]
                + df.overestimation.value_counts()["yellow"]
                >= (df["Realized DR"].notnull().sum() * 20) / 100
        ):
            return "total overestimation is yellow"
    elif list(df.overestimation).count("yellow") > 0:
        if (
                df.overestimation.value_counts()["yellow"]
                > (df["Realized DR"].notnull().sum() * 20) / 100
        ):
            return "total overestimation is yellow"
    else:

        return "total overestimation is green"


def result_binom(df):
    if (
            color_over(df) == "total overestimation is red"
            or color_under(df) == "total underestimation is red"
    ):
        return "Красный"
    elif (
            color_over(df) == "total overestimation is yellow"
            or color_under(df) == "total underestimation is yellow"
    ):
        return "Желтый"
    else:
        return "Зеленый"


def psi(score_initial, score_new, num_bins=10, mode="fixed"):
    eps = 1e-4

    # Sort the data
    score_initial.sort()
    score_new.sort()

    # Prepare the bins
    min_val = min(min(score_initial), min(score_new))
    max_val = max(max(score_initial), max(score_new))
    if mode == "fixed":
        bins = [
            min_val + (max_val - min_val) * (j) / num_bins for j in range(num_bins + 1)
        ]
    elif mode == "quantile":
        bins = pd.qcut(score_initial, q=num_bins, retbins=True, duplicates="drop")[
            1
        ]  # Create the quantiles based on the initial population

    else:
        raise ValueError(
            f"Mode '{mode}' not recognized. Your options are 'fixed' and 'quantile'"
        )
    bins[0] = min_val - eps  # Correct the lower boundary
    bins[-1] = max_val + eps  # Correct the higher boundary

    # Bucketize the initial population and count the sample inside each bucket
    bins_initial = pd.cut(score_initial, bins=bins, labels=range(1, num_bins + 1))
    df_initial = pd.DataFrame({"initial": score_initial, "bin": bins_initial})
    grp_initial = df_initial.groupby("bin").count()
    grp_initial["percent_initial"] = grp_initial["initial"] / sum(
        grp_initial["initial"]
    )

    # Bucketize the new population and count the sample inside each bucket
    bins_new = pd.cut(score_new, bins=bins, labels=range(1, num_bins + 1))
    df_new = pd.DataFrame({"new": score_new, "bin": bins_new})
    grp_new = df_new.groupby("bin").count()
    grp_new["percent_new"] = grp_new["new"] / sum(grp_new["new"])

    # Compare the bins to calculate PSI
    psi_df = grp_initial.join(grp_new, on="bin", how="inner")

    # Add a small value for when the percent is zero
    psi_df["percent_initial"] = psi_df["percent_initial"].apply(
        lambda x: eps if x == 0 else x
    )
    psi_df["percent_new"] = psi_df["percent_new"].apply(lambda x: eps if x == 0 else x)

    # Calculate the psi
    psi_df["psi"] = (psi_df["percent_initial"] - psi_df["percent_new"]) * np.log(
        psi_df["percent_initial"] / psi_df["percent_new"]
    )

    # Return the psi values
    return psi_df["psi"].values


def for_result_psi(df, type_block, type_model):
    if (type_block == "Розничный блок") or (type_block == "Бизнес блок") or (
            type_block == "Корпоративный блок" and type_model == "Антифрод модель"):
        if any([x >= 0.2 for x in df["TARGET_0"].to_list()]) or any(
                [x >= 0.2 for x in df["TARGET_1"].to_list()]
        ):
            return "Красный"
        elif any([0.1 <= x < 0.2 for x in df["TARGET_0"].to_list()]) or any(
                [0.1 <= x < 0.2 for x in df["TARGET_1"].to_list()]
        ):
            return "Желтый"
        else:
            return "Зеленый"

    elif type_block == "Корпоративный блок" and type_model == "Скоринговая модель":
        if any([x >= 0.25 for x in df["TARGET_0"].to_list()]) or any(
                [x >= 0.25 for x in df["TARGET_1"].to_list()]
        ):
            return "Красный"
        elif any([0.1 <= x < 0.25 for x in df["TARGET_0"].to_list()]) or any(
                [0.1 <= x < 0.25 for x in df["TARGET_1"].to_list()]
        ):
            return "Желтый"
        else:
            return "Зеленый"


def VIF(df, featx):
    df = df.fillna(0)
    X = sm.add_constant(df[featx])
    res = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
    )
    res = pd.DataFrame(res)
    res = res.rename(columns={0: "VIF"})
    res["оценка"] = ""
    for i in range(len(res)):
        if float(res["VIF"][i]) > 5:
            res["оценка"][i] = "красный"
        elif 3 < float(res["VIF"][i]) <= 5:
            res["оценка"][i] = "желтый"
        else:
            res["оценка"][i] = "зеленый"

    return res


def for_result_vif(df):
    if any([x == "красный" for x in df["оценка"].to_list()[1:]]):
        return "Красный"
    elif (
            df["оценка"].to_list().count("желтый") / (len(df["оценка"].to_list()) - 1)
            >= 0.1
    ):
        return "Желтый"
    else:
        return "Зеленый"


def coef_gini_model(valid_pred, y_valid):
    """
    возвращает коэффициент Джини по всей модели
    """
    fpr, tpr, thresholds = roc_curve(
        y_valid, valid_pred, pos_label=None, sample_weight=None, drop_intermediate=True
    )
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=1,
        label="valid "
              + "ROC curve (area = %0.2f), " % roc_auc
              + "Gini = %0.3f" % (roc_auc * 2 - 1),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.show()
    return round(roc_auc * 2 - 1, 3)


def corilation_p(df, features):
    """
    функция принимает на вход датасет и признаки
    возвращает таблицию с корреляцией Пирсона
    """
    sample = df[features].copy()
    if len(features) <= 10:
        size = (11, 9)
    elif len(features) > 10:
        size = (22, 18)
    corr = sample.corr() * 100
    # mask = np.triu(np.ones_like(corr, dtype=bool))
    # f, ax = plt.subplots(figsize = (size))
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
    #        linewidths=5, cmap=cmap, vmin=-100, vmax=100,
    #        cbar_kws={"shrink": .8}, square=True)
    return corr


def result_for_gini(gini, type_block):
    if type_block == "Розничный блок":
        if gini < 0.45:
            return "Красный"
        elif 0.45 < gini < 0.55:
            return "Желтый"
        else:
            return "Зеленый"

    elif type_block == "Корпоративный блок":
        if gini < 0.45:
            return "Красный"
        elif 0.45 < gini < 0.50:
            return "Желтый"
        else:
            return "Зеленый"

    elif type_block == "Бизнес блок":
        if gini < 0.35:
            return "Красный"
        elif 0.35 < gini < 0.45:
            return "Желтый"
        else:
            return "Зеленый"


# Функция для проведения процедуры Bootstrap


def coef_gini_model_without_roc(valid_pred, target):
    fpr, tpr, thresholds = roc_curve(
        target, valid_pred, pos_label=None, sample_weight=None, drop_intermediate=True
    )

    roc_auc = auc(fpr, tpr)
    return roc_auc * 2 - 1


def conf_interval_for_bootstrap_test(
        data, n_bootstraps, train_size_percent, features_columns, target, model_copy
):
    coefs_gini = []
    for i in range(n_bootstraps):
        df_element = data.sample(
            n=int((train_size_percent / 100) * len(data)), random_state=i
        )
        remaining_data = data.drop(df_element.index)
        model_copy.fit(df_element[features_columns], df_element[target])
        try:
            # y_pred = model_copy.predict_proba(df_element[features_columns])[:, 1]
            y_pred = model_copy.predict_proba(remaining_data[features_columns])[:, 1]

        except:
            # y_pred = model_copy.predict(df_element[features_columns])
            y_pred = model_copy.predict(remaining_data[features_columns])

        coefs_gini.append(coef_gini_model_without_roc(y_pred, remaining_data[target]))

    coefs_gini = pd.Series(coefs_gini)

    mean = coefs_gini.mean()
    upper = coefs_gini.quantile(0.975)
    lower = coefs_gini.quantile(0.025)

    df = pd.DataFrame(
        {
            "Среднее gini:": [round(mean, 3)],
            "Доверитильный интервал (95%)": [(round(upper, 3), round(lower, 3))],
            "MIN": [round(min(coefs_gini), 3)],
            "MAX": [round(max(coefs_gini), 3)],
        }
    )

    #     print(f'Среднее gini: {mean:.3f}')
    #     print(f'Доверитильный интервал (95%): {lower:.3f} - {upper:.3f}')
    #     print(f'MIN: {round(min(coefs_gini), 3)}, MAX: {round(max(coefs_gini), 3)}')
    return df


def data_depth(data, date_column, target, type_block, type_model):
    """
    Функция принимает на вход таблицу с данными, наименование столбца с датой и целевым событием.
    Данные группируютcя по дате, добавляются столбцы с номером квартала и годом, и количеством каждой уникальной даты.
    Также считается количество уникальных дат, на основании чего выставляется итоговый результат по тесту.
    На выходе получаем сгруппированную таблицу с датами и
    """
    data = data.copy()
    if data[date_column].dtype == "object":
        data[date_column] = pd.to_datetime(data[date_column])
    data[date_column] = data[date_column].dt.to_period("m")

    def get_quarter(date):
        for months, quarter in [
            ([1, 2, 3], 1),
            ([4, 5, 6], 2),
            ([7, 8, 9], 3),
            ([10, 11, 12], 4),
        ]:
            if date.month in months:
                return quarter

    length = len(data[date_column].sort_values().unique())

    pvt = data.pivot_table(index=date_column, values=target, aggfunc="sum")
    pvt = pvt.reset_index()
    pvt = pvt.rename(columns={"TARGET": "target_quantity"})

    pvt["year"] = 0
    pvt["quarter"] = 0
    for i, k in enumerate(pvt[date_column]):
        pvt["quarter"][i] = get_quarter(k)
        pvt["year"][i] = k.year

    if type_block in ["Розничный блок", "Корпоративный блок"] and type_model == "Антифрод модель":
        if length <= 6:
            color = "Красный"
        elif 6 < length < 9:
            color = 'Желтый'
        else:
            color = "Зеленый"

    else:
        if length <= 11:
            color = "Красный"
        # elif 6 < length < 9:
        #     color = 'Желтый'
        else:
            color = "Зеленый"

    pvt[date_column] = pvt[date_column].astype(str)
    return pvt, color, length


def calculate_month_difference(data, date_col, development_date, type_block, type_model):
    # Приведение всех дат к единому формату
    data["formatted_date"] = pd.to_datetime(data[date_col], errors="coerce")
    development_date = pd.to_datetime(development_date, errors="coerce")

    # Определение максимальной даты в таблице
    max_date = data["formatted_date"].max()
    try:
        # Рассчет разницы в месяцах
        month_difference = abs((development_date - max_date).days // 30)

        res_table = pd.DataFrame(
            {
                "Дата разработки": [development_date],
                "Последняя дата реализации целевого события": [max_date],
                "Разница дат в месяцах": [month_difference],
            }
        )

        if type_block == "Розничный блок" or (
                type_block == "Корпоративный блок" and type_model == "Скоринговая модель"):

            for i in res_table["Разница дат в месяцах"]:
                if i >= 24:
                    # print('\033[31m {}'.format('Данный тест модель проходит на красный цвет'))
                    return max_date, "Красный", res_table
                elif 18 <= i < 24:
                    # print('\033[33m {}'.format('Данный тест модель проходит на желтый цвет'))
                    return max_date, "Желтый", res_table
                else:
                    # print('\033[32m {}'.format('Данный тест модель проходит на зеленый цвет'))
                    return max_date, "Зеленый", res_table

        elif type_block == "Бизнес блок" or (type_block == "Корпоративный блок" and type_model == "Антифрод модель"):
            for i in res_table["Разница дат в месяцах"]:
                if i >= 24:
                    # print('\033[31m {}'.format('Данный тест модель проходит на красный цвет'))
                    return max_date, "Красный", res_table
                elif 6 <= i < 24:
                    # print('\033[33m {}'.format('Данный тест модель проходит на желтый цвет'))
                    return max_date, "Желтый", res_table
                else:
                    # print('\033[32m {}'.format('Данный тест модель проходит на зеленый цвет'))
                    return max_date, "Зеленый", res_table
    except:
        return "Проверьте данные!"


"""
Тест M2.6: Оценка производительности (общей эффективности) модели (Значимый тест)
"""


def accuracy(train_y, test_y, prob_tr, prob_ts, thresh):
    preds_tr = np.where(prob_tr > thresh, 1, 0)
    preds_ts = np.where(prob_ts > thresh, 1, 0)

    # Calculate Accuracy for the training and test sets
    accuracy_tr = round(accuracy_score(train_y, preds_tr), 3)
    accuracy_ts = round(accuracy_score(test_y, preds_ts), 3)

    vals = [accuracy_tr, accuracy_ts]

    # Define colors based on the criteria
    train_color = (
        "green"
        if 0.70 <= accuracy_tr < 1
        else "yellow" if 0.60 <= accuracy_tr < 0.70 else "red"
    )
    test_color = (
        "green"
        if 0.70 <= accuracy_ts < 1
        else "yellow" if 0.60 <= accuracy_ts < 0.70 else "red"
    )

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plotting
    bars = ax.bar(
        ["Train", "Test"], [accuracy_tr, accuracy_ts], color=[train_color, test_color]
    )
    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(val), ha="center", va="bottom")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Sample accuracy values")

    fig.savefig("test_results/accuracy_plot", bbox_inches='tight')

    return train_color, test_color, accuracy_tr, accuracy_ts, fig


"""
Тест M2.7: Оценка точности модели в определении истинно положительных результатов (Значимый тест)
"""


def precision(y_test, prob_ts, thresh, type_block, type_model):
    """
    Функция для проведения теста precision

    Аргументы:
    model: Обученная модель классификатора
    prob_ts: Вероятности принадлежности к классу 1 для тестового набора данных
    y_test: Истинные метки классов для тестового набора данных
    thresh: Порог для преобразования вероятностей в метки классов

    Возвращает:
    str: оценка результата теста (красный, желтый, зеленый)
    """

    preds_ts = np.where(prob_ts > thresh, 1, 0)
    prec_ts = round(precision_score(y_test, preds_ts), 3)

    if type_block in ["Корпоративный блок", "Розничный блок"] and type_model == 'Антифрод модель':
        if prec_ts < 0.25:
            color = "red"
            return prec_ts, color
        elif 0.25 <= prec_ts < 0.40:
            color = "yellow"
            return prec_ts, color
        elif prec_ts >= 0.40:
            color = "green"
            return prec_ts, color
    else:
        if prec_ts < 0.30:
            color = "red"
            return prec_ts, color
        elif 0.30 <= prec_ts < 0.50:
            color = "yellow"
            return prec_ts, color
        elif prec_ts >= 0.50:
            color = "green"
            return prec_ts, color


"""
Тест M2.8: Оценка полноты (частоты) модели в определении истинно положительных результатов (Значимый тест)
"""


def recall(y_test, y_proba, thresh, type_block, type_model):
    # Применение порога для преобразования вероятностей в метки классов
    y_preds = np.where(y_proba > thresh, 1, 0)

    # Вычисление Recall
    recall = round(recall_score(y_test, y_preds), 3)

    if type_block in ["Корпоративный блок", "Розничный блок"] and type_model == 'Антифрод модель':
        # Оценка по критериям
        if recall < 0.40:
            return recall, "red"
        elif 0.40 <= recall < 0.60:
            return recall, "yellow"
        elif recall >= 0.60:
            return recall, "green"
    else:
        # Оценка по критериям
        if recall < 0.30:
            return recall, "red"
        elif 0.30 <= recall < 0.50:
            return recall, "yellow"
        elif recall >= 0.50:
            return recall, "green"


"""
Тест M2.9: Оценка среднего гармонического значения между точностью и полнотой модели (Значимый тест)
"""


def f1_score_eval(y_test, y_proba, thresh, type_block, type_model):
    # Применение порога для преобразования вероятностей в метки классов
    y_preds = np.where(y_proba > thresh, 1, 0)

    if type_block in ["Корпоративный блок", "Розничный блок"] and type_model == 'Антифрод модель':
        # Вычисление F1-меры
        f1 = round(f1_score(y_test, y_preds), 3)
        if f1 < 0.25:
            return f1, "red"
        elif 0.25 <= f1 < 0.40:
            return f1, "yellow"
        elif f1 >= 0.40:
            return f1, "green"
    else:
        # Вычисление F1-меры
        f1 = round(f1_score(y_test, y_preds), 3)
        if f1 < 0.30:
            return f1, "red"
        elif 0.30 <= f1 < 0.50:
            return f1, "yellow"
        elif f1 >= 0.50:
            return f1, "green"


"""
Тест M2.10: Оценка эффективности модели – матрица ошибок (Значимый тест)
"""


def all_metrics(prob_tr, prob_ts, train_y, test_y, thresh):
    preds_tr = np.where(prob_tr > thresh, 1, 0)
    preds_ts = np.where(prob_ts > thresh, 1, 0)

    # Построение матрицы ошибок
    conf_matrix_tr = confusion_matrix(train_y, preds_tr)
    conf_matrix_ts = confusion_matrix(test_y, preds_ts)

    # Получение уникальных меток классов
    classes = unique_labels(train_y, preds_tr)

    # Вычисление TP, TN, FP, FN
    TN_tr = conf_matrix_tr[0, 0]
    FP_tr = conf_matrix_tr[0, 1]
    FN_tr = conf_matrix_tr[1, 0]
    TP_tr = conf_matrix_tr[1, 1]

    TN_ts = conf_matrix_ts[0, 0]
    FP_ts = conf_matrix_ts[0, 1]
    FN_ts = conf_matrix_ts[1, 0]
    TP_ts = conf_matrix_ts[1, 1]

    # Отображение матрицы ошибок с метками TP, TN, FP, FN
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    sns.heatmap(
        conf_matrix_tr, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax[0]
    )
    ax[0].set_title("Train")
    ax[0].set_xlabel("Predicted label")
    ax[0].set_ylabel("True label")
    ax[0].text(0, 0.1, f"TN", color="white")
    ax[0].text(1, 0.1, f"FP", color="black")
    ax[0].text(0, 1.1, f"FN", color="black")
    ax[0].text(1, 1.1, f"TP", color="black")

    sns.heatmap(
        conf_matrix_ts, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax[1]
    )
    ax[1].set_title("Test")
    ax[1].set_xlabel("Predicted label")
    ax[1].set_ylabel("True label")
    ax[1].text(0, 0.1, f"TN", color="white")
    ax[1].text(1, 0.1, f"FP", color="black")
    ax[1].text(0, 1.1, f"FN", color="black")
    ax[1].text(1, 1.1, f"TP", color="black")

    fig.savefig("test_results/error_matrix_plot", bbox_inches='tight')
    first_tr = np.round(((FP_tr / (TN_tr + FP_tr)) * 100), 3)
    second_tr = np.round(((FN_tr / (TP_tr + FN_tr)) * 100), 3)

    first_ts = np.round(((FP_ts / (TN_ts + FP_ts)) * 100), 3)
    second_ts = np.round(((FN_ts / (TP_ts + FN_ts)) * 100), 3)

    df = pd.DataFrame(
        {
            "Train": [first_tr, second_tr],
            "Test": [first_ts, second_ts],
        },
        index=["Ошибка первого рода", "Ошибка второго рода"],
    ).round(3)

    def color_cell(x):
        if x > 50:
            return "red"
        elif 30 < x <= 50:
            return "yellow"
        else:
            return "green"

    colors_list = [color_cell(val) for val in df.values.flatten()]
    if 'red' in colors_list:
        color = 'Красный'
    elif 'yellow' in colors_list:
        color = 'Желтый'
    else:
        color = 'Зеленый'

    styled_df = df.style.applymap(lambda x: f"background-color: {color_cell(x)}")
    return color, fig, styled_df


"""
Тест M2.11: Оценка качества вероятностных предсказаний (Значимый тест)
"""


def log_loss(y_true, prob_ts, thresh):
    epsilon = 1e-15  # Малое число для предотвращения деления на ноль
    preds_ts = np.where(prob_ts > thresh, 1, 0)
    y_pred_proba = np.clip(preds_ts, epsilon, 1 - epsilon)  # Ограничиваем вероятности
    log_loss_values = -(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    mean_log_loss = np.mean(log_loss_values)

    if mean_log_loss > 0.7:
        result = "red"
    elif 0.4 <= mean_log_loss <= 0.7:
        result = "orange"
    else:
        result = "green"

    # print("log_loss:", log_loss)
    # print('\n')
    # print("mean_log_loss:", mean_log_loss)
    return log_loss_values, mean_log_loss, result


def m1_1(train, test, type_block, type_model, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    if train.shape[0] != 0 and test.shape[0] != 0:
        features = train.drop(columns="TARGET").columns.tolist()
        features.remove("DECIL")
        features.remove("SCORE")
        psi_df = pd.DataFrame(index=features, columns=[0, 1])
        for i in train.TARGET.unique():
            for col in features:
                if col != "TARGET":
                    try:
                        csi_values = psi(
                            train[train["TARGET"] == i][col].values,
                            test[test["TARGET"] == i][col].values,
                            mode="fixed",
                        )
                        csi = np.mean(csi_values)
                        psi_df.loc[col, i] = round(csi, 4)
                    except:
                        psi_df.loc[col, i] = np.nan
        psi_df = psi_df.rename(columns={0: "TARGET_0", 1: "TARGET_1"})
        color = for_result_psi(psi_df, type_block, type_model)
        if color == "Красный":
            color_picker = ":red"
        elif color == "Желтый":
            color_picker = ":orange"
        else:
            color_picker = ":green"

        tab2.write(
            "**М1.1 АНАЛИЗ РЕПРЕЗЕНТАТИВНОСТИ ВЫБОРКИ (ЗНАЧИМЫЙ ТЕСТ)-** "
            + color_picker
            + "[**{}**]".format(for_result_psi(psi_df, type_block, type_model))
        )

        if type_block == "Розничный блок" or type_block == "Бизнес блок" or (
                type_block == "Корпоративный блок" and type_model == "Антифрод модель"):
            psi_df = psi_df.style.applymap(
                lambda x: (
                    "background-color: red"
                    if x >= 0.2
                    else (
                        "background-color: orange"
                        if 0.1 <= x < 0.2
                        else "background-color: green"
                    )
                )
            )

        elif type_block == "Корпоративный блок" and type_model == "Скоринговая модель":
            psi_df = psi_df.style.applymap(
                lambda x: (
                    "background-color: red"
                    if x >= 0.25
                    else (
                        "background-color: orange"
                        if 0.1 <= x < 0.25
                        else "background-color: green"
                    )
                )
            )

        tab2.dataframe(psi_df)

    return psi_df, color


def m1_2(data_for_dyn, development_date, type_block, type_model, tab2):
    max_data, color, df = calculate_month_difference(
        data_for_dyn, "ISSUE_DATE", development_date, type_block, type_model
    )
    if development_date < max_data:
        tab2.error(
            "**M1.2 АКТУАЛЬНОСТЬ ДАННЫХ (ЗНАЧИМЫЙ ТЕСТ)-** :red[**ДАННЫЕ УСТАРЕЛИ**"
            " **ИЛИ ВЫБРАН НЕКОРРЕКТНЫЙ ПЕРИОД ДЛЯ РАЗРАБОТКИ МОДЕЛИ**]"
        )
    else:
        if color == "Красный":
            color_picker = ":red"
        elif color == "Желтый":
            color_picker = ":orange"
        else:
            color_picker = ":green"
        tab2.write(
            "**M1.2 АКТУАЛЬНОСТЬ ДАННЫХ (ЗНАЧИМЫЙ ТЕСТ)-** "
            + color_picker
            + "[**{}**]".format(color)
        )
        tab2.write(
            color_picker + f"[**Данный тест модель проходит на {color} цвет.**]"
        )
        tab2.write(df)

    return color, df


def m1_3(df_raw, features_raw, tab2):
    if df_raw.shape[0] != 0:
        a_empty = analys_nan(df_raw, features_raw)
        tab2.write("**М1.3 АНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ.(ИНФОРМАТИВНЫЙ ТЕСТ)**")
        tab2.dataframe(a_empty)
    return a_empty


def m1_3_stability(df_raw, tab2):
    features_for_nan = df_raw.columns.tolist()
    try:
        features_for_nan.remove("SCORE")
    except:
        pass
    try:
        features_for_nan.remove("DECIL")
    except:
        pass
    try:
        features_for_nan.remove("TARGET")
    except:
        pass
    if "ISSUE_DATE" not in features_for_nan:
        features_for_nan.append("ISSUE_DATE")

    calc_month = calculate_missing_percentage_by_month(
        df_raw[features_for_nan], "ISSUE_DATE"
    )
    calc_quart = calculate_missing_percentage_by_quart(
        df_raw[features_for_nan], "ISSUE_DATE"
    )
    tab2.write(
        "**СТАБИЛЬНОСТЬ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ(ПОМЕСЯЧНО).(ИНФОРМАТИВНЫЙ ТЕСТ)**"
    )
    tab2.dataframe(calc_month[0])
    tab2.pyplot(calc_nan_plot(calc_month[1], calc_month[2], 'month_nan_plot'))

    tab2.write(
        "**СТАБИЛЬНОСТЬ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ(ПОКВАРТАЛЬНО).(ИНФОРМАТИВНЫЙ ТЕСТ)**"
    )
    tab2.dataframe(calc_quart[0])
    tab2.pyplot(calc_nan_plot(calc_quart[1], calc_quart[2], 'quart_nan_plot'))

    return calc_month[0], calc_quart[0], calc_month[-1]


def m1_4_glubina(data_for_dyn, type_block, type_model, tab2):
    data_dep, color, length = data_depth(
        data_for_dyn, "ISSUE_DATE", "TARGET", type_block, type_model
    )
    if color == "Красный":
        color_picker = ":red"
    elif color == "Желтый":
        color_picker = ":orange"
    else:
        color_picker = ":green"
    tab2.write(
        "**М1.4 ПРОВЕРКА ГЛУБИНЫ И КАЧЕСТВА ДАННЫХ, ИСПОЛЬЗОВАННЫХ В РАЗРАБОТКЕ МОДЕЛИ.(ИНФОРМАТИВНЫЙ ТЕСТ)-** "
        + color_picker
        + "[**{}**]".format(color)
    )
    tab2.write(
        color_picker + f"[**Глубина данных составляет {length} месяцев.**]"
    )
    tab2.dataframe(data_dep)
    return color, length, data_dep


def m1_4_analyze(test, train, tab2):
    train = train.copy()
    test = test.copy()
    test.columns = [x.upper() for x in test.columns]
    if (
            "ISSUE_DATE" in list(train.columns)
            and train.shape[0] != 0
            and test.shape[0] != 0
    ):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    train.DECIL = train.DECIL.astype(float)
    test.DECIL = test.DECIL.astype(float)
    j = (
        test["DECIL"]
        .value_counts(normalize=True, sort=False, dropna=False)
        .replace(0, 0.0000001)
        .sort_index()
    )
    resv = sum(map(lambda i: i * i, j)) * 100
    j = (
        train["DECIL"]
        .value_counts(normalize=True, sort=False, dropna=False)
        .replace(0, 0.0000001)
        .sort_index()
    )
    rest = sum(map(lambda i: i * i, j)) * 100

    if res_actual(resv) == "Красный" or res_actual(rest) == "Красный":
        word_AD = "Красный"
        color_picker_AD = ":red"
    elif res_actual(resv) == "Желтый" or res_actual(rest) == "Желтый":
        word_AD = "Желтый"
        color_picker_AD = ":orange"
    else:
        word_AD = "Зеленый"
        color_picker_AD = ":green"
    tab2.write(
        "**М1.4 АНАЛИЗ НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ ФАКТОРОВ И ПРОВЕРКА НА НАЛИЧИЕ ВЫБРОСОВ (ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ** "
        + color_picker_AD
        + "[**{}**]".format(word_AD)
    )
    tab2.write(f"Индекс Херфиндаля для обучающей выборки = {rest:.2f}")
    tab2.write(f"Индекс Херфиндаля для валидационной выборки = {resv:.2f}")
    train1 = train.copy()
    test1 = test.copy()
    tab2.pyplot(decils_plot(train1, "DECIL", "TARGET", "TRAIN"))
    tab2.pyplot(decils_plot(test1, "DECIL", "TARGET", "TEST"))


def m2_1_all_model(train, test, type_block, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    train_gini = metrics.roc_auc_score(train["TARGET"], train["SCORE"]) * 2 - 1
    test_gini = metrics.roc_auc_score(test["TARGET"], test["SCORE"]) * 2 - 1
    # tab2.write(result_for_gini(test_gini))
    # tab2.write(result_for_gini(train_gini))
    if (
            result_for_gini(test_gini, type_block) == "Красный"
            or result_for_gini(train_gini, type_block) == "Красный"
    ):
        color_picker_ts = ":red"
        word_gini = "Красный"
    elif (
            result_for_gini(test_gini, type_block) == "Желтый"
            or result_for_gini(train_gini, type_block) == "Желтый"
    ):
        color_picker_ts = ":orange"
        word_gini = "Желтый"
    else:
        color_picker_ts = ":green"
        word_gini = "Зеленый"

    tab2.write(
        "**М2.1 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЕ ВСЕЙ МОДЕЛИ.(ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_ts
        + "[**{}**]".format(word_gini)
    )

    # Пример использования функции с данными для тренировки и тестирования
    scores_labels = [
        (train.SCORE, train.TARGET, "Train"),
        (test.SCORE, test.TARGET, "Test")
    ]
    roc_fig = ROC(scores_labels, "Comparative ROC Curve")
    tab2.pyplot(roc_fig)
    return word_gini, roc_fig


def m2_2_bootstrap(train, test, type_block, model_clf, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    X_columns_for_boots = train.columns.tolist().copy()
    X_columns_for_boots.remove("TARGET")
    X_columns_for_boots.remove("DECIL")
    X_columns_for_boots.remove("SCORE")
    data_boots = pd.concat([train, test])
    data_boots.columns = [x.upper() for x in data_boots.columns]
    conf = conf_interval_for_bootstrap_test(
        data_boots, 100, 80, X_columns_for_boots, "TARGET", model_clf
    )
    # tab2.write('**Доверительный интервал для коэффициента Джини методом бутстрэпа = {}**%'.format(
    #     tuple([i * 100 for i in conf])))

    if type_block == "Розничный блок":
        if ((conf["MIN"][0] + conf["MAX"][0]) * 100 / 2) < 45:
            color_for_boots = ":red"
            word_for_boots = "Красный"
        elif 45 < ((conf["MIN"][0] + conf["MAX"][0]) * 100 / 2) < 55:
            color_for_boots = ":orange"
            word_for_boots = "Желтый"
        else:
            color_for_boots = ":green"
            word_for_boots = "Зеленый"

    elif type_block == "Бизнес блок":
        if ((conf["MIN"][0] + conf["MAX"][0]) * 100 / 2) < 35:
            color_for_boots = ":red"
            word_for_boots = "Красный"
        elif 35 < ((conf["MIN"][0] + conf["MAX"][0]) * 100 / 2) < 45:
            color_for_boots = ":orange"
            word_for_boots = "Желтый"
        else:
            color_for_boots = ":green"
            word_for_boots = "Зеленый"

    elif type_block == "Корпоративный блок":
        if ((conf["MIN"][0] + conf["MAX"][0]) * 100 / 2) < 45:
            color_for_boots = ":red"
            word_for_boots = "Красный"
        elif 45 < ((conf["MIN"][0] + conf["MAX"][0]) * 100 / 2) < 50:
            color_for_boots = ":orange"
            word_for_boots = "Желтый"
        else:
            color_for_boots = ":green"
            word_for_boots = "Зеленый"
    tab2.write(
        "**M2.2 Использование процедуры бутстрепа (ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_for_boots
        + "[**{}**]".format(word_for_boots)
    )
    tab2.dataframe(conf)
    return word_for_boots, conf


def m2_3(model_clf, train, test, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    train_gini_for_without_test = (
            metrics.roc_auc_score(train["TARGET"], train["SCORE"]) * 2 - 1
    )
    test_gini_for_without_test = (
            metrics.roc_auc_score(test["TARGET"], test["SCORE"]) * 2 - 1
    )
    X_columns_for_without = train.columns.tolist().copy()
    y_columns_for_without = train.TARGET
    X_columns_for_without.remove("TARGET")
    X_columns_for_without.remove("DECIL")
    X_columns_for_without.remove("SCORE")
    # X_columns_for_without.remove('ISSUE_DATE')

    range_without_feat = range_effect_without_f(
        train,
        test,
        X_columns_for_without,
        "TARGET",
        model_clf,
        train_gini_for_without_test * 100,
        test_gini_for_without_test * 100,
    )[0]
    range_without_feat111 = range_effect_without_f(
        train,
        test,
        X_columns_for_without,
        "TARGET",
        model_clf,
        train_gini_for_without_test * 100,
        test_gini_for_without_test * 100,
    )[1]

    if [
        True
        for i, j in zip(
            range_without_feat111["gini_train-new_gini_train"],
            range_without_feat111["gini_test-new_gini_test"],
        )
        if i > 5 and j > 5
    ]:
        color_picker_ts = ":red"
        word_gini = "Красный"
    elif [
        True
        for i, j in zip(
            range_without_feat111["gini_train-new_gini_train"],
            range_without_feat111["gini_test-new_gini_test"],
        )
        if i > 0 and j > 0
    ]:
        color_picker_ts = ":orange"
        word_gini = "Желтый"
    else:
        color_picker_ts = ":green"
        word_gini = "Зеленый"

    tab2.write(
        "**М2.3 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ С ИСКЛЮЧЕНИЕМ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_ts
        + "[**{}**]".format(word_gini)
    )

    tab2.dataframe(range_without_feat)
    return word_gini, range_without_feat


def m2_4(train, test, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    list_features_for_erof = list(train.columns)
    list_features_for_erof.remove("TARGET")
    list_features_for_erof.remove("DECIL")
    list_features_for_erof.remove("SCORE")
    e_r_o_f_TR = effect_range_otdel(train, list_features_for_erof, "TARGET")
    e_r_o_f_TS = effect_range_otdel(test, list_features_for_erof, "TARGET")
    # tab2.write([x for x in e_r_o_f_TR.Gini.tolist() if abs(x)>1 and abs(x)<=5])
    # tab2.write(len([x for x in e_r_o_f_TS.Gini.tolist() if abs(x)<=1]) > 0)

    if (
            len([x for x in e_r_o_f_TR.Gini.tolist() if abs(x) <= 1]) > 0
            or len([x for x in e_r_o_f_TS.Gini.tolist() if abs(x) <= 1]) > 0
    ):
        word_erof = "Красный"
        color_picker_erof = ":red"
    elif (
            len([x for x in e_r_o_f_TR.Gini.tolist() if (1 < abs(x) <= 5)]) > 0
            or len([x for x in e_r_o_f_TS.Gini.tolist() if (1 < abs(x) <= 5)])
            > 0
    ):
        word_erof = "Желтый"
        color_picker_erof = ":orange"
    else:
        word_erof = "Зеленый"
        color_picker_erof = ":green"
    tab2.write(
        "**М2.4 ЭФФЕКТИВНОСТЬ РАНЖИРОВАНИЯ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ. (ИНФОРМАТИВНЫЙ ТЕСТ ДЛЯ ИНТЕПРЕТИРУЕМЫХ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_erof
        + "[**{}**]".format(word_erof)
    )

    col1, col2 = tab2.columns(2)
    col1.header("Train")
    e_r_o_f_TR = e_r_o_f_TR.style.applymap(
        lambda x: (
            "background-color: red"
            if abs(x) <= 1
            else (
                "background-color: orange"
                if 1 < abs(x) <= 5
                else "background-color: green"
            )
        )
    )

    col1.dataframe(e_r_o_f_TR)

    col2.header("Test")
    e_r_o_f_TS = e_r_o_f_TS.style.applymap(
        lambda x: (
            "background-color: red"
            if abs(x) <= 1
            else (
                "background-color: orange"
                if 1 < abs(x) <= 5
                else "background-color: green"
            )
        )
    )
    col2.dataframe(e_r_o_f_TS)
    return word_erof, e_r_o_f_TR, e_r_o_f_TS


def m2_5_dinamic_gini(data_for_dyn, tab2, format_date):
    data_for_dyn.columns = [x.upper() for x in data_for_dyn.columns]
    model = pd.DataFrame()
    model["target_column"] = data_for_dyn.TARGET
    model["pred_column"] = data_for_dyn.SCORE
    model["date_column"] = pd.to_datetime(
        data_for_dyn.ISSUE_DATE, format=format_date
    )
    model["model_id"] = 1
    model["sub_model"] = 1

    mounth_dyn = gini_dynamic(model, time_slice="month")
    quart_dyn = gini_dynamic(model, time_slice="quarter")

    data_res_dyn = mounth_dyn[mounth_dyn.value_type == "gini"]
    if (
            len([x for x in data_res_dyn.value.to_list() if x <= 0.15])
            / len(data_res_dyn)
            >= 0.3
    ):

        color_picker_MD = ":red"
    elif (
            len([x for x in data_res_dyn.value.to_list() if x <= 0.15]) > 0
    ) or (
            (
                    len([x for x in data_res_dyn.value.to_list() if x <= 0.35])
                    / len(data_res_dyn)
            )
            >= 0.3
    ):

        color_picker_MD = ":orange"

    else:
        color_picker_MD = ":green"

    tab2.write(
        "**М2.5 ДИНАМИКА КОЭФФИЦИЕНТА ДЖИНИ (ПОМЕСЯЧНО). (ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_MD
        + "[**{}**]".format(res_gini_dyn(mounth_dyn))
    )
    tab2.dataframe(mounth_dyn)
    tab2.pyplot(gini_dynamic_graph(model, time_slice="month", figname='gini_dinamic_month_plot'))

    data_res_dyn_Q = quart_dyn[quart_dyn.value_type == "gini"]
    if (
            len([x for x in data_res_dyn_Q.value.to_list() if x <= 0.15])
            / len(data_res_dyn_Q)
            >= 0.3
    ):
        color_picker_QD = ":red"
    elif (
            len([x for x in data_res_dyn_Q.value.to_list() if x <= 0.15]) > 0
    ) or (
            (
                    len([x for x in data_res_dyn_Q.value.to_list() if x <= 0.35])
                    / len(data_res_dyn_Q)
            )
            >= 0.3
    ):
        color_picker_QD = ":orange"
    else:
        color_picker_QD = ":green"

    tab2.write(
        "**М2.5 ДИНАМИКА КОЭФФИЦИЕНТА ДЖИНИ (ПОКВАРТАЛЬНО). (ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_QD
        + "[**{}**]".format(res_gini_dyn(quart_dyn))
    )

    tab2.dataframe(quart_dyn)
    tab2.pyplot(gini_dynamic_graph(model, time_slice="quarter", figname="gini_dinamic_quart_plot"))
    return res_gini_dyn(mounth_dyn), res_gini_dyn(quart_dyn)


def m3_2(test, tab2, test_or_oot):
    features_for_stat = test.columns.tolist()
    features_for_stat.remove("DECIL")
    features_for_stat.remove("SCORE")
    features_for_stat.remove("TARGET")
    complited_train = test.copy()
    complited_train = complited_train.fillna(-3)
    complited_test = test.copy()
    complited_test = complited_test.fillna(-3)
    if "ISSUE_DATE" in features_for_stat:
        features_for_stat.remove("ISSUE_DATE")

    if test_or_oot == "oot":
        wald_test = wald_oot(
            complited_train, complited_test, "TARGET", features_for_stat
        )
    else:
        wald_test = wald(
            complited_train, complited_test, "TARGET", features_for_stat
        )

    tab2.write(
        "**М3.2 СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ ВЕСОВ ФАКТОРОВ (ЗНАЧИМЫЙ ТЕСТ. НЕ ПРЕМЕНИМ В НЕИНТЕРПРЕТИРУЕМЫХ МОДЕЛЯХ)-** "
        + wald_test[1]
        + "[**{}**]".format(wald_test[2])
    )
    tab2.dataframe(wald_test[0])
    return wald_test[2], wald_test[0]


def m3_3(test, tab2):
    test = test.copy()
    if "ISSUE_DATE" in list(test.columns):
        test = test.drop(columns="ISSUE_DATE")
    features_for_VIF = test.columns.tolist()
    features_for_VIF.remove("DECIL")
    features_for_VIF.remove("SCORE")
    vif = VIF(test, features_for_VIF)
    vif = vif.tail(-1)
    if for_result_vif(vif) == "Красный":
        color_picker = ":red"
    elif for_result_vif(vif) == "Желтый":
        color_picker = ":orange"
    else:
        color_picker = ":green"
    tab2.write(
        "**M3.3 ТЕСТ НА НАЛИЧИЕ МУЛЬТИКОЛЛИНЕАРНОСТИ (ЗНАЧИМЫЙ ТЕСТ)-** "
        + color_picker
        + "[**{}**]".format(for_result_vif(vif))
    )
    vif_df = vif.style.applymap(
        lambda x: (
            "background-color: red"
            if x == "красный"
            else (
                "background-color: orange"
                if x == "желтый"
                else "background-color: green"
            )
        ),
        subset=["оценка"],
    )
    tab2.dataframe(vif_df)
    return for_result_vif(vif), vif_df


def m4_1(test, tab2):
    PR_TR = prediction_fact_compare(test.TARGET, test.SCORE)
    if (
            PR_TR.P[0] > PR_TR.interval_green[0][0]
            and PR_TR.P[0] < PR_TR.interval_green[0][1]
    ):
        color_picker_PR_TR = ":green"
        word_PR_TR = "Зеленый"
    elif (
            PR_TR.P[0] > PR_TR.interval_yellow[0][0]
            and PR_TR.P[0] < PR_TR.interval_yellow[0][1]
    ):
        color_picker_PR_TR = ":yellow"
        word_PR_TR = "Желтый"
    else:
        color_picker_PR_TR = ":red"
        word_PR_TR = "Красный"
    tab2.write(
        "**M4.1 СРАВНЕНИЕ ПРОГНОЗНОГО И ФАКТИЧЕСКОГО TR[2] НА УРОВНЕ ВЫБОРКИ (ИНФОРМАТИВНЫЙ ТЕСТ)-** "
        + color_picker_PR_TR
        + "[**{}**]".format(word_PR_TR)
    )
    tab2.dataframe(PR_TR)
    return word_PR_TR, PR_TR


def m4_2_binom(train, test, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    binom = binomi(
        train,
        test,
        "SCORE",
        "DECIL",
        "TARGET",
        [[0.025, 0.975], [0.005, 0.995]],
    )[1]
    if result_binom(binom) == "Красный":
        color_picker = ":red"
    elif result_binom(binom) == "Желтый":
        color_picker = ":orange"
    else:
        color_picker = ":green"

    tab2.write(
        "**М4.2 ТОЧНОСТЬ КАЛИБРОВОЧНОЙ КРИВОЙ (БИНОМИАЛЬНЫЙ ТЕСТ). (ИНФОРМАТИВНЫЙ ТЕСТ)** "
        + color_picker
        + "[**{}**]".format(result_binom(binom))
    )
    binom_df = binom.style.applymap(
        lambda x: (
            "background-color: red"
            if x == "red"
            else (
                "background-color: orange"
                if x == "yellow"
                else "background-color: green"
            )
        ),
        subset=["overestimation", "underestimation"],
    )
    tab2.dataframe(binom_df)
    return result_binom(binom), binom_df


def m5_1(train, test, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    train_gini = metrics.roc_auc_score(train["TARGET"], train["SCORE"]) * 2 - 1
    test_gini = metrics.roc_auc_score(test["TARGET"], test["SCORE"]) * 2 - 1

    data_stable = pd.DataFrame(
        {
            "Train": [round(train_gini * 100, 2)],
            "Test": [round(test_gini * 100, 2)],
        }
    )
    data_stable["Абсолютное изменение"] = abs(
        data_stable.Train - data_stable.Test
    )
    data_stable["Относительное изменение"] = (
                                                     data_stable["Абсолютное изменение"] / data_stable.Train
                                             ) * 100

    if (
            data_stable["Абсолютное изменение"][0] >= 10
            and data_stable["Относительное изменение"][0] >= 20
    ):
        word = "Красный"
        color_picker_DS = ":red"
    elif (
            data_stable["Абсолютное изменение"][0] >= 5
            and data_stable["Относительное изменение"][0] >= 10
    ):
        word = "Желтый"
        color_picker_DS = ":orange"
    else:
        word = "Зеленый"
        color_picker_DS = ":green"

    tab2.write(
        "**М5.1 СРАВНЕНИЕ ЭФФЕКТИВНОСТИ РАНЖИРОВАНИЯ МОДЕЛИ ВО ВРЕМЯ РАЗРАБОТКИ И ВО ВРЕМЯ ВАЛИДАЦИИ. (ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_DS
        + "[**{}**]".format(word)
    )
    tab2.dataframe(data_stable)
    # tab2.pyplot(plot_diff_gini_all(data_stable))
    return word, data_stable


def m5_2(train, test, type_block, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    list_features_for_erof = list(train.columns)
    list_features_for_erof.remove("TARGET")
    list_features_for_erof.remove("DECIL")
    list_features_for_erof.remove("SCORE")

    abs_rel_df = abs_rel(
        train, test, list_features_for_erof, "TARGET", type_block
    )

    if "red" in abs_rel_df["Оценка"].tolist():
        word = "Красный"
        color_picker_DSr = ":red"
    elif (
            abs_rel_df["Оценка"].tolist().count("yellow") / abs_rel_df.shape[0]
            >= 0.1
    ):
        word = "Желтый"
        color_picker_DSr = ":orange"
    else:
        word = "Зеленый"
        color_picker_DSr = ":green"

    tab2.write(
        "**М5.2 СРАВНЕНИЕ ЭФФЕКТИВНОСТИ РАНЖИРОВАНИЕ ОТДЕЛЬНЫХ ФАКТОРОВ МОДЕЛИ ВО ВРЕМЯ РАЗРАБОТКИ И ВО ВРЕМЯ ВАЛИДАЦИИ. (ИНФОРМАТИВНЫЙ ТЕСТ ДЛЯ ИНТЕРПРЕТИРУЕМЫХ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + color_picker_DSr
        + "[**{}**]".format(word)
    )
    abs_rel_df = abs_rel_df.style.applymap(
        lambda x: (
            "background-color: red"
            if x == "red"
            else (
                "background-color: orange"
                if x == "yellow"
                else "background-color: green"
            )
        ),
        subset=["Оценка"],
    )
    tab2.dataframe(abs_rel_df)
    fig = plot_diff_gini(abs_rel(train, test, list_features_for_erof, "TARGET", type_block))
    tab2.pyplot(fig)
    return word, abs_rel_df


def coef_corr(train, test, tab2):
    train = train.copy()
    test = test.copy()
    if "ISSUE_DATE" in list(train.columns):
        train = train.drop(columns="ISSUE_DATE")
        test = test.drop(columns="ISSUE_DATE")
    df = pd.concat([train, test])
    df = df.drop(columns=["SCORE", "DECIL"])
    tab2.write("**КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ ПИРСОНА (ИНФОРМАТИВНЫЙ ТЕСТ)**")
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    fig.savefig('test_results/coef_corr', bbox_inches='tight')
    tab2.pyplot(fig)


def m3_1(train, test, type_block, tab2):
    X_feats = train.columns.tolist()
    try:
        X_feats.remove('TARGET')
        X_feats.remove('SCORE')
        X_feats.remove('DECIL')
        X_feats.remove('ISSUE_DATE')
    except:
        pass
    X_train = train[X_feats]
    y_train = train['TARGET']
    X_test = test[X_feats]
    y_test = test['TARGET']

    if type_block == 'Корпоративный блок':
        tab2.write(
            '**М3.1 АНАЛИЗ КОРРЕКТНОСТИ ДИСКРЕТНОГО ПРЕОБРАЗОВАНИЯ ФАКТОРОВ (ИНФОРМАТИВНЫЙ ТЕСТ) (НЕ ПРЕМЕНИМ В НЕИНТЕРПРЕТИРУЕМЫХ МОДЕЛЯХ)** ')
    elif type_block == 'Бизнес блок' or type_block == 'Розничный блок':
        tab2.write(
            '**М3.1 АНАЛИЗ КОРРЕКТНОСТИ ДИСКРЕТНОГО ПРЕОБРАЗОВАНИЯ ФАКТОРОВ (ЗНАЧИМЫЙ ТЕСТ) (НЕ ПРЕМЕНИМ В НЕИНТЕРПРЕТИРУЕМЫХ МОДЕЛЯХ)** ')
    tab2.write(str(len(X_feats)) + " не обновляй")

    cols = tab2.columns(len(X_feats))
    for i in range(len(X_feats)):
        cols[i].pyplot(plot_woe_bars(X_train, y_train, X_test, y_test, 'TARGET', X_feats[i]))


def m2_6(train, test, tab2):
    X_feats = train.columns.tolist()
    try:
        X_feats.remove("TARGET")
        X_feats.remove("SCORE")
        X_feats.remove("DECIL")
        X_feats.remove("ISSUE_DATE")
    except:
        pass
    X_train = train[X_feats]
    y_train = train["TARGET"]
    X_test = test[X_feats]
    y_test = test["TARGET"]

    prob_tr = train["SCORE"]  # Вероятности для обучающих данных
    prob_ts = test["SCORE"]  # Вероятности для тестовых данных
    thresh = 0.5
    tab2.write(
        "**M2.6 ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ (ОБЩЕЙ ЭФФЕКТИВНОСТИ) МОДЕЛИ (ИНФОРМАТИВНЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
    )

    train_color, test_color, accuracy_tr, accuracy_ts, fig = accuracy(y_train, y_test, prob_tr, prob_ts, thresh)
    tab2.pyplot(fig)
    if 'red' in [train_color, test_color]:
        color_picker = "Красный"
    elif 'yellow' in [train_color, test_color]:
        color_picker = "Желтый"
    else:
        color_picker = "Зеленый"
    return color_picker


def m2_7(train, test, type_block, type_model, tab2):
    X_feats = train.columns.tolist()
    try:
        X_feats.remove("TARGET")
        X_feats.remove("SCORE")
        X_feats.remove("DECIL")
        X_feats.remove("ISSUE_DATE")
    except:
        pass
    X_train = train[X_feats]
    y_train = train["TARGET"]
    X_test = test[X_feats]
    y_test = test["TARGET"]

    prob_tr = train["SCORE"]  # Вероятности для обучающих данных
    prob_ts = test["SCORE"]  # Вероятности для тестовых данных
    thresh = 0.5
    result, color = precision(y_test, prob_ts, thresh, type_block, type_model)
    tab2.write(
        "**М2.7 ОЦЕНКА ТОЧНОСТИ МОДЕЛИ В ОПРЕДЕЛЕНИИ ИСТИННО ПОЛОЖИТЕЛЬНЫХ РЕЗУЛЬТАТОВ (ИНФОРМАТИВНЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + f":{color}"
        + f"[**Precision: {result}**]"
    )
    if color == 'red':
        color = 'Красный'
    elif color == 'orange':
        color = 'Желтый'
    else:
        color = 'Зеленый'
    return result, color


def m2_8(train, test, type_block, type_model, tab2):
    X_feats = train.columns.tolist()
    try:
        X_feats.remove("TARGET")
        X_feats.remove("SCORE")
        X_feats.remove("DECIL")
        X_feats.remove("ISSUE_DATE")
    except:
        pass
    X_train = train[X_feats]
    y_train = train["TARGET"]
    X_test = test[X_feats]
    y_test = test["TARGET"]

    prob_tr = train["SCORE"]  # Вероятности для обучающих данных
    prob_ts = test["SCORE"]  # Вероятности для тестовых данных
    thresh = 0.5
    result, color = recall(y_test, prob_ts, thresh, type_block, type_model)
    tab2.write(
        "**M2.8 ОЦЕНКА ПОЛНОТЫ (ЧАСТОТЫ) МОДЕЛИ В ОПРЕДЕЛЕНИИ ИСТИННО ПОЛОЖИТЕЛЬНЫХ РЕЗУЛЬТАТОВ (ИНФОРМАТИВНЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + f":{color}"
        + f"[**Recall: {result}**]"
    )
    if color == 'red':
        color = 'Красный'
    elif color == 'orange':
        color = 'Желтый'
    else:
        color = 'Зеленый'
    return color, result


def m2_9(train, test, type_block, type_model, tab2):
    X_feats = train.columns.tolist()
    try:
        X_feats.remove("TARGET")
        X_feats.remove("SCORE")
        X_feats.remove("DECIL")
        X_feats.remove("ISSUE_DATE")
    except:
        pass
    X_train = train[X_feats]
    y_train = train["TARGET"]
    X_test = test[X_feats]
    y_test = test["TARGET"]

    prob_tr = train["SCORE"]  # Вероятности для обучающих данных
    prob_ts = test["SCORE"]  # Вероятности для тестовых данных
    thresh = 0.5
    result, color = f1_score_eval(y_test, prob_ts, thresh, type_block, type_model)
    tab2.write(
        "**M2.9 ОЦЕНКА СРЕДНЕГО ГАРМОНИЧЕСКОГО ЗНАЧЕНИЯ МЕЖДУ ТОЧНОСТЬЮ И ПОЛНОТОЙ МОДЕЛИ (ИНФОРМАТИВНЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + f":{color}"
        + f"[**F1-score: {result}**]"
    )
    if color == 'red':
        color = 'Красный'
    elif color == 'orange':
        color = 'Желтый'
    else:
        color = 'Зеленый'
    return color, result


def m2_10(train, test, tab2):
    X_feats = train.columns.tolist()
    try:
        X_feats.remove("TARGET")
        X_feats.remove("SCORE")
        X_feats.remove("DECIL")
        X_feats.remove("ISSUE_DATE")
    except:
        pass
    X_train = train[X_feats]
    y_train = train["TARGET"]
    X_test = test[X_feats]
    y_test = test["TARGET"]

    prob_tr = train["SCORE"]  # Вероятности для обучающих данных
    prob_ts = test["SCORE"]  # Вероятности для тестовых данных
    thresh = 0.5
    tab2.write(
        "**M2.10 ОЦЕНКА ЭФФЕКТИВНОСТИ МОДЕЛИ – МАТРИЦА ОШИБОК (ИНФОРМАТИВНЫЙ ТЕСТ)**"
    )
    color, fig, df = all_metrics(prob_tr, prob_ts, y_train, y_test, thresh)
    col1, col2 = tab2.columns(2)
    with col1:
        tab2.pyplot(fig)
    with col2:
        tab2.dataframe(df)
    return color, df


def m2_11(test, tab2):
    y_test = test["TARGET"]
    prob_ts = test["SCORE"]
    thresh = 0.5
    log_loss_value, mean_log_loss, color = log_loss(y_test, prob_ts, thresh)
    tab2.write(
        "**M2.11: ОЦЕНКА КАЧЕСТВА ВЕРОЯТНОСТНЫХ ПРЕДСКАЗАНИЙ (ЗНАЧИМЫЙ ТЕСТ)**"
        + " **РЕЗУЛЬТАТ ТЕСТА -** "
        + f":{color}"
        + f"[**Mean Log Loss: {mean_log_loss}**]"
    )
    if color == 'red':
        color = 'Красный'
    elif color == 'orange':
        color = 'Желтый'
    else:
        color = 'Зеленый'
    return color, mean_log_loss