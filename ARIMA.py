import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs
import pmdarima as pm

class ARIMA_st():
    def __init__(self, df, train):
        self.df = df.to_frame()
        self.train = train
        self.test = 1.0 - self.train

    def solve(self):

        c1, c2, c3 = st.columns([1, 4, 1])
        c2.write('''
        ## ARIMA
        ''')

        # Generar conjuntos de entrenamiento y validacion
        split = int(np.round(len(self.df['Close'])*self.train, 0))
        df_train = self.df['Close'][:split]
        df_test = self.df['Close'][split:]
        df_train = df_train.to_frame()
        df_test = df_test.to_frame()

        c1, c2 = st.columns([6, 2])

        opcion = c2.radio('Parámetros', ['Manual', 'Optimizado'], key='ARIMA_')
        if opcion == 'Manual':
            p = int(c2.number_input('p', 0, None, value=0))
            d = int(c2.number_input('d', 0, None, value=0))
            q = int(c2.number_input('q', 0, None, value=0))
            m = ARIMA(df_train['Close'].values, order=(p, d, q)).fit()
            df_train['ARIMA'] = m.fittedvalues
        elif opcion == 'Optimizado':
            # Auto-ARIMA
            m = pm.auto_arima(df_train['Close'].values,
                            start_p=1,
                            start_q=1,
                            test='adf',
                            max_p=4,
                            max_q=4,
                            m=12,
                            d=None,
                            seasonal=False,
                            start_P=0,
                            D=0,
                            start_Q=0,
                            trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=False)
            arima_fit=m.fit(df_train['Close'].values)
            df_train['ARIMA'] = arima_fit.predict_in_sample(return_conf_int=False)

        predict_button = c1.checkbox('Pronosticar', key='pred_box')
        if predict_button:
            if opcion == 'Manual': df_train = self._pred(df_test, df_train, opcion, p, d, q)
            elif opcion == 'Optimizado': df_train = self._pred(df_test, df_train, opcion)

        # Grafico
        fig = plt.figure(figsize=(5, 4))
        plt.plot(df_train['Close'], label='Datos Reales', color='blue')
        plt.plot(df_train['ARIMA'][:split], label='Datos Entrenamiento', color='red')
        plt.plot(df_train['ARIMA'][split:], label='Datos Testeo', color='green')
        plt.legend(loc='best')
        plt.xlabel('Tiempo (Mes-Año)')
        plt.ylabel('Valor Promedio Mensual')

        c1.plotly_chart(fig)

        # Metricas
        y_true = df_train['Close'][split:]
        y_test = df_train['ARIMA'][split:]
        MAE_metric = np.round(self.MAE(y_true, y_test), 3)
        RMSE_metric = np.round(self.RMSE(y_true, y_test), 3)


        # Next Month Prediction
        if predict_button:
            if opcion == 'Manual':
                m = ARIMA(df_train['Close'].values, order=(p, d, q)).fit()
                NMP = m.predict(0)[-1]
            elif opcion == 'Optimizado':
                m = pm.auto_arima(df_train['Close'].values,
                                 start_p=1,
                                 start_q=1,
                                 test='adf',
                                 max_p=4,
                                 max_q=4,
                                 m=12,
                                 d=None,
                                 seasonal=False,
                                 start_Q=0,
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=False)
                NMP = m.predict(1)[0]
            NMP = np.round(NMP, 2)

            # Current Month Price
            CMP = np.round(df_test['Close'][-1], 2)

            # Delta
            Delta = f'{np.round(((NMP/CMP)-1)*100, 0)} %'


        c2.metric(label='MAE', value=MAE_metric)
        c2.metric(label='RMSE', value=RMSE_metric)
        if predict_button: c2.metric(label='Prediction Price', value=NMP, delta=Delta)


        # ACF
        acf = c1.checkbox('ACF', key='acf_arima')
        if acf:
            c1, c2 = st.columns([3, 1])
            df_acf = self.df['Close'].to_frame()
            fig1 = plot_acf(df_acf['Close'].values, fft=1)
            fig1.figsize = (5, 4)
            plt.title('ACF')
            plt.xlabel('lag')
            plt.ylabel('acf coefficient')
            c1.pyplot(fig1)
            c2.write('''
            ### Nota:
            - Podemos seleccionar el parámetro $q$ de un modelo $MA(q)$ mirando los
            puntos significantes del gráfico y corroborando que, luego en el lag $q$, se observe
            un descenso brusco de los puntos tendiendo a 0.
            ''')

        # PACF
        pacf = c1.checkbox('PACF', key='pacf_arima')
        if pacf:
            c1, c2 = st.columns([3, 1])
            df_pacf = self.df['Close'].to_frame()
            fig1 = plot_pacf(df_pacf['Close'].values)
            fig1.figsize = (5, 4)
            plt.title('PACF')
            plt.xlabel('lag')
            plt.ylabel('pacf coefficient')
            c1.pyplot(fig1)
            c2.write('''
            ### Nota:
            - Nosotros podemos ajustar el parámetro $p$ de un modelo $AR(p)$
            basado en los puntos significantes del gráfico. Un indicador de
            un modelo $AR$ se observa cuando los puntos en el grafico decaen
            lentamente.
            ''')

        # ndiffs
        ndiff = c1.checkbox('diff tests', key='diff tests')
        if ndiff:
            y = self.df['Close'].to_frame().values
            ## Adf Test
            c1.write(f"adf test: {ndiffs(y, test='adf')}")
            # KPSS test
            c1.write(f"kpss test: {ndiffs(y, test='kpss')}")
            # PP test:
            c1.write(f"pp test: {ndiffs(y, test='pp')}")

        # Resumen resultados
        summary = c1.checkbox('Resumen', key='summary')
        if summary: st.write(m.summary())


    def MAE(self, y_true, y_pred):
      return np.mean(np.abs(y_true - y_pred))

    def RMSE(self, y_true, y_pred):
      return np.sqrt(np.mean((y_true - y_pred)**2))

    #@st.cache
    def _pred(self, A, B, C, D=0, E=0, F=0):

        n_predictions = len(A['Close'])
        for pred in range(n_predictions):
         # agregamos una nueva fila con valores 0 ['Close', 'MA']
            B.loc[len(B)] = [0, 0]
            # Hacemos la prediccion
            if C == 'Manual':
                m = ARIMA(B['Close'].values, order=(D, E, F)).fit()
                B['ARIMA'][-1] = m.predict(0)[-1]
            elif C == 'Optimizado':
                m = pm.auto_arima(B['Close'][:-1].values,
                                 start_p=1,
                                 start_q=1,
                                 test='adf',
                                 max_p=4,
                                 max_q=4,
                                 m=12,
                                 d=None,
                                 seasonal=False,
                                 start_Q=0,
                                 start_P=0,
                                 D=0,
                                 trace=True,
                                 error_action='ignore',
                                 suppress_warnings=True,
                                 stepwise=False)
                #B['ARIMA'][-1] = m.fit_predict(B['Close'].values, n_periods=1)
                B['ARIMA'][-1] = m.predict(1)

            # Luego agregamos el valor guardado en test
            B['Close'][len(B['Close'])-1] = A['Close'][pred]

        B = B.set_index([self.df.index])
        return B
