import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_pacf

class EST_st():
    def __init__(self, df, train):
        self.df = df.to_frame()
        self.train = train
        self.test = 1.0 - self.train

    def solve(self):

        c1, c2, c3 = st.columns([1, 4, 1])
        c2.write('''
        ## Suavización Exponencial Triple (Holt-Winter)
        ''')

        # Generar conjuntos de entrenamiento y validacion
        split = int(np.round(len(self.df['Close'])*self.train, 0))
        df_train = self.df['Close'][:split]
        df_test = self.df['Close'][split:]
        df_train = df_train.to_frame()
        df_test = df_test.to_frame()

        c1, c2 = st.columns([6, 2])

        opcion = c2.radio('Parámetros', ['Manual', 'Optimizado'], key='EST')
        if opcion == 'Manual':
            smooth_level = float(c2.slider('smooth_level', 0.0, 1.0, 0.3, key='slider1_est'))
            smooth_slope = float(c2.slider('smooth_slope', 0.0, 1.0, 0.3, key='slider2_est'))
            seasonal = int(c2.slider('Estacionalidad', 1, 12, 6, key='slider3_est'))
            m = ExponentialSmoothing(df_train['Close'].values, seasonal_periods=seasonal, trend='add', seasonal='add').fit(smoothing_level=smooth_level, smoothing_slope=smooth_slope)
        elif opcion == 'Optimizado':
            seasonal = int(c2.slider('Estacionalidad', 1, 12, 6, key='slider4_est'))
            m = ExponentialSmoothing(df_train['Close'].values, trend='add', seasonal='add', seasonal_periods=seasonal).fit(optimized=True)

        df_train['EST'] = m.fittedvalues

        #st.dataframe(df_train)
        n_predictions = len(df_test['Close'])
        for pred in range(n_predictions):
            # agregamos una nueva fila con valores 0 ['Close', 'MA']
            df_train.loc[len(df_train)] = [0, 0]
            # Hacemos la prediccion
            if opcion == 'Manual': m = ExponentialSmoothing(df_train['Close'].values, seasonal_periods=seasonal, trend='add', seasonal='add').fit(smoothing_level=smooth_level, smoothing_slope=smooth_slope)
            elif opcion == 'Optimizado': m = ExponentialSmoothing(df_train['Close'].values, trend='add', seasonal='add', seasonal_periods=seasonal).fit(optimized=True)
            df_train['EST'][-1] = m.predict(0)[-1]
            # Luego agregamos el valor guardado en test
            df_train['Close'][len(df_train['Close'])-1] = df_test['Close'][pred]

        df_train = df_train.set_index([self.df.index])

        # Grafico
        fig = plt.figure(figsize=(5, 4))
        plt.plot(df_train['Close'], label='Datos Reales', color='blue')
        plt.plot(df_train['EST'][:split], label='Datos Entrenamiento', color='red')
        plt.plot(df_train['EST'][split:], label='Datos Testeo', color='green')
        plt.legend(loc='best')
        plt.xlabel('Tiempo (Mes-Año)')
        plt.ylabel('Valor Promedio Mensual')

        c1.plotly_chart(fig)

        # Metricas
        y_true = df_train['Close'][split:]
        y_test = df_train['EST'][split:]
        MAE_metric = np.round(self.MAE(y_true, y_test), 3)
        RMSE_metric = np.round(self.RMSE(y_true, y_test), 3)

#
        # Next Month Prediction
        NMP = m.predict(0)[-1]
        NMP = np.round(NMP, 2)

        # Current Month Price
        CMP = np.round(df_test['Close'][-1], 2)

        # Delta
        Delta = f'{np.round(((NMP/CMP)-1)*100, 0)} %'
#


        c2.metric(label='MAE', value=MAE_metric)
        c2.metric(label='RMSE', value=RMSE_metric)
        c2.metric(label='Prediction Price', value=NMP, delta=Delta)

        # PACF
        pacf = c1.checkbox('PACF', key='pacf_est')
        if pacf:
            c1, c2 = st.columns([3, 1])
            df_pacf = self.df['Close'].to_frame()
            fig1 = plot_pacf(df_pacf['Close'].values, fft=1)
            fig1.figsize = (5, 4)
            plt.title('PACF')
            plt.xlabel('Periodos')
            plt.ylabel('Autocorrelacion')
            c1.pyplot(fig1)
            c2.write('''
            ### Nota:
            - El grafico de Autocorrelaciones Parciales nos puede ayudar a
            determinar la estacionalidad de la serie, al **observar cada cuanto se repite
            un patrón en el tiempo**.

            ''')
    def MAE(self, y_true, y_pred):
      return np.mean(np.abs(y_true - y_pred))

    def RMSE(self, y_true, y_pred):
      return np.sqrt(np.mean((y_true - y_pred)**2))
