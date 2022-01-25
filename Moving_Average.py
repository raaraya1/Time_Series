import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MA_st():
    def __init__(self, df, train):
        self.df = df.to_frame()
        self.train = train
        self.test = 1.0 - self.train

    def solve(self):

        c1, c2, c3 = st.columns([1, 4, 1])
        c2.write('''
        ## Medias Moviles
        ''')
        # Generar conjuntos de entrenamiento y validacion
        split = int(np.round(len(self.df['Close'])*self.train, 0))
        df_train = self.df['Close'][:split]
        df_test = self.df['Close'][split:]
        df_train = df_train.to_frame()
        df_test = df_test.to_frame()

        c1, c2 = st.columns([6, 2])

        window = int(c2.slider('Ventana Movil', 1, 5, 3))
        df_train['MA'] = df_train['Close'].rolling(window=window).mean().shift()

        #st.dataframe(df_train)
        n_predictions = len(df_test['Close'])
        for pred in range(n_predictions):
            # agregamos una nueva fila con valores 0 ['Close', 'MA']
            df_train.loc[len(df_train)] = [0, 0]
            # Hacemos la prediccion
            df_train['MA'] = df_train['Close'].rolling(window=window).mean().shift(1)
            # Luego agregamos el valor guardado en test
            df_train['Close'][len(df_train['Close'])-1] = df_test['Close'][pred]

        df_train = df_train.set_index([self.df.index])

        # Grafico
        fig = plt.figure(figsize=(5, 4))
        plt.plot(df_train['Close'], label='Datos Reales', color='blue')
        plt.plot(df_train['MA'][:split], label='Datos Entrenamiento', color='red')
        plt.plot(df_train['MA'][split:], label='Datos Testeo', color='green')
        plt.legend(loc='best')
        plt.xlabel('Tiempo (Mes-Año)')
        plt.ylabel('Valor Promedio Mensual')

        c1.plotly_chart(fig)


        # Metricas
        y_true = df_train['Close'][split:]
        y_test = df_train['MA'][split:]
        MAE_metric = np.round(self.MAE(y_true, y_test), 3)
        RMSE_metric = np.round(self.RMSE(y_true, y_test), 3)

        # Next Month Prediction
        NMP = df_test['Close'][int(n_predictions-window):]
        NMP = np.round(NMP.mean(), 2)

        # Current Month Price
        CMP = np.round(df_test['Close'][-1], 2)

        # Delta
        Delta = f'{np.round(((NMP/CMP)-1)*100, 0)} %'

        c2.metric(label='MAE', value=MAE_metric)
        c2.metric(label='RMSE', value=RMSE_metric)
        c2.metric(label='Prediction Price', value=NMP, delta=Delta)


    def MAE(self, y_true, y_pred):
      return np.mean(np.abs(y_true - y_pred))

    def RMSE(self, y_true, y_pred):
      return np.sqrt(np.mean((y_true - y_pred)**2))
