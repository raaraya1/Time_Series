import streamlit as st
import pandas as pd
import yfinance as yf
from yahoo_finance import yf_st
from Moving_Average import MA_st
from Exponential_smooth import ES_st
from Exponential_Smoothing_Double import ESD_st
from Exponential_Smoothing_Triple import EST_st
from ARIMA import ARIMA_st

st.write('# TIME SERIES')
st.write('''
        Ahora ultimo, estaba interesado en invertir en la compra de acciones para luego poder
        manejar otra fuente de ingresos. Asi, mi objetivo en esta ocacion es simular
        la compra de acciones, para luego replicar las buenas tomas de decisiones
        en el mundo real.

        Con este objetivo en mente, me gustaria poner en practica un modelo basico
        de "Portafolio Managment" y, adicionalmente, apoyarme con modelos de series
        temporales para pronosticar el precio de cierre de las acciones.
        ''')

decision = st.sidebar.selectbox('Opciones', options=['Intro',
                                                    'Time Series Analysis',
                                                    'Portfolio Management',
                                                    'Portfolio Management + Time Series'])
if decision == 'Time Series Analysis':
    df = yf_st.menu()
    c1, c2, c3 = st.columns(3)
    MA = c1.checkbox('Movil Average')
    ES = c1.checkbox('Exponential Smoothing')
    ESD = c2.checkbox('Exponential Smoothing Double')
    EST = c2.checkbox('Exponential Smoothing Triple')
    _ARIMA = c3.checkbox('ARIMA')

    if MA: MA_st(df, train=0.8).solve()
    if ES: ES_st(df, train=0.8).solve()
    if ESD: ESD_st(df, train=0.8).solve()
    if EST: EST_st(df, train=0.8).solve()
    if _ARIMA: ARIMA_st(df, train=0.8).solve()

if decision == 'Portfolio Management': st.write('En Construccion')

if decision == 'Portfolio Management + Time Series': st.write('En Construccion')

if decision == 'Intro':

    with st.expander('Portafolio Managment Model'):
        st.write(r'''
        ## Formulación de un Portafolio Optimo

        Una de las formas de encontrar este conjunto de instrumentos financieros (acciones) de manera eficiente es por medio de un problema de optimizacion donde se considera una formulacion dual:

        - 1) Determinar los pesos ($w_{i}$) **que minimizan la variannza** sujeto a un retorno requerido ($R_{P}^*$)

          - **F.O.**

            $$
            min \quad var(R_{p})=\sigma_{p}^{2} = \sum_{i=1}^{N}\sum_{j=1}^{N}w_{i}w_{j}\sigma_{i}\sigma_{j}
            $$

          - **S.A.**

            $$
            \sum_{i=1}^{N} w_{i}\bar{R_{i}} \geq R_{p}^{*}
            $$

            $$
            \sum_{i=1}^{N} w_{i} = 1
            $$

            $$
            w_{i} \geq 0 \quad \forall i \in (1, ..., N)
            $$

          - Siendo $\bar{R_{i}}$ = Rendimiento esperado del activo $i$

        - 2) Consiste en determinar los pesos ($w_{i}$) que maximizan el retorno esperado ($R_{P}$) sujeto a un maximo riesgo ($R_{p}^{*}$)

          - **F.O.**

          $$
          max \quad R_{P} = \sum_{i=1}^{N}w_{i}\bar{R_{i}}
          $$

          - **S.A.**

          $$
          \sum_{i=1}^{N}\sum_{j=1}^{N}w_{i}w_{j}\sigma_{i}\sigma_{j} \leq \sigma_{p}^{2}
          $$

          $$
          \sum_{i=1}^{N}w_{i} = 1
          $$

          $$
          w_{i} \geq 0 \quad \forall i \in (1, ..., N)
          $$


        ''')

    st.write('''
            Mirando en detalle el modelo basico de "Portfolio Management", podremos notar
            que la esperaza de la rentabilidad de la accion, no es otra cosa que el
            promedio de las rentabilidades en un periodo de tiempo. Es justamente en esta que
            propongo hacer un cambio y utilizar la prediccion de series temporales para
            el calculo de la esperanza en la rentabilidad de la accion.

            Para testear esta hipotesis, lo que podemos hacer es seleccionar, de manera
            aleatoria, un conjunto de empresas y luego realizar el siguiente experimento.

            - Obtener los datos de los precios de cierre de las empresas (fuente: **Yahoo Finance**)
            - Separar los datos en **train set (80%)** y **test set (20%)**
            - Definir **monto inicial** para invertir


            - Para cada periodo de tiempo, se realizan los siguentes calculos segun el modelo.


              - **Modelo basico (Portfolio Management)**
                - Calculo de las rentabilidades de las acciones
                - Calculo de la esperanza de las acciones por empresa
                  - Promedio de las acciones
                - Calculo de las desviaciones estandar (riesgo de la accion)
                - Resolver con Guroby y guardar los resultados


              - **Modelo Adaptado (Portfolio Management + Time Series)**
                - Calculo de las rentabilidades de las acciones
                - Definir modelo de series temporal
                  - Utilizar el train set y escojer el modelo que mejor se ajusta a los datos
                - Calculo de la esperanza de las acciones por empresa
                  - [Valor pronosticado (t+1) / valor cierre actual (t)] - 1
                - Calculo de las desviaciones estandar (riesgo de la accion)
                - Resolver con Guroby y guardar los resultados


            -  Evaluar para cada periodo de tiempo (test set), la toma de acciones de ambos modelos.

            ''')
