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
        Ahora último, he estado intrigado por como optimizar la toma de decisiones
        en la compra de acciones. Es por esta razón que mi meta en este trabajo será
        el de simular la rentabilidad que va adquiriendo un portafolio de acciones en el tiempo.

        Con este objetivo en mente, me gustaría poner en práctica un modelo básico
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

        Una de las formas de encontrar este conjunto de instrumentos financieros (acciones) de manera eficiente es por medio de un problema de optimización donde se considera una formulación dual:

        - 1) Determinar los pesos ($w_{i}$) **que minimizan la varianza** sujeto a un retorno requerido ($R_{P}^*$)


          - **F.O.**

            $$
            min \quad var(R_{p})=\sigma_{p}^{2} = \sum_{i=1}^{N}\sum_{j=1}^{N}w_{i}w_{j}\sigma_{i}\sigma_{j}\rho_{(i, j)}
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

        - 2) Consiste en determinar los pesos ($w_{i}$) que maximizan el retorno esperado ($R_{P}$) sujeto a un máximo riesgo ($R_{p}^{*}$)

          - **F.O.**

          $$
          max \quad R_{P} = \sum_{i=1}^{N}w_{i}\bar{R_{i}}
          $$

          - **S.A.**

          $$
          \sum_{i=1}^{N}\sum_{j=1}^{N}w_{i}w_{j}\sigma_{i}\sigma_{j}\rho_{(i, j)} \leq \sigma_{p}^{2}
          $$

          $$
          \sum_{i=1}^{N}w_{i} = 1
          $$

          $$
          w_{i} \geq 0 \quad \forall i \in (1, ..., N)
          $$


        ''')

    st.write('''
            Mirando en detalle el modelo básico de "Portfolio Management", podremos notar
            que la esperanza de la rentabilidad de la acción no es otra cosa que el
            promedio de las rentabilidades en un periodo de tiempo. Es justamente en esta parte que
            propongo hacer un cambio y utilizar la predicción de series temporales para
            el cálculo de la esperanza en la rentabilidad de la acción.

            Para testear si este cambio genera mayores rentabilidades en el tiempo, lo
            que podemos hacer es seleccionar, de manera semi aleatoria, un variado conjunto de
            empresas y luego realizar el siguiente experimento.

            - Obtener los datos de los precios de cierre de las empresas (fuente: **Yahoo Finance**)
            - Separar los datos en **train set (80%)** y **test set (20%)**
            - Definir **monto inicial** para invertir


            - Para cada periodo de tiempo, se realizan los siguientes cálculos según el modelo.


              - **Modelo básico (Portfolio Management)**
                - Calculo de las rentabilidades de las acciones
                - Calculo de la esperanza de las acciones por empresa
                  - Promedio de las acciones
                - Calculo de las desviaciones estándar (riesgo de la acción)
                - Resolver con Guroby


              - **Modelo Adaptado (Portfolio Management + Time Series)**
                - Extracción de información sobre el precio de las acciones
                - Definir modelo de series temporal
                  - Utilizar el train set y escoger el modelo que mejor se ajusta a los datos
                - Calculo de la esperanza de las acciones por empresa
                  - [Valor pronosticado (t+1) / valor cierre actual (t)] - 1
                - Calculo de las desviaciones estándar (riesgo de la acción)
                - Resolver con Guroby


            -  Evaluar para cada periodo de tiempo (test set), la toma de acciones de ambos modelos.

            ''')
