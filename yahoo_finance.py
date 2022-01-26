import streamlit as st
import pandas as pd
import yfinance as yf
import urllib.request
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class yf_st:
    def menu():
        ticker_name = st.text_input('Código de la acción')
        if ticker_name != '':
            c1, c2 = st.columns(2)

            ticker = yf.Ticker(ticker_name)
            info = ticker.info
            name = info['longName']
            industry = info['industry']
            country = info['country']
            current_price = info['currentPrice']


            logo_url = info['logo_url']
            urllib.request.urlretrieve(logo_url, "logo.png")
            img = Image.open("logo.png")

            c1.image(img)
            c2.write(f'''**Nombre:** {name}''')
            c2.write(f'''**Industria:** {industry}''')
            c2.write(f'''**Pais:** {country}''')
            c2.write(f'''**Precio Actual:** $ {current_price}''')

            Mostrar_graficos = c2.checkbox('Graficos')

            a1, a2 = st.columns([5, 1])
            period = a2.selectbox('Periodo', options=[
                        '5d','1mo','3mo','6mo',
                        '1y','2y','5y'])
            df = ticker.history(period=period)
            if Mostrar_graficos:
                ax = df['Close'].plot(title=f"{name} stock price")
                fig = ax.get_figure()
                a1.plotly_chart(fig)

            # ---------Calculo de rentabilidades------
            #    -------------- antes ----------------------------
            df = df['Close'].to_frame()
            df['date'] = df.index
            df['Month-Year'] = df['date'].apply(lambda x:pd.Timestamp(x).strftime('%Y-%m'))

            # ANTES
            #df = df.groupby(by=['Month-Year']).tail(1).reset_index() #ultimo valor del mes

            # NUEVO
            df = df.groupby(by=['Month-Year']).mean().reset_index() # Valor promedio del mes

            df['Month-Year'] = pd.to_datetime(df['Month-Year'], infer_datetime_format=True)
            df = df.set_index(['Month-Year'])
            df = df['Close']

            # NUEVO
            df_output = df.copy()
            df_output = df_output.to_frame()
            #st.dataframe(df_output)


            # calculo de rentabilidades (y_{t}/y_{t-1})-1
            df_Rt = (df/df.shift())-1
            df_Rt = df_Rt.dropna()

            df_plot_Rt = df_Rt.copy()
            df_plot_Rt = df_plot_Rt.to_frame()
            df_plot_Rt['dates'] = df_plot_Rt.index

            fig1 = plt.figure()
            plt.xlabel('Date')
            plt.ylabel('Rentabilidad')
            plt.plot(df_plot_Rt['Close'])

            E_Rt = df_plot_Rt['Close'].mean()
            sigma = df_plot_Rt['Close'].std()

            c1, c2 = st.columns([5, 1])

            if Mostrar_graficos:
                c1.plotly_chart(fig1)
                c2.metric(label='Rent Esp', value=np.round(E_Rt, 2))
                c2.metric(label='Desv Est', value=np.round(sigma, 2))

            st.write('''
            ## Analisis de Series Temporales
            ''')

            # ANTES
            #return df_plot_Rt['Close']

            # NUEVO
            return df_output['Close']
