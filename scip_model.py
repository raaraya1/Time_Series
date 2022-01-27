import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from io import StringIO
from pyscipopt import Model, quicksum

class scip_st():
    def __init__(self):
        self.dic_prices = None
        self.dic_rent_sigma = None
        self.dic_corr = None

        with st.expander('Parametros'):
            c1, c2 = st.columns(2)
            self.period = c2.selectbox('Periodos', options=['3y', '5y'])
            self.min_var = c2.number_input('min_var', 0.001, 1.0, 0.02)
            self.portfolio = c1.file_uploader('Subir archivo con portafolio de acciones (csv)')
            if self.portfolio is not None:
                 content = StringIO(self.portfolio.getvalue().decode("utf-8"))
                 content = content.read()
                 content = content.split(',')
                 content_fixed = [content[0]]
                 for i in content[1:-1]: content_fixed.append(i[2:])
                 content_fixed.append(content[-1][2:-2])

                 self.fix_portfolio = content_fixed

    def get_info(self):
        portafolio = self.fix_portfolio
        min_var = self.min_var
        period = self.period

        self.dic_rent_sigma = {i: self.R_and_Sigma(i, period)[:2] for i in portafolio}
        self.dic_prices = {i: self.R_and_Sigma(i, period)[2] for i in portafolio}
        self.dic_corr = {(i, j):np.corrcoef(self.dic_prices[i], self.dic_prices[j])[0][1] for i in portafolio for j in portafolio}

    def solve(self):

        portafolio = self.fix_portfolio
        min_var = self.min_var
        period = self.period

        self.get_info()

        # Modelo
        m = Model()

        # Variables
        w = {}
        for i in portafolio:
            w[i] = m.addVar(lb=0.0, ub=1.0, vtype='C')

        obj = m.addVar(vtype='C')

        # F.O.
        m.addCons(obj == quicksum(self.dic_rent_sigma[i][0]*w[i] for i in portafolio))

        # Restricciones
        m.addCons(quicksum(w[i]*w[j]*self.dic_rent_sigma[i][1]*self.dic_rent_sigma[j][1]*self.dic_corr[i, j] for i in portafolio for j in portafolio) <= min_var)

        m.addCons(quicksum(w[i] for i in portafolio) == 1.0)

        m.setObjective(obj, 'maximize')
        m.optimize()

        # mostrar resultados
        w_out = {i:m.getVal(w[i]) for i in w}
        self.plot_results(m.getVal(obj), w_out)

        return m.getVal(obj)

    def plot_results(self, z, w):
        c1, c2 = st.columns([4, 1])

        # Grafico
        df = pd.DataFrame(w.values(), index=w.keys(), columns=['c1'])
        cond = df.c1> 0.001
        df = df['c1'][cond]
        plot = df.plot.pie(y='c1',
                           title="Portafolio Optimo",
                           legend=False,
                           autopct='%1.1f%%',
                           shadow=True,
                           startangle=0,
                           figsize=(3, 3))

        fig = plot.get_figure()
        c1.pyplot(fig)

        # Metrica
        c2.metric('Rent Port', f'{np.round(z*100, 2)}%')



    def R_and_Sigma(self, nombre_accion, period='5y'):

        ticker = yf.Ticker(nombre_accion)
        df = ticker.history(period=period)

        df = df['Close'].to_frame()
        df['date'] = df.index
        df['Month-Year'] = df['date'].apply(lambda x:pd.Timestamp(x).strftime('%Y-%m'))

        df = df.groupby(by=['Month-Year']).mean().reset_index() # Valor promedio del mes
        df['Month-Year'] = pd.to_datetime(df['Month-Year'], infer_datetime_format=True)
        df = df.set_index(['Month-Year'])

        df1 = df['Close'].copy()
        df_Rt = (df1/df1.shift())-1
        df_Rt = df_Rt.dropna()

        df_Rt = df_Rt.to_frame()

        df_Rt['dates'] = df_Rt.index
        E_Rt = df_Rt['Close'].mean()
        sigma = df_Rt['Close'].std()

        return np.round(E_Rt, 3), np.round(sigma, 3), np.array(df['Close'])
