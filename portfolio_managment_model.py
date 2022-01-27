import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from io import StringIO
from gurobipy import Model, quicksum, GRB

class MPT_st():
    def __init__(self):
        self.dic_prices = None
        self.dic_rent_sigma = None
        self.dic_corr = None
        self.period = st.selectbox('Periodos', options=['3y', '5y'])
        self.min_var = st.number_input('min_var', 0.001, 1.0, 0.02)
        self.portfolio = st.file_uploader('Subir archivo con portafolio de acciones (csv)')
        if self.portfolio is not None:
             content = StringIO(self.portfolio.getvalue().decode("utf-8"))
             content = content.read()
             content = content.split(',')
             content_fixed = [content[0]]
             for i in content[1:-1]: content_fixed.append(i[2:])
             content_fixed.append(content[-1][2:-2])

             self.fix_portfolio = content_fixed



    def solve(self):
        portafolio = self.fix_portfolio
        min_var = self.min_var
        period = self.period

        self.dic_rent_sigma = {i: self.R_and_Sigma(i, period)[:2] for i in portafolio}
        self.dic_prices = {i: self.R_and_Sigma(i, period)[2] for i in portafolio}
        self.dic_corr = {(i, j):np.corrcoef(self.dic_prices[i], self.dic_prices[j])[0][1] for i in portafolio for j in portafolio}


        # Modelo
        m = Model()

        # Variables
        w = {}
        for i in portafolio:
            w[i] = m.addVar(lb=0.0, ub=1.0, vtype='C')

        obj = m.addVar(vtype='C')

        # F.O.
        m.addConstr(obj == quicksum(self.dic_rent_sigma[i][0]*w[i] for i in portafolio))

        # Restricciones
        m.addConstr(quicksum(w[i]*w[j]*self.dic_rent_sigma[i][1]*self.dic_rent_sigma[j][1]*self.dic_corr[i, j] for i in portafolio for j in portafolio) <= min_var)

        m.addConstr(quicksum(w[i] for i in portafolio) == 1.0)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.optimize()

        st.write(obj.x)

        return obj.x

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
