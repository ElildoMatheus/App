from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
from sklearn.metrics import r2_score, mean_absolute_error
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import DataConversionWarning
import warnings
import streamlit as st
import pandas as pd
st.set_page_config(page_title="ESTIMATIVA DE CUSTO")

# lendo a base de dados
baseR01 = pd.read_excel("BD1.xlsx")
baseR01 = baseR01.drop(['Obra'], axis=1)

st.title("ESTIMATIVA DE CUSTO")
st.sidebar.title("Selecione os parâmetros do projeto")

bancadas = st.sidebar.selectbox("Padrão de bancadas", ["Alto", "Médio", "Baixo"])
revestimentos = st.sidebar.selectbox("Padrão de revestimentos", ["Alto", "Médio", "Baixo"])
loucas = st.sidebar.selectbox("Padrão de louças", ["Alto", "Médio", "Baixo"])
metais = st.sidebar.selectbox("Padrão de metais", ["Alto", "Médio", "Baixo"])
esquadrias = st.sidebar.selectbox("Padrão de esquadrias", ["Alto", "Médio", "Baixo"])
fundação = st.sidebar.selectbox("Tipo de fundação",["Sapata", "Estacão","Hélice"])
Elevadores = st.sidebar.number_input("Quantidade de elevadores")
uni = st.sidebar.number_input("Total de unidades privativas")
indice_aço = st.sidebar.number_input("Indice de aço (Kg/m²)")
indice_concreto = st.sidebar.number_input("Indice de concreto (m³/m²)")
indice_forma = st.sidebar.number_input("Indice de forme (m²/m²)")
ic = st.sidebar.number_input("IC Atual")
prazo = st.sidebar.number_input("Prazo de execução")
subsolos = st.sidebar.number_input("Quantidade de subsolos")

areaequivalente = 23660.249
areaconstrutiva = 24323.73
areaprivativa = 15725.45
perimetroterreno = 150
tipo = 27

uni = 88
areaprivativauni = (areaprivativa)/uni

indice_aço = 24.941
indice_concreto = 0.240
indice_forma = 1.90

aço = indice_aço*areaconstrutiva
concreto = indice_concreto*areaconstrutiva
forma = indice_forma*areaconstrutiva

subsolos = 3
fundação = 3

bancadas = 2
revestimentos = 2
loucas = 2
metais = 2
esq = 2
elevadores = 3

prazo = 37

ic= 3.0151
icbase = 3.00304384793037
orçamento_base = 75990005.5081899
orçamento_real = orçamento_base*ic/icbase

# AREA

# Avaliação
x_train = baseR01[['ÁREA EQUIVALENTE']]
y_train = baseR01['ORÇAMENTO1']

# Inicializar os modelos
linear_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()
rf_model = RandomForestRegressor()
xgb_model = XGBRegressor()
dt_model = DecisionTreeRegressor()
poly = PolynomialFeatures(degree=2)

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[areaequivalente]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred1_linear = linear_model.predict(novo_dado)
pred1_lasso = lasso_model.predict(novo_dado)
pred1_ridge = ridge_model.predict(novo_dado)
pred1_rf = rf_model.predict(novo_dado)
pred1_xgb = xgb_model.predict(novo_dado)
pred1_dt = dt_model.predict(novo_dado)
pred1_poly = poly_model.predict(novo_dado_poly)

# ESTRUTURA

# Avaliação
x_train = baseR01[['AÇO','CONCRETO','FORMA']]
y_train = baseR01['ORÇAMENTO1']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[aço,concreto,forma]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred2_linear = linear_model.predict(novo_dado)
pred2_lasso = lasso_model.predict(novo_dado)
pred2_ridge = ridge_model.predict(novo_dado)
pred2_rf = rf_model.predict(novo_dado)
pred2_xgb = xgb_model.predict(novo_dado)
pred2_dt = dt_model.predict(novo_dado)
pred2_poly = poly_model.predict(novo_dado_poly)

# BANCADAS

# Avaliação
x_train = baseR01[['PADRÃO DE BANCADAS','ÁREA PRIVATIVA2']]
y_train = baseR01['BANCADAS']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[bancadas,areaprivativauni]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred3_linear = linear_model.predict(novo_dado)
pred3_lasso = lasso_model.predict(novo_dado)
pred3_ridge = ridge_model.predict(novo_dado)
pred3_rf = rf_model.predict(novo_dado)
pred3_xgb = xgb_model.predict(novo_dado)
pred3_dt = dt_model.predict(novo_dado)
pred3_poly = poly_model.predict(novo_dado_poly)

# REVESTIMENTOS

# Avaliação
x_train = baseR01[['PADRÃO REVESTIMENTO','ÁREA PRIVATIVA2']]
y_train = baseR01['REVESTIMENTOS']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[revestimentos,areaprivativauni]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred4_linear = linear_model.predict(novo_dado)
pred4_lasso = lasso_model.predict(novo_dado)
pred4_ridge = ridge_model.predict(novo_dado)
pred4_rf = rf_model.predict(novo_dado)
pred4_xgb = xgb_model.predict(novo_dado)
pred4_dt = dt_model.predict(novo_dado)
pred4_poly = poly_model.predict(novo_dado_poly)

# LOUÇAS

# Avaliação
x_train = baseR01[['PADRÃO LOUÇA','ÁREA PRIVATIVA2']]
y_train = baseR01['LOUCAS']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[loucas,areaprivativauni]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred5_linear = linear_model.predict(novo_dado)
pred5_lasso = lasso_model.predict(novo_dado)
pred5_ridge = ridge_model.predict(novo_dado)
pred5_rf = rf_model.predict(novo_dado)
pred5_xgb = xgb_model.predict(novo_dado)
pred5_dt = dt_model.predict(novo_dado)
pred5_poly = poly_model.predict(novo_dado_poly)

# METAIS

# Avaliação
x_train = baseR01[['PADRÃO METAIS','ÁREA PRIVATIVA2']]
y_train = baseR01['METAIS']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[metais,areaprivativauni]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred6_linear = linear_model.predict(novo_dado)
pred6_lasso = lasso_model.predict(novo_dado)
pred6_ridge = ridge_model.predict(novo_dado)
pred6_rf = rf_model.predict(novo_dado)
pred6_xgb = xgb_model.predict(novo_dado)
pred6_dt = dt_model.predict(novo_dado)
pred6_poly = poly_model.predict(novo_dado_poly)

# ESQUADRIAS DE ALUMINIO

# Avaliação
x_train = baseR01[['PADRÃO DE ESQUADRIA','ÁREA PRIVATIVA2']]
y_train = baseR01['ESQUADRIAS DE ALUMINIO']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[esq,tipo]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred7_linear = linear_model.predict(novo_dado)
pred7_lasso = lasso_model.predict(novo_dado)
pred7_ridge = ridge_model.predict(novo_dado)
pred7_rf = rf_model.predict(novo_dado)
pred7_xgb = xgb_model.predict(novo_dado)
pred7_dt = dt_model.predict(novo_dado)
pred7_poly = poly_model.predict(novo_dado_poly)

# INSTALAÇÕES

# Avaliação
x_train = baseR01[['UNIDADES']]
y_train = baseR01['INSTALAÇÕES']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[uni]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred8_linear = linear_model.predict(novo_dado)
pred8_lasso = lasso_model.predict(novo_dado)
pred8_ridge = ridge_model.predict(novo_dado)
pred8_rf = rf_model.predict(novo_dado)
pred8_xgb = xgb_model.predict(novo_dado)
pred8_dt = dt_model.predict(novo_dado)
pred8_poly = poly_model.predict(novo_dado_poly)

# ELEVADORES

# Avaliação
x_train = baseR01[['NºELEVADORES']]
y_train = baseR01['ELEVADORES']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[elevadores]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred9_linear = linear_model.predict(novo_dado)
pred9_lasso = lasso_model.predict(novo_dado)
pred9_ridge = ridge_model.predict(novo_dado)
pred9_rf = rf_model.predict(novo_dado)
pred9_xgb = xgb_model.predict(novo_dado)
pred9_dt = dt_model.predict(novo_dado)
pred9_poly = poly_model.predict(novo_dado_poly)

# FUNDAÇÃO

# Avaliação
x_train = baseR01[['SUBSOLOS','ÁREA CONSTRUTIVA','TIPO DE FUNDAÇÃO','PERIMETRO DO TERRENO']]
y_train = baseR01['FUNDAÇÃO']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[subsolos,areaconstrutiva,fundação,perimetroterreno]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred10_linear = linear_model.predict(novo_dado)
pred10_lasso = lasso_model.predict(novo_dado)
pred10_ridge = ridge_model.predict(novo_dado)
pred10_rf = rf_model.predict(novo_dado)
pred10_xgb = xgb_model.predict(novo_dado)
pred10_dt = dt_model.predict(novo_dado)
pred10_poly = poly_model.predict(novo_dado_poly)

# CUSTOS DE PRAZO
# Avaliação
x_train = baseR01[['PRAZO','ÁREA EQUIVALENTE']]
y_train = baseR01['INDIRETOS']

# Treinar os modelos
linear_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)
rf_model.fit(x_train, y_train)
xgb_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

# Transformar os dados para regressão polinomial
x_train_poly = poly.fit_transform(x_train)

# Inicializar o modelo de regressão polinomial
poly_model = LinearRegression()

# Treinar o modelo de regressão polinomial
poly_model.fit(x_train_poly, y_train)

# Preparar os dados de entrada para previsão
novo_dado = [[prazo, areaequivalente]]

# Transformar os novos dados para regressão polinomial
novo_dado_poly = poly.transform(novo_dado)

# Fazer previsões
pred11_linear = linear_model.predict(novo_dado)
pred11_lasso = lasso_model.predict(novo_dado)
pred11_ridge = ridge_model.predict(novo_dado)
pred11_rf = rf_model.predict(novo_dado)
pred11_xgb = xgb_model.predict(novo_dado)
pred11_dt = dt_model.predict(novo_dado)
pred11_poly = poly_model.predict(novo_dado_poly)

# Tabela de resultados dos métodos orçamentos total
modelos = ['Regressão Linear','Lasso','Ridge','Random Forest','XGBoost','Árvore de Decisão','Regressão Polinomial']
area = [pred1_linear, pred1_lasso, pred1_ridge, pred1_rf, pred1_xgb, pred1_dt, pred1_poly]
estrutura = [pred2_linear, pred2_lasso, pred2_ridge, pred2_rf, pred2_xgb, pred2_dt, pred2_poly]
bancadas = [pred3_linear*uni, pred3_lasso*uni, pred3_ridge*uni, pred3_rf*uni, pred3_xgb*uni, pred3_dt*uni, pred3_poly*uni]
revestimentos = [pred4_linear*uni, pred4_lasso*uni, pred4_ridge*uni, pred4_rf*uni, pred4_xgb*uni, pred4_dt, pred4_poly*uni]
louças = [pred5_linear*uni, pred5_lasso*uni, pred5_ridge*uni, pred5_rf*uni, pred5_xgb*uni, pred5_dt*uni, pred5_poly*uni]
metais = [pred6_linear*uni, pred6_lasso*uni, pred6_ridge*uni, pred6_rf*uni, pred6_xgb*uni, pred6_dt*uni, pred6_poly*uni]
esquadrias = [pred7_linear*tipo, pred7_lasso*tipo, pred7_ridge*tipo, pred7_rf*tipo, pred7_xgb*tipo, pred7_dt*tipo, pred7_poly*tipo]
instalações = [pred8_linear, pred8_lasso, pred8_ridge, pred8_rf, pred8_xgb, pred8_dt, pred8_poly]
elevadores = [pred9_linear, pred9_lasso, pred9_ridge, pred9_rf, pred9_xgb, pred9_dt, pred9_poly]
Fundação = [pred10_linear, pred10_lasso, pred10_ridge, pred10_rf, pred10_xgb, pred10_dt, pred10_poly]
Indiretos = [pred11_linear, pred11_lasso, pred11_ridge, pred11_rf, pred11_xgb, pred11_dt, pred11_poly]

# Extrair apenas o primeiro elemento de cada lista
area = [valor[0] for valor in area]
estrutura = [valor[0] for valor in estrutura]
bancadas = [valor[0] for valor in bancadas]
revestimentos = [valor[0] for valor in revestimentos]
louças = [valor[0] for valor in louças]
metais = [valor[0] for valor in metais]
esquadrias = [valor[0] for valor in esquadrias]
instalações = [valor[0] for valor in instalações]
elevadores = [valor[0] for valor in elevadores]
Fundação = [valor[0] for valor in Fundação]
Indiretos = [valor[0] for valor in Indiretos]

# Criar DataFrame
data = {'Modelos': modelos,'Área': area,'Estrutura': estrutura, 'Bancadas': bancadas, 'Revestimento': revestimentos, 'Louças': louças, 'Metais': metais, 'Esquadrias': esquadrias, 'Instalações': instalações, 'Elevadores': elevadores, 'Fundação': Fundação, 'Indiretos': Indiretos}
tabela1 = pd.DataFrame(data)

# Adicionar as médias ao final de cada coluna
tabela1.loc[len(tabela1)] = ['Média', tabela1['Área'].mean(), tabela1['Estrutura'].mean(), tabela1['Bancadas'].mean(), tabela1['Revestimento'].mean(), tabela1['Louças'].mean(), tabela1['Metais'].mean(), tabela1['Esquadrias'].mean(), tabela1['Instalações'].mean(), tabela1['Elevadores'].mean(),tabela1['Fundação'].mean(),tabela1['Indiretos'].mean() ]

# Definir os dados
revestimentos = float((pred4_linear*uni + pred4_lasso*uni + pred4_ridge*uni)/3)
instalações = float(pred8_poly)
louças = float((pred5_linear*uni + pred5_lasso*uni + pred5_ridge*uni)/3)
metais = float((pred6_linear*uni + pred6_lasso*uni + pred6_ridge*uni)/3)
bancadas = float(pred3_linear*uni + pred3_lasso*uni + pred3_ridge*uni)/3
area = float(tabela1['Área'].mean()+tabela1['Estrutura'].mean())/2
elevadores = float((pred9_rf + pred9_xgb + pred9_dt)/3)
infraestrutura = float((pred10_linear + pred10_lasso + pred10_ridge)/3)
custosdeprazo = float((pred11_linear + pred11_lasso + pred11_ridge)/3)
orcamento = custosdeprazo + revestimentos + instalações + louças + metais + bancadas + area + elevadores + infraestrutura + tabela1["Esquadrias"].mean()
orcamento_previsto = orcamento*1.01

st.write("ORÇAMENTO PREVISTO: R$", f'{"{:,.2f}".format(orcamento_previsto)}')
st.write("ORÇAMENTO PREVISTO: IC$", f'{"{:,.2f}".format(orcamento_previsto/ic)}')

dados = {
    'Descrição': [
        'CUSTOS INDIRETOS',
        '01.01 - SERVICOS PRELIMINARES E GERAIS',
        '01.02 - IMPLANTAÇÃO DA OBRA',
        '01.03 - EQUIPAMENTOS/FERRAMENTAS',
        '01.04 - CUSTOS ADMINISTRATIVOS',
        '01.05 - LIMPEZA E DESMOBILIZAÇÃO',
        '01.06 - OUTRAS DESPESAS DE OBRA',
        'CUSTOS DIRETOS',
        '02.01 - CONTENÇÃO/ESCAVAÇÃO',
        '02.02 - INFRAESTRUTURA',
        '02.03 - SUPERESTRUTURA',
        '02.04 - MARCACAO  ALV. EXT. C/ TALISCA',
        '02.06 - CONTRAPISO',
        '02.07 - ALVENARIA INTERNA',
        '02.08 - CONTRAMARCO',
        '02.09 - REBOCO EXTERNO',
        '02.10 - INST. ELÉTRICAS/HIDROSANITARIAS/GLP',
        '02.13 - REBOCO INTERNO',
        '02.14 - FORRO',
        '02.15 - IMPERMEABILIZACAO',
        '02.16 - REVESTIMENTOS',
        '02.17 - BANCADAS',
        '02.18 - EMASSAMENTO  1 DEMAO',
        '02.19 - REVESTIMENTO EXTERNO',
        '02.20 - ESQUADRIAS DE ALUMINIO',
        '02.21 - ELEVADORES',
        '02.22 - LIMPEZA GROSSA',
        '02.24 - ESQUADRIAS DE MADEIRA/PCF',
        '02.25 - PINTURA FINAL',
        '02.26 - LOUCAS',
        '02.27 - METAIS',
        '02.28 - LIMPEZA FINAL',
        '02.29 - SERVICOS COMPLEMENTARES',
        '02.30 - VISTORIAS/ENTREGA',
        '02.31 - DESPESAS POS-ENTREGA DA OBRA'
    ],
    'Custo Previsto R$': [
        f'R$ {"{:,.2f}".format(28.3559/100*area)}',#CUSTOS INDIRETOS
        
        f'R$ {"{:,.2f}".format(3.5833/100*area)}',#SERVICOS PRELIMINARES E GERAIS
        f'R$ {"{:,.2f}".format(2.1607/100*area)}',#IMPLANTAÇÃO DA OBRA
        f'R$ {"{:,.2f}".format(6.6338/100*area)}',#EQUIPAMENTOS/FERRAMENTAS
        f'R$ {"{:,.2f}".format(15.1586/100*area)}',#CUSTOS ADMINISTRATIVOS
        f'R$ {"{:,.2f}".format(0.5625/100*area)}',#LIMPEZA E DESMOBILIZAÇÃO
        f'R$ {"{:,.2f}".format(0.2570/100*area)}',#OUTRAS DESPESAS DE OBRA
        
        f'R$ {"{:,.2f}".format(71.6441/100*area+tabela1["Esquadrias"].mean()+revestimentos+louças+metais+bancadas+elevadores+instalações+infraestrutura+orcamento_previsto*0.01)}',#CUSTOS DIRETOS
         
        f'R$ {"{:,.2f}".format(2.8433/100*area)}',#CONTENÇÃO/ESCAVAÇÃO
        f'R$ {"{:,.2f}".format(infraestrutura)}',#INFRAESTRUTURA
        f'R$ {"{:,.2f}".format(33.0522/100*area)}',#SUPERESTRUTURA
        f'R$ {"{:,.2f}".format(2.3747/100*area)}',#MARCACAO  ALV. EXT. C/ TALISCA
        f'R$ {"{:,.2f}".format(1.1178/100*area)}',#CONTRAPISO
        f'R$ {"{:,.2f}".format(6.4668/100*area)}',#ALVENARIA INTERNA
        f'R$ {"{:,.2f}".format(0.5824/100*area)}',#CONTRAMARCO
        f'R$ {"{:,.2f}".format(1.7357/100*area)}',#REBOCO EXTERNO
        f'R$ {"{:,.2f}".format(instalações)}',#INST. ELÉTRICAS/HIDROSANITARIAS/GLP
        f'R$ {"{:,.2f}".format(3.4541/100*area)}',#REBOCO INTERNO
        f'R$ {"{:,.2f}".format(3.0525/100*area)}',#FORRO
        f'R$ {"{:,.2f}".format(3.3839/100*area)}',#IMPERMEABILIZACAO
        f'R$ {"{:,.2f}".format(revestimentos)}',#REVESTIMENTOS
        f'R$ {"{:,.2f}".format(bancadas)}',#BANCADAS
        f'R$ {"{:,.2f}".format(1.8849/100*area)}',#EMASSAMENTO  1 DEMAO
        f'R$ {"{:,.2f}".format(0.9586/100*area)}',#REVESTIMENTO EXTERNO
        f'R$ {"{:,.2f}".format(tabela1["Esquadrias"].mean())}',#ESQUADRIAS DE ALUMINIO
        f'R$ {"{:,.2f}".format(elevadores)}',#ELEVADORES
        f'R$ {"{:,.2f}".format(0.4877/100*area)}',#LIMPEZA GROSSA
        f'R$ {"{:,.2f}".format(2.1531/100*area)}',#ESQUADRIAS DE MADEIRA/PCF
        f'R$ {"{:,.2f}".format(2.3842/100*area)}',#PINTURA FINAL
        f'R$ {"{:,.2f}".format(louças)}',#LOUCAS
        f'R$ {"{:,.2f}".format(metais)}',#METAIS
        f'R$ {"{:,.2f}".format(0.8782/100*area)}',#LIMPEZA FINAL
        f'R$ {"{:,.2f}".format(4.6593/100*area)}',#SERVICOS COMPLEMENTARES
        f'R$ {"{:,.2f}".format(0.1747/100*area)}',#VISTORIAS/ENTREGA
        f'R$ {"{:,.2f}".format(1.0/100*orcamento_previsto)}'#DESPESAS POS-ENTREGA DA OBRA
    ],
        'Custo Previsto IC$': [
        f'IC$ {"{:,.2f}".format(28.3559/100*area/ic)}',#CUSTOS INDIRETOS
        
        f'IC$ {"{:,.2f}".format(3.5833/100*area/ic)}',#SERVICOS PRELIMINARES E GERAIS
        f'IC$ {"{:,.2f}".format(2.1607/100*area/ic)}',#IMPLANTAÇÃO DA OBRA
        f'IC$ {"{:,.2f}".format(6.6338/100*area/ic)}',#EQUIPAMENTOS/FERRAMENTAS
        f'IC$ {"{:,.2f}".format(15.1586/100*area/ic)}',#CUSTOS ADMINISTRATIVOS
        f'IC$ {"{:,.2f}".format(0.5625/100*area/ic)}',#LIMPEZA E DESMOBILIZAÇÃO
        f'IC$ {"{:,.2f}".format(0.2570/100*area/ic)}',#OUTRAS DESPESAS DE OBRA
        
        f'IC$ {"{:,.2f}".format((71.6441/100*area+tabela1["Esquadrias"].mean()+revestimentos+louças+metais+bancadas+elevadores+instalações+infraestrutura+orcamento_previsto*0.01)/ic)}',#CUSTOS DIRETOS
         
        f'IC$ {"{:,.2f}".format(2.8433/100*area/ic)}',#CONTENÇÃO/ESCAVAÇÃO
        f'IC$ {"{:,.2f}".format(infraestrutura/ic)}',#INFRAESTRUTURA
        f'IC$ {"{:,.2f}".format(33.0522/100*area/ic)}',#SUPERESTRUTURA
        f'IC$ {"{:,.2f}".format(2.3747/100*area/ic)}',#MARCACAO  ALV. EXT. C/ TALISCA
        f'IC$ {"{:,.2f}".format(1.1178/100*area/ic)}',#CONTRAPISO
        f'IC$ {"{:,.2f}".format(6.4668/100*area/ic)}',#ALVENARIA INTERNA
        f'IC$ {"{:,.2f}".format(0.5824/100*area/ic)}',#CONTRAMARCO
        f'IC$ {"{:,.2f}".format(1.7357/100*area/ic)}',#REBOCO EXTERNO
        f'IC$ {"{:,.2f}".format(instalações/ic)}',#INST. ELÉTRICAS/HIDROSANITARIAS/GLP
        f'IC$ {"{:,.2f}".format(3.4541/100*area/ic)}',#REBOCO INTERNO
        f'IC$ {"{:,.2f}".format(3.0525/100*area/ic)}',#FORRO
        f'IC$ {"{:,.2f}".format(3.3839/100*area/ic)}',#IMPERMEABILIZACAO
        f'IC$ {"{:,.2f}".format(revestimentos/ic)}',#REVESTIMENTOS
        f'IC$ {"{:,.2f}".format(bancadas/ic)}',#BANCADAS
        f'IC$ {"{:,.2f}".format(1.8849/100*area/ic)}',#EMASSAMENTO  1 DEMAO
        f'IC$ {"{:,.2f}".format(0.9586/100*area/ic)}',#REVESTIMENTO EXTERNO
        f'IC$ {"{:,.2f}".format(tabela1["Esquadrias"].mean()/ic)}',#ESQUADRIAS DE ALUMINIO
        f'IC$ {"{:,.2f}".format(elevadores/ic)}',#ELEVADORES
        f'IC$ {"{:,.2f}".format(0.4877/100*area/ic)}',#LIMPEZA GROSSA
        f'IC$ {"{:,.2f}".format(2.1531/100*area/ic)}',#ESQUADRIAS DE MADEIRA/PCF
        f'IC$ {"{:,.2f}".format(2.3842/100*area/ic)}',#PINTURA FINAL
        f'IC$ {"{:,.2f}".format(louças/ic)}',#LOUCAS
        f'IC$ {"{:,.2f}".format(metais/ic)}',#METAIS
        f'IC$ {"{:,.2f}".format(0.8782/100*area/ic)}',#LIMPEZA FINAL
        f'IC$ {"{:,.2f}".format(4.6593/100*area/ic)}',#SERVICOS COMPLEMENTARES
        f'IC$ {"{:,.2f}".format(0.1747/100*area/ic)}',#VISTORIAS/ENTREGA
        f'IC$ {"{:,.2f}".format(1.0/100*orcamento_previsto/ic)}'#DESPESAS POS-ENTREGA DA OBRA
        ]
}

# Criar o DataFrame
tabela = pd.DataFrame(dados)

# Criar uma função para aplicar a formatação
def formata_linha(linha):
    if linha.name in [0, 7]:  # Verificar se o índice da linha é 0 ou 7
        return ['background-color: lightblue; font-weight: bold' for _ in linha]  # Aplicar negrito e cor de fundo azul claro
    return ['' for _ in linha]  # Não aplicar formatação às outras linhas

# Aplicar a formatação ao DataFrame
tabela_estilizada = tabela.style.apply(formata_linha, axis=1)
# Exibindo a tabela
st.dataframe(tabela_estilizada, width=10000, height=1262)



