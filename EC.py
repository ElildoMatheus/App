from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import streamlit as st

# Define as configurações da página primeiro
st.set_page_config(
    page_title="Estimativa de Custo",
    layout="wide"
)

# lendo a base de dados
baseR01 = pd.read_excel("BD1.xlsx")
baseR01 = baseR01.drop(['Obra'], axis=1)


# Escrever o título "Estimativa de Custo" e "AI" em tamanho de fonte 1.5x
st.write("<h1 style='font-size: 3em;'>Estimativa de Custo <span style='font-size: 0.5em; color: red;'>AI</span></h1>", unsafe_allow_html=True)



st.sidebar.subheader("Selecione os parâmetros do projeto")

estilo_caixa = """
    background-color: #111;
    color: #fff;
    padding: 1px;
    border-radius: 10px;
    margin-bottom: 20px;
"""
# Exibe a caixa estilizada no texto principal
st.markdown(f'<div style="{estilo_caixa}"></div>', unsafe_allow_html=True)

estilo_caixa = """
    background-color: #111;
    color: #fff;
    padding: 1px;
    border-radius: 10px;
    margin-bottom: 20px;
"""

# Exibe a caixa estilizada na barra lateral
st.sidebar.markdown(f'<div style="{estilo_caixa}"></div>', unsafe_allow_html=True)


# Caixa de texto para o nome do projeto
nome_do_projeto = st.sidebar.text_input("Nome do Projeto")

estilo_retangulo = """
    background-color: #D3D3D3;
    color: black;
    padding: 5px;
    border-radius: 5px;
    margin-bottom: 10px;
"""

# Exibe as informações dentro dos retângulos estilizados
st.markdown(f'<div style="{estilo_retangulo}">Projeto: {nome_do_projeto}</div>', unsafe_allow_html=True)

# Criar uma linha extensa usando st.markdown()
st.markdown("")

# Estimativa Inicial
with st.expander("Estimativa Inicial", expanded=False):
    # Seção Gerais
    with st.sidebar.expander("**1. Gerais**", expanded=False):
        ic = st.number_input("IC Atual")
        prazo = st.number_input("Prazo de execução", help="Meses")

    # Seção 1: Área
    with st.sidebar.expander("**2. Área**", expanded=False):
        # Define o layout de duas colunas dentro do expander
        col1, col2 = st.columns(2)

        # Adiciona os widgets em cada coluna
        with col1:
            subsolos = st.number_input("Qtde. Subsolo")
            terreo = st.number_input("Qtde. Terreo")
            garagem = st.number_input("Qtde. Garagem")
            lazer = st.number_input("Qtde. Lazer")
            tipo = st.number_input("Qtde. Tipo")
            penthouses = st.number_input("Qtde. Penthouse")
            rooftop = st.number_input("Qtde. Rooftop")
        with col2:
            areasubsolo = st.number_input("Área Subsolos", help="Área média dos pavimentos de Subsolos (m²)")
            areaterreo = st.number_input("Área Terreo", help="Área média dos pavimentos de Térreo (m²)")
            areagaragem = st.number_input("Área Garagem", help="Área média dos pavimentos de Garagens (m²)")
            arealazer = st.number_input("Área Lazer", help="Área média dos pavimentos de Lazers (m²)")
            areatipo = st.number_input("Área Tipo", help="Área média dos pavimentos de Tipo (m²)")
            areapenthouses = st.number_input("Área Penthouse", help="Área média dos pavimentos de Penthouses (m²)")
            arearooftop = st.number_input("Área Rooftop", help="Área média dos pavimentos de Rooftop (m²)")
        uni = st.number_input("Total de unidades privativas")

    # Seção 2: Fundação
    with st.sidebar.expander("**3. Fundação**", expanded=False):
        # Widgets para a segunda seção
        valores_fundacao = {"Hélice": 3, "Estacão": 2, "Sapata": 1}
        valor_fundacao = st.radio("Tipo de fundação", list(valores_fundacao.keys()))
        fundação = valores_fundacao[valor_fundacao]
        perimetroterreno= st.number_input("Perímetro do terreno", help="m")

    # Seção 4: Estrutura
    with st.sidebar.expander("**4. Estrutura**", expanded=False):
        # Widgets para a terceira seção
        indice_aço = st.number_input("Índice de aço (Kg/m²)")
        indice_concreto = st.number_input("Índice de concreto (m³/m²)")
        indice_forma = st.number_input("Índice de forma (m²/m²)")

    # Seção 4: Elevadores
    with st.sidebar.expander("**5. Elevadores**", expanded=False):
        # Widget para a quarta seção
        elevadores = st.number_input("Quantidade de elevadores", value=0.0)

    # Seção 5: Acabamento
    with st.sidebar.expander("**6. Acabamento**", expanded=False):
        # Widgets para a quinta seção
        valores_acabamento = {"Alto": 3, "Médio": 2, "Baixo": 1}

        valor_bancadas = st.radio("Padrão de bancadas", ["Alto", "Médio", "Baixo"])
        bancadas = valores_acabamento[valor_bancadas]

        valor_revestimentos = st.radio("Padrão de revestimentos", ["Alto", "Médio", "Baixo"])
        revestimentos = valores_acabamento[valor_revestimentos]

        valor_loucas = st.radio("Padrão de louças", ["Alto", "Médio", "Baixo"])
        loucas = valores_acabamento[valor_loucas]

        valor_metais = st.radio("Padrão de metais", ["Alto", "Médio", "Baixo"])
        metais = valores_acabamento[valor_metais]

        valor_esquadrias = st.radio("Padrão de esquadrias", ["Alto", "Médio", "Baixo"])
        esq= valores_acabamento[valor_esquadrias]



# Texto no final da barra lateral com estilo de fonte personalizado
st.sidebar.markdown("<p style='text-align: center; font-family: Courier New, monospace; font-size: 12px;'>© 2024 Planejamento e Controle de Obras</p>", unsafe_allow_html=True)



# Condicional para atribuir valores com base na quantidade de subsolos
if subsolos == 3:
    valor = 0.9538
elif subsolos == 2:
    valor = 0.8183
elif subsolos == 1:
    valor = 0.7124
else:
    valor = 0.8183

areaequivalente = (subsolos*areasubsolo*valor + terreo*areaterreo*0.7592 +
             garagem*areagaragem*0.7797 + lazer*arealazer*0.9054 + tipo*areatipo*1 + penthouses*areapenthouses*0.9351 +
             rooftop*arearooftop*1.0301 + 140*0.7)

areaconstrutiva = (subsolos*areasubsolo + terreo*areaterreo + garagem*areagaragem + lazer*arealazer + tipo*areatipo*1 + penthouses*areapenthouses +
             rooftop*arearooftop + 140.00)


areaprivativa = tipo*areatipo+ penthouses*areapenthouses + rooftop*arearooftop

if uni == 0:
    uni = 1

areaprivativauni = (areaprivativa)/uni

aço = indice_aço*areaconstrutiva
concreto = indice_concreto*areaconstrutiva
forma = indice_forma*areaconstrutiva

# AREA

# Avaliação
x_train = baseR01[['ÁREA EQUIVALENTE']]
y_train = baseR01['ORÇAMENTO1']

# Inicializar os modelos
linear_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()
rf_model = RandomForestRegressor(random_state=42)  # Definindo a semente aleatória como 42
xgb_model = XGBRegressor(random_state=42)  # Definindo a semente aleatória como 42
dt_model = DecisionTreeRegressor(random_state=42)  # Definindo a semente aleatória como 42
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
area = float(tabela1['Área'].mean() + tabela1['Estrutura'].mean())/2
elevadores = float(tabela1['Elevadores'].mean())
infraestrutura = float((pred10_linear + pred10_lasso + pred10_ridge)/3)
custosdeprazo = float((pred11_linear + pred11_lasso + pred11_ridge)/3)
esquadrias = float((pred7_linear*tipo + pred7_lasso*tipo + pred7_ridge*tipo)/3)
orcamento = custosdeprazo + revestimentos + instalações + louças + metais + bancadas + area + elevadores + infraestrutura + esquadrias
orcamento_previsto = orcamento*1.01


# Calcular o valor de orçamento_previsto/ic dividido por areaequivalente
orcamento_ic_por_areaequivalente = orcamento_previsto / ic / areaequivalente

# Formatar os três valores em caixas separadas com tamanho aumentado e cor de fundo cinza claro, com uma barra fina vertical à esquerda de cada caixa
html = f'<div style="display: flex;">' \
       f'<div style="flex: 1; border-left: 4px solid #888888; background-color: #f2f2f2; padding: 20px; border-radius: 5px; margin-right: 10px; height: 100px;">' \
           f'<p style="font-size: 0.9em; font-weight: bold; margin-bottom: 5px;">CUSTO PREVISTO:</p>' \
           f'<p style="font-size: 1.5em; font-weight: bold;">R$ {"{:,.2f}".format(orcamento_previsto)}</p>' \
       f'</div>' \
       f'<div style="flex: 1; border-left: 4px solid #888888; background-color: #f2f2f2; padding: 20px; border-radius: 5px; margin-right: 10px; height: 100px;">' \
           f'<p style="font-size: 0.9em; font-weight: bold; margin-bottom: 5px;">CUSTO PREVISTO:</p>' \
           f'<p style="font-size: 1.5em; font-weight: bold;">IC$ {"{:,.2f}".format(orcamento_previsto/ic)}</p>' \
       f'</div>' \
       f'<div style="flex: 1; border-left: 4px solid #888888; background-color: #f2f2f2; padding: 20px; border-radius: 5px; height: 100px;">' \
           f'<p style="font-size: 0.9em; font-weight: bold; margin-bottom: 5px;">IC$/AE:</p>' \
           f'<p style="font-size: 1.5em; font-weight: bold;">{"{:,.2f}".format(orcamento_ic_por_areaequivalente)}</p>' \
       f'</div>' \
       f'</div>'

# Renderizar as caixas na interface do Streamlit usando st.markdown()
st.markdown(html, unsafe_allow_html=True)


# Criar uma linha extensa usando st.markdown()
st.markdown("")


Orçamento_AE = orcamento_previsto/areaequivalente



dados = {
    'DESCRIÇÃO': [
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
    'CUSTO PREVISTO R$': [
        f'R$ {"{:,.2f}".format(custosdeprazo)}',#CUSTOS INDIRETOS
        
        f'R$ {"{:,.2f}".format(12.7076/100*custosdeprazo)}',#SERVICOS PRELIMINARES E GERAIS
        f'R$ {"{:,.2f}".format(7.6307/100*custosdeprazo)}',#IMPLANTAÇÃO DA OBRA
        f'R$ {"{:,.2f}".format(23.4389/100*custosdeprazo)}',#EQUIPAMENTOS/FERRAMENTAS
        f'R$ {"{:,.2f}".format(53.3418/100*custosdeprazo)}',#CUSTOS ADMINISTRATIVOS
        f'R$ {"{:,.2f}".format(1.9807/100*custosdeprazo)}',#LIMPEZA E DESMOBILIZAÇÃO
        f'R$ {"{:,.2f}".format(0.9003/100*custosdeprazo)}',#OUTRAS DESPESAS DE OBRA
        
        f'R$ {"{:,.2f}".format(area+esquadrias+revestimentos+louças+metais+bancadas+elevadores+instalações+infraestrutura+orcamento_previsto*0.01)}',#CUSTOS DIRETOS
         
        f'R$ {"{:,.2f}".format(26.0307/100*infraestrutura)}',#CONTENÇÃO/ESCAVAÇÃO
        f'R$ {"{:,.2f}".format(73.9693/100*infraestrutura)}',#INFRAESTRUTURA
        f'R$ {"{:,.2f}".format(48.1251/100*area)}',#SUPERESTRUTURA
        f'R$ {"{:,.2f}".format(3.4514/100*area)}',#MARCACAO  ALV. EXT. C/ TALISCA
        f'R$ {"{:,.2f}".format(1.6258/100*area)}',#CONTRAPISO
        f'R$ {"{:,.2f}".format(9.3271/100*area)}',#ALVENARIA INTERNA
        f'R$ {"{:,.2f}".format(0.8445/100*area)}',#CONTRAMARCO
        f'R$ {"{:,.2f}".format(2.5130/100*area)}',#REBOCO EXTERNO
        f'R$ {"{:,.2f}".format(instalações)}',#INST. ELÉTRICAS/HIDROSANITARIAS/GLP
        f'R$ {"{:,.2f}".format(5.0129/100*area)}',#REBOCO INTERNO
        f'R$ {"{:,.2f}".format(4.4077/100*area)}',#FORRO
        f'R$ {"{:,.2f}".format(4.9245/100*area)}',#IMPERMEABILIZACAO
        f'R$ {"{:,.2f}".format(revestimentos)}',#REVESTIMENTOS
        f'R$ {"{:,.2f}".format(bancadas)}',#BANCADAS
        f'R$ {"{:,.2f}".format(2.7311/100*area)}',#EMASSAMENTO  1 DEMAO
        f'R$ {"{:,.2f}".format(1.4011/100*area)}',#REVESTIMENTO EXTERNO
        f'R$ {"{:,.2f}".format(esquadrias)}',#ESQUADRIAS DE ALUMINIO
        f'R$ {"{:,.2f}".format(elevadores)}',#ELEVADORES
        f'R$ {"{:,.2f}".format(0.7087/100*area)}',#LIMPEZA GROSSA
        f'R$ {"{:,.2f}".format(3.1158/100*area)}',#ESQUADRIAS DE MADEIRA/PCF
        f'R$ {"{:,.2f}".format(3.4601/100*area)}',#PINTURA FINAL
        f'R$ {"{:,.2f}".format(louças)}',#LOUCAS
        f'R$ {"{:,.2f}".format(metais)}',#METAIS
        f'R$ {"{:,.2f}".format(1.2862/100*area)}',#LIMPEZA FINAL
        f'R$ {"{:,.2f}".format(6.8106/100*area)}',#SERVICOS COMPLEMENTARES
        f'R$ {"{:,.2f}".format(0.2545/100*area)}',#VISTORIAS/ENTREGA
        f'R$ {"{:,.2f}".format(1.0/100*orcamento_previsto)}'#DESPESAS POS-ENTREGA DA OBRA
    ],
        'CUSTO PREVISTO IC$': [
    f'R$ {"{:,.2f}".format(custosdeprazo/ic)}',#CUSTOS INDIRETOS
        
        f'R$ {"{:,.2f}".format(12.7076/100*custosdeprazo/ic)}',#SERVICOS PRELIMINARES E GERAIS
        f'R$ {"{:,.2f}".format(7.6307/100*custosdeprazo/ic)}',#IMPLANTAÇÃO DA OBRA
        f'R$ {"{:,.2f}".format(23.4389/100*custosdeprazo/ic)}',#EQUIPAMENTOS/FERRAMENTAS
        f'R$ {"{:,.2f}".format(53.3418/100*custosdeprazo/ic)}',#CUSTOS ADMINISTRATIVOS
        f'R$ {"{:,.2f}".format(1.9807/100*custosdeprazo/ic)}',#LIMPEZA E DESMOBILIZAÇÃO
        f'R$ {"{:,.2f}".format(0.9003/100*custosdeprazo/ic)}',#OUTRAS DESPESAS DE OBRA
        
        f'R$ {"{:,.2f}".format((area+esquadrias+revestimentos+louças+metais+bancadas+elevadores+instalações+infraestrutura+orcamento_previsto*0.01)/ic)}',#CUSTOS DIRETOS
         
        f'R$ {"{:,.2f}".format(26.0307/100*infraestrutura/ic)}',#CONTENÇÃO/ESCAVAÇÃO
        f'R$ {"{:,.2f}".format(73.9693/100*infraestrutura/ic)}',#INFRAESTRUTURA
        f'R$ {"{:,.2f}".format(48.1251/100*area/ic)}',#SUPERESTRUTURA
        f'R$ {"{:,.2f}".format(3.4514/100*area/ic)}',#MARCACAO  ALV. EXT. C/ TALISCA
        f'R$ {"{:,.2f}".format(1.6258/100*area/ic)}',#CONTRAPISO
        f'R$ {"{:,.2f}".format(9.3271/100*area/ic)}',#ALVENARIA INTERNA
        f'R$ {"{:,.2f}".format(0.8445/100*area/ic)}',#CONTRAMARCO
        f'R$ {"{:,.2f}".format(2.5130/100*area/ic)}',#REBOCO EXTERNO
        f'R$ {"{:,.2f}".format(instalações/ic)}',#INST. ELÉTRICAS/HIDROSANITARIAS/GLP
        f'R$ {"{:,.2f}".format(5.0129/100*area/ic)}',#REBOCO INTERNO
        f'R$ {"{:,.2f}".format(4.4077/100*area/ic)}',#FORRO
        f'R$ {"{:,.2f}".format(4.9245/100*area/ic)}',#IMPERMEABILIZACAO
        f'R$ {"{:,.2f}".format(revestimentos/ic)}',#REVESTIMENTOS
        f'R$ {"{:,.2f}".format(bancadas/ic)}',#BANCADAS
        f'R$ {"{:,.2f}".format(2.7311/100*area/ic)}',#EMASSAMENTO  1 DEMAO
        f'R$ {"{:,.2f}".format(1.4011/100*area/ic)}',#REVESTIMENTO EXTERNO
        f'R$ {"{:,.2f}".format(esquadrias/ic)}',#ESQUADRIAS DE ALUMINIO
        f'R$ {"{:,.2f}".format(elevadores/ic)}',#ELEVADORES
        f'R$ {"{:,.2f}".format(0.7087/100*area/ic)}',#LIMPEZA GROSSA
        f'R$ {"{:,.2f}".format(3.1158/100*area/ic)}',#ESQUADRIAS DE MADEIRA/PCF
        f'R$ {"{:,.2f}".format(3.4601/100*area/ic)}',#PINTURA FINAL
        f'R$ {"{:,.2f}".format(louças/ic)}',#LOUCAS
        f'R$ {"{:,.2f}".format(metais/ic)}',#METAIS
        f'R$ {"{:,.2f}".format(1.2862/100*area/ic)}',#LIMPEZA FINAL
        f'R$ {"{:,.2f}".format(6.8106/100*area/ic)}',#SERVICOS COMPLEMENTARES
        f'R$ {"{:,.2f}".format(0.2545/100*area/ic)}',#VISTORIAS/ENTREGA
        f'R$ {"{:,.2f}".format(1.0/100*orcamento_previsto/ic)}'#DESPESAS POS-ENTREGA DA OBRA
        ]
}

# Criar o DataFrame
tabela = pd.DataFrame(dados)

# Convertendo os valores da coluna 'CUSTO PREVISTO IC$' para float
tabela['CUSTO PREVISTO IC$'] = tabela['CUSTO PREVISTO IC$'].str.replace('R$', '').str.replace(',', '').astype(float)

# Calculando o custo previsto por unidade de área equivalente
tabela['IC$/AE'] = tabela['CUSTO PREVISTO IC$'] / areaequivalente

# Calculando o custo por área equivalente dividido por Orçamento_AE
tabela['%'] = tabela['IC$/AE'] / Orçamento_AE
# Formatando a coluna de porcentagem '%' diretamente
tabela['%'] = tabela['%'].map(lambda x: "{:.2%}".format(x))

# Adicionando o formato desejado às colunas 'CUSTO PREVISTO IC$' e 'IC$/AE'
tabela['CUSTO PREVISTO IC$'] = tabela['CUSTO PREVISTO IC$'].map(lambda x: f'IC$ {"{:,.2f}".format(x)}')
tabela['IC$/AE'] = tabela['IC$/AE'].map(lambda x: f'IC$ {"{:,.2f}".format(x)}')
# Adicionando o texto "/AE" ao final de cada número na coluna 'IC$/AE'
tabela['IC$/AE'] = tabela['IC$/AE'].map(lambda x: f'{x} /AE')

# Criar uma função para aplicar a formatação
def formata_linha(linha):
    if linha.name == 0 or linha.name == 7:  # Verificar se o índice da linha é 0 ou 7
        return ['color: #000000; font-weight: bold; background-color: #D3D3D3' for _ in linha]  # Aplicar negrito, cor de texto preto e cor de fundo cinza claro
    return ['color: #000000; font-weight: normal' for _ in linha]  # Aplicar cor preta aos títulos das colunas

# Aplicar a formatação ao DataFrame
tabela_estilizada = tabela.style.apply(formata_linha, axis=1).set_table_styles([{
    'selector': 'thead',  # Seletor para o cabeçalho da tabela
    'props': [('background-color', '#404040')]  # Estilo para o fundo do cabeçalho (fundo cinza escuro)
}])


# Exibindo a tabela
st.dataframe(tabela_estilizada, width=8000, height=1262)
