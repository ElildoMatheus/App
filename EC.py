import streamlit as st
import pandas as pd
st.set_page_config(page_title="ESTIMATIVA DE CUSTO")

st.markdown(
    """
    <style>
        .stApp {
            margin-left: auto;
            margin-right: auto;
        }
        .stSideBar {
            transform: translate(-50%);
        }
    </style>
    """,
    unsafe_allow_html=True
)



st.title("ESTIMATIVA DE CUSTO")
st.sidebar.title("Selecione os parâmetros do projeto")

bar = st.sidebar
bancadas = bar.selectbox("Padrão de bancadas", ["Alto", "Médio", "Baixo"])
revestimentos = bar.selectbox("Padrão de revestimentos", ["Alto", "Médio", "Baixo"])
loucas = bar.selectbox("Padrão de louças", ["Alto", "Médio", "Baixo"])
metais = bar.selectbox("Padrão de metais", ["Alto", "Médio", "Baixo"])
esquadrias = bar.selectbox("Padrão de esquadrias", ["Alto", "Médio", "Baixo"])
fundação = bar.selectbox("Tipo de fundação",["Sapata", "Estacão","Hélice"])
Elevadores = st.sidebar.number_input("Quantidade de elevadores")
uni = st.sidebar.number_input("Total de unidades privativas")
indice_aço = st.sidebar.number_input("Indice de aço (Kg/m²)")
indice_concreto = st.sidebar.number_input("Indice de concreto (m³/m²)")
indice_forma = st.sidebar.number_input("Indice de forme (m²/m²)")
ic = st.sidebar.number_input("IC Atual")
prazo = st.sidebar.number_input("Prazo de execução")



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
    ]
}

# Criando um DataFrame do Pandas com os dados
tabela = pd.DataFrame(dados)
# Criar uma função para aplicar a formatação
def formata_linha(linha):
    if linha.name in [0, 7]:  # Verificar se o índice da linha é 0 ou 7
        return ['background-color: lightblue; font-weight: bold' for _ in linha]  # Aplicar negrito e cor de fundo azul claro
    return ['' for _ in linha]  # Não aplicar formatação às outras linhas

# Aplicar a formatação ao DataFrame
tabela_estilizada = tabela.style.apply(formata_linha, axis=1)

# Exibindo a tabela
st.dataframe(tabela_estilizada, width=800)



