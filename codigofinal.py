import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

def load_data(file_path):
    """
    Carrega o dataset a partir de um arquivo CSV.
    
    Args:
        file_path (str): Caminho para o arquivo CSV.
        
    Returns:
        pd.DataFrame: DataFrame contendo os dados carregados.
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        return df
    except FileNotFoundError:
        print(f"Erro: Arquivo {file_path} não encontrado.")
        return None

def clean_data(df):
    """
    Limpa os dados removendo valores nulos nas colunas relevantes.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados originais.
        
    Returns:
        pd.DataFrame: DataFrame contendo os dados limpos.
    """
    return df.dropna(subset=['Qty', 'Amount', 'Category'])

def plot_sales_by_category(df):
    """
    Plota a receita total por categoria.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados limpos.
    """
    sales_by_category = df.groupby('Category')['Amount'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sales_by_category, x='Category', y='Amount', palette='viridis')
    plt.title("Receita por Categoria")
    plt.xticks(rotation=45)
    plt.xlabel("Categoria")
    plt.ylabel("Receita Total")
    plt.savefig('grafico_receita_por_categoria.png')
    plt.show()

def plot_correlation(df):
    """
    Plota a correlação entre quantidade vendida e valor total.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados limpos.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Qty', y='Amount', alpha=0.7)
    plt.title("Correlação entre Quantidade Vendida e Valor Total")
    plt.xlabel("Quantidade Vendida")
    plt.ylabel("Valor Total")
    plt.savefig('grafico_correlacao.png')
    plt.show()

def plot_monthly_sales(df):
    """
    Plota as vendas mensais.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados limpos.
    """
    df['Date'] = pd.to_datetime(df['Date'])
    monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(kind='line')
    plt.title("Vendas Mensais")
    plt.xlabel("Mês")
    plt.ylabel("Receita Total")
    plt.savefig('grafico_vendas_mensais.png')
    plt.show()

def calculate_rfm(df):
    """
    Calcula Recência, Frequência e Valor Monetário (RFM).
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados limpos.
        
    Returns:
        pd.DataFrame: DataFrame contendo os valores RFM.
    """
    reference_date = dt.datetime(2025, 3, 7)
    df['Recency'] = (reference_date - pd.to_datetime(df['Date'])).dt.days
    rfm = df.groupby('Order ID').agg({
        'Recency': 'min',
        'Order ID': 'count',
        'Amount': 'sum'
    }).rename(columns={'Order ID': 'Frequency', 'Amount': 'Monetary'})
    return rfm

def calculate_clv(df):
    """
    Calcula o valor de vida do cliente (CLV).
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados limpos.
        
    Returns:
        pd.DataFrame: DataFrame contendo os valores CLV.
    """
    clv = df.groupby('Order ID')['Amount'].sum().reset_index().rename(columns={'Amount': 'CLV'})
    return clv

def identify_top_products(df):
    """
    Identifica os produtos mais vendidos.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados limpos.
        
    Returns:
        pd.DataFrame: DataFrame contendo os produtos mais vendidos.
    """
    top_products = df.groupby('Style')['Qty'].sum().reset_index().sort_values(by='Qty', ascending=False).head(10)
    return top_products

def main():
    # Carregar e limpar os dados
    df = load_data('amazon.csv')
    if df is not None:
        df = clean_data(df)
        
        # Visualizar os primeiros dados
        print("Primeiras linhas do dataset:")
        print(df.head())
        
        # Informações gerais do dataset
        print("\nInformações do dataset:")
        print(df.info())
        
        # Estatísticas descritivas
        print("\nEstatísticas descritivas:")
        print(df.describe())
        
        # Plotar gráficos
        plot_sales_by_category(df)
        plot_correlation(df)
        plot_monthly_sales(df)
        
        # Calcular RFM
        rfm = calculate_rfm(df)
        print("\nRFM:")
        print(rfm.head())
        
        # Calcular CLV
        clv = calculate_clv(df)
        print("\nCLV:")
        print(clv.head())
        
        # Identificar os produtos mais vendidos
        top_products = identify_top_products(df)
        print("\nProdutos mais vendidos:")
        print(top_products)

if __name__ == "__main__":
    main()