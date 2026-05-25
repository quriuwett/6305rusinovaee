import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger("metetl")

def run_analysis(csv_path: str, output_dir: str):
    logger.info("Запуск анализа данных из CSV...")
    os.makedirs(output_dir, exist_ok=True)
    
    cols = ['Medium', 'Object Begin Date', 'Object End Date']
    
    def process_data():
        chunks = pd.read_csv(csv_path, chunksize=10000, usecols=cols)
        for df in chunks:
            df = df.dropna().copy()
            df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'], errors='coerce')
            df['Object End Date'] = pd.to_numeric(df['Object End Date'], errors='coerce')
            df = df.dropna()
            df['Duration'] = df['Object End Date'] - df['Object Begin Date']
            df['Duration_sq'] = df['Duration'] ** 2
            yield df.groupby(['Medium', 'Object Begin Date']).agg(
                count=('Duration', 'count'),
                sum_dur=('Duration', 'sum'),
                sum_dur_sq=('Duration_sq', 'sum')
            ).reset_index()

    final_data = pd.concat(process_data()).groupby(['Medium', 'Object Begin Date']).sum().reset_index()
    
    m_stats = final_data.groupby('Medium').sum()
    top_10 = m_stats.nlargest(10, 'count').copy()
    
    n = top_10['count']
    sum_x = top_10['sum_dur']
    sum_x2 = top_10['sum_dur_sq']
    top_10['mean'] = sum_x / n
    std = np.sqrt(((sum_x2 - (sum_x**2)/n) / (n-1)).clip(lower=0))
    top_10['ci_95'] = 1.96 * (std / np.sqrt(n))
    top_10['pi_95'] = 1.96 * std

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    x_pos = np.arange(len(top_10))
    ax1.bar(x_pos, top_10['mean'], yerr=top_10['ci_95'], capsize=5, color='skyblue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(top_10.index, rotation=45, ha='right')
    ax1.set_title('Top-10 материалов')

    leader = top_10['mean'].idxmax()
    tl = final_data[final_data['Medium'] == leader].sort_values('Object Begin Date')
    ax2.plot(tl['Object Begin Date'], (tl['sum_dur']/tl['count']).rolling(15).mean(), color='firebrick')
    ax2.set_title(f'Динамика: {leader}')

    plot_path = os.path.join(output_dir, 'analysis_results.png')
    plt.savefig(plot_path)
    plt.close(fig)
    logger.info(f"Анализ завершен. График сохранен в {plot_path}")