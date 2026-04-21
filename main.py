import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_data(filepath, chunksize=10000):
    cols = ['Medium', 'Object Begin Date', 'Object End Date']
    for chunk in pd.read_csv(filepath, chunksize=chunksize, usecols=cols):
        yield chunk

def filter_and_calculate(chunks):
    for df in chunks:
        df = df.dropna(subset=['Medium', 'Object Begin Date', 'Object End Date']).copy()
        df['Object Begin Date'] = pd.to_numeric(df['Object Begin Date'])
        df['Object End Date'] = pd.to_numeric(df['Object End Date'])
        df = df.dropna()
        
        df['Duration'] = df['Object End Date'] - df['Object Begin Date']
        df['Duration_sq'] = df['Duration'] ** 2
        yield df

def aggregate_chunk(chunks):
    for df in chunks:
        agg = df.groupby(['Medium', 'Object Begin Date']).agg(
            count=('Duration', 'count'),
            sum_dur=('Duration', 'sum'),
            sum_dur_sq=('Duration_sq', 'sum')
        ).reset_index()
        yield agg

def global_reduce(pipeline):
    return pd.concat(pipeline, ignore_index=True).groupby(['Medium', 'Object Begin Date'], as_index=False).sum()

def main():
    filepath = 'MetObjects.csv'

    pipeline = aggregate_chunk(filter_and_calculate(read_data(filepath)))
    
    final_data = global_reduce(pipeline)

    medium_stats = final_data.groupby('Medium').sum()
    top_10 = medium_stats.nlargest(10, 'count').copy()

    n = top_10['count']
    sum_x = top_10['sum_dur']
    sum_x2 = top_10['sum_dur_sq']

    top_10['mean'] = sum_x / n
    variance = (sum_x2 - (sum_x ** 2) / n) / (n - 1).clip(lower=1)
    std = np.sqrt(variance.clip(lower=0))

    top_10['ci_95'] = 1.96 * (std / np.sqrt(n))
    top_10['pi_95'] = 1.96 * std


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3)

    x_pos = np.arange(len(top_10))
    ax1.bar(x_pos, top_10['mean'], yerr=top_10['ci_95'], capsize=5, 
            color='skyblue', alpha=0.8, label='Среднее (95% Доверительный интервал)')
    ax1.errorbar(x_pos, top_10['mean'], yerr=top_10['pi_95'], fmt='none', 
                 ecolor='red', alpha=0.4, label='95% Интервал рассеивания')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(top_10.index, rotation=45, ha='right')
    ax1.set_title('Top-10 материалов: Среднее время создания', fontweight='bold')
    ax1.legend()

    leader = top_10['mean'].idxmax()
    timeline = final_data[final_data['Medium'] == leader].sort_values('Object Begin Date')
    y_yearly = timeline['sum_dur'] / timeline['count']
    y_rolling = y_yearly.rolling(window=15, min_periods=1).mean()

    ax2.scatter(timeline['Object Begin Date'], y_yearly, alpha=0.3, s=15, color='gray', label='Среднее по году')
    ax2.plot(timeline['Object Begin Date'], y_rolling, color='firebrick', linewidth=2.5, label='Скользящее среднее')

    ax2.set_title(f'Динамика времени создания: {leader}', fontweight='bold')
    ax2.set_xlabel('Год начала')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.show()

if __name__ == '__main__':
    main()