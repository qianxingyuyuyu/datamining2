import dask.dataframe as dd
import pandas as pd
import json
from mlxtend.frequent_patterns import apriori, association_rules,fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import networkx as nx

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.width', None)       # 自动调整宽度（避免换行）
pd.set_option('display.max_colwidth', None)  # 显示完整列内容（不截断文本）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
ddf = dd.read_parquet('processed_data.parquet/')
print(ddf.head(5))
df = ddf.compute()


# 任务1.1: 支付方式与商品类别的关联规则
def payment_category_association(df):
    # 重置索引确保order_id是列
    df_reset = df.reset_index()

    # 生成(商品类别, 支付方式)的二元组
    transactions = []
    for _, row in df_reset.iterrows():
        payment = row['payment_method']
        categories = set()  # 使用set避免重复类别

        # 假设item_category可能是单个值或列表
        if isinstance(row['item_category'], list):
            categories.update(row['item_category'])
        else:
            categories.add(row['item_category'])

        # 为每个商品类别创建单独的交易记录
        for category in categories:
            transactions.append([category, payment])

    print("交易数据示例:", transactions[:5])  # 打印前5个交易数据

    # 转换为布尔矩阵
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘频繁项集
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    print("频繁项集示例:", frequent_itemsets.head(5))  # 打印前5个频繁项集
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    # 筛选支付方式→商品类别的规则
    payment_to_category = rules[
        rules['consequents'].apply(
            lambda x: any(p in x for p in ['微信支付', '信用卡', '现金', '云闪付', '储蓄卡'])) &
        rules['antecedents'].apply(
            lambda x: not any(p in x for p in ['微信支付', '信用卡', '现金', '云闪付', '储蓄卡']))
        ]

    # 筛选商品类别→支付方式的规则
    category_to_payment = rules[
        rules['antecedents'].apply(
            lambda x: any(p in x for p in ['微信支付', '信用卡', '现金', '云闪付', '储蓄卡'])) &
        rules['consequents'].apply(
            lambda x: not any(p in x for p in ['微信支付', '信用卡', '现金', '云闪付', '储蓄卡']))
        ]

    print("\n支付方式→商品类别的规则:")
    print(payment_to_category[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    print("\n商品类别→支付方式的规则:")
    print(category_to_payment[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

    return {
        'payment_to_category': payment_to_category,
        'category_to_payment': category_to_payment
    }


# 任务1.2: 高价值商品的支付方式分析
def high_value_payment(df):
    high_value = df[df['price'] > 5000]
    payment_dist = high_value['payment_method'].value_counts(normalize=True)
    return payment_dist


# 执行分析
payment_category_rules = payment_category_association(df)
high_value_payment_dist = high_value_payment(df)

print("\n高价值商品支付方式分布:")
print(high_value_payment_dist)


# 任务2: 时间序列模式分析
def temporal_patterns(df):
    # 季节性模式分析
    seasonal = df.groupby(['year', 'quarter']).size().unstack().fillna(0)

    # 月度购买频率
    monthly = df.groupby(['month', 'item_category']).size().unstack().fillna(0)

    # 星期购买模式
    weekday = df.groupby(['day_of_week', 'item_category']).size().unstack().fillna(0)

    # 序列模式分析 (先A后B)
    df_sorted = df.sort_values(['user_name', 'purchase_date'])
    sequences = df_sorted.groupby('user_name')['item_category'].apply(list)

    # 计算常见的2项序列
    from collections import defaultdict
    seq_counts = defaultdict(int)
    for seq in sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            seq_counts[pair] += 1

    # 转换为DataFrame并计算频率
    seq_df = pd.DataFrame.from_dict(seq_counts, orient='index', columns=['count'])
    seq_df['frequency'] = seq_df['count'] / len(sequences)
    common_sequences = seq_df[seq_df['frequency'] >= 0.01].sort_values('frequency', ascending=False)

    return {
        'seasonal': seasonal,
        'monthly': monthly,
        'weekday': weekday,
        'sequences': common_sequences
    }


# 执行分析
time_patterns = temporal_patterns(df)

print("\n季度购买模式:")
print(time_patterns['seasonal'])
print("\n月度商品类别购买频率:")
print(time_patterns['monthly'].head())
print("\n常见购买序列:")
print(time_patterns['sequences'].head(10))


# 任务3: 退款模式分析
def refund_patterns(df):
    # 筛选退款订单（确保order_id是列）
    refunded = df[df['payment_status'].isin(['已退款', '部分退款'])].reset_index()

    # 任务1：退款关联规则分析（商品类别→退款状态）
    # -------------------------------------------
    # 按订单聚合商品类别和退款状态
    order_refunds = (
        refunded.groupby('order_id')
        .agg({
            'item_category': lambda x: list(set(x)),  # 去重后的商品类别列表
            'payment_status': 'first'  # 每个订单只取一个状态
        })
    )

    # 生成事务列表（商品类别+退款状态）
    transactions = []
    for _, row in order_refunds.iterrows():
        transaction = row['item_category'] + [row['payment_status']]
        transactions.append(transaction)
    print(transactions[:5])  # 打印前5个事务数据
    # 编码为布尔矩阵
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘关联规则
    frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
    # 筛选有意义规则（商品类别→退款状态）
    meaningful_rules = rules[
        (rules['lift'] > 1) &
        (rules['consequents'].apply(lambda x: any(s in x for s in ['已退款', '部分退款'])) &
         (rules['antecedents'].apply(lambda x: not any(s in x for s in ['已退款', '部分退款']))))
         ].sort_values('confidence', ascending=False)
    print(meaningful_rules.head(10))
    # 任务2：退款商品组合分析
    # -------------------------------------------
    # 统计最常见的商品类别组合（按订单）
    refund_combos = (
     refunded.groupby('order_id')['item_category']
    .apply(frozenset)  # 使用frozenset避免可变集合的问题
    .value_counts()
    .head(10))
    print(refund_combos)
    return {
        'refund_rules': meaningful_rules,
        'common_combos': refund_combos
    }


# 执行分析
refund_results = refund_patterns(df)

print("\n退款关联规则:")
print(refund_results['refund_rules'][['antecedents', 'consequents', 'support', 'confidence', 'lift']])
print("\n常见退款商品组合:")
print(refund_results['common_combos'])

# 可视化部分
import matplotlib.pyplot as plt

# 1. 高价值商品支付方式可视化
plt.figure(figsize=(10, 6))
high_value_payment_dist.plot(kind='bar')
plt.title('高价值商品支付方式分布')
plt.ylabel('比例')
plt.savefig('high_value_payment_distribution2.png')

# 2. 季节性模式可视化
plt.figure(figsize=(12, 6))
time_patterns['seasonal'].plot(marker='o')
plt.title('季度购买趋势')
plt.ylabel('订单量')
plt.savefig('seasonal_trends2.png')

# 3. 退款商品组合可视化
plt.figure(figsize=(10, 6))
refund_results['common_combos'].head(5).plot(kind='barh')
plt.title('最常见的退款商品组合')
plt.xlabel('出现次数')
plt.savefig('common_refund_combos2.png')
