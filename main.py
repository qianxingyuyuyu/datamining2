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
from dask.diagnostics import ProgressBar
from collections import defaultdict
import dask

# ==================== 全局配置 ====================
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 加载商品目录数据
with open('product_catalog.json', 'r') as f:
    product_catalog = json.load(f)['products']
# 创建商品信息字典 {id: {category, price}}
product_info = {
    item['id']: {
        'category': item['category'],
        'price': float(item['price'])
    } for item in product_catalog
}

category_mapping = {
    # 电子产品
    "智能手机": "电子产品",
    "笔记本电脑": "电子产品",
    "平板电脑": "电子产品",
    "智能手表": "电子产品",
    "耳机": "电子产品",
    "音响": "电子产品",
    "相机": "电子产品",
    "摄像机": "电子产品",
    "游戏机": "电子产品",
    # 服装
    "上衣": "服装",
    "裤子": "服装",
    "裙子": "服装",
    "内衣": "服装",
    "鞋子": "服装",
    "帽子": "服装",
    "手套": "服装",
    "围巾": "服装",
    "外套": "服装",
    # 食品
    "零食": "食品",
    "饮料": "食品",
    "调味品": "食品",
    "米面": "食品",
    "水产": "食品",
    "肉类": "食品",
    "蛋奶": "食品",
    "水果": "食品",
    "蔬菜": "食品",
    # 家居
    "家具": "家居",
    "床上用品": "家居",
    "厨具": "家居",
    "卫浴用品": "家居",
    # 办公
    "文具": "办公",
    "办公用品": "办公",
    # 运动户外
    "健身器材": "运动户外",
    "户外装备": "运动户外",
    # 玩具
    "玩具": "玩具",
    "模型": "玩具",
    "益智玩具": "玩具",
    # 母婴
    "婴儿用品": "母婴",
    "儿童课外读物": "母婴",
    # 汽车用品
    "车载电子": "汽车用品",
    "汽车装饰": "汽车用品"
}

# ==================== 数据预处理 ====================
def process_partition(partition):
    records = []
    for idx, row in partition.iterrows():
        try:
            history = json.loads(row['purchase_history'])
            purchase_date = pd.to_datetime(history.get('purchase_date', ''))
            purchase_date_str = purchase_date.strftime("%Y%m%d%H%M%S")
            user_prefix = row['user_name'].upper()
            order_id = f"{user_prefix}_{purchase_date_str}"

            # 收集订单级信息
            order_categories = []
            item_count = 0

            for item in history.get('items', []):
                item_id = item.get('id')
                product = product_info.get(item_id, {
                    'category': '未知类别',
                    'price': 0.0
                })
                original_category = product['category']
                main_category = category_mapping.get(original_category, "未知类别")
                order_categories.append(main_category)

                # 创建商品记录
                record = {
                    'user_name': row['user_name'],
                    'order_id': order_id,
                    'item_id': item_id,
                    'item_category': main_category,
                    'payment_method': history.get('payment_method', ''),
                    'payment_status': history.get('payment_status', ''),
                    'price': product['price'],
                    'purchase_date': pd.to_datetime(
                        history.get('purchase_date', '')
                    )
                }
                records.append(record)
                item_count += 1

            # 添加订单级信息到每个商品记录（优化关键点）
            if item_count > 0:
                unique_categories = list(set(order_categories))
                for i in range(len(records) - item_count, len(records)):
                    records[i]['order_categories'] = unique_categories
                    records[i]['category_count'] = len(unique_categories)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    return pd.DataFrame(records)

def process_data(ddf_path):
    # 初始元数据（仅包含原始字段）
    base_meta = {
        'user_name': 'object',
        'order_id': 'object',
        'item_id': 'int64',
        'item_category': 'object',
        'payment_method': 'object',
        'payment_status': 'object',
        'price': 'float64',
        'purchase_date': 'datetime64[ns]',
        'order_categories': 'object',
        'category_count': 'int64'
    }

    # 完整元数据（包含衍生字段）
    full_meta = {
        **base_meta,
        'is_high_value': 'bool',
        'year': 'int64',
        'quarter': 'int64',
        'month': 'int64',
        'day_of_week': 'int64'
    }

    ddf = dd.read_parquet(ddf_path,
                         columns=['user_name', 'purchase_history'],
                         engine='pyarrow')
    
    # 第一步：基础处理（使用基础元数据）
    processed_ddf = ddf.map_partitions(
        process_partition,
        meta=base_meta
    )

    # 第二步：添加衍生字段（使用完整元数据）
    processed_ddf = processed_ddf.map_partitions(
        lambda df: df.assign(
            is_high_value=(df['price'] > 5000).astype('bool'),
            year=df['purchase_date'].dt.year.astype('int64'),
            quarter=df['purchase_date'].dt.quarter.astype('int64'),
            month=df['purchase_date'].dt.month.astype('int64'),
            day_of_week=(df['purchase_date'].dt.dayofweek + 1).astype('int64')
        ),
        meta=full_meta
    ).persist()  # 先持久化确保元数据固定

    # 第三步：设置索引（不再传递meta参数）
    processed_ddf = processed_ddf.set_index(
        'order_id',
        shuffle='disk',
        npartitions='auto'
    )

    return processed_ddf

# ==================== 任务1：商品关联规则 ====================
def task1_association_rules(processed_ddf):
    # 第一步：获取所有可能的商品类别（全局统一）
    all_categories = processed_ddf['item_category'].unique().compute().tolist()

    # 创建全局TransactionEncoder并拟合所有可能类别
    global_te = TransactionEncoder()
    global_te.fit([all_categories])

    # 第二步：处理交易数据
    filtered = processed_ddf[processed_ddf['category_count'] > 1]
    unique_orders = filtered.index.drop_duplicates()
    filtered = filtered.loc[unique_orders].persist()
    # 使用map_partitions并行处理交易数据
    def encode_transactions(df):
        # 将字符串形式的列表转换为实际的Python列表
        import ast
        transactions = df['order_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        ).tolist()

        # 创建新的编码器实例
        te = TransactionEncoder()
        te.columns_ = global_te.columns_
        te.columns_mapping_ = {k: v for v, k in enumerate(te.columns_)}

        # 转换数据
        return pd.DataFrame(
            te.transform(transactions),
            columns=te.columns_,
            dtype=bool
        )

    # 并行编码
    encoded_parts = filtered.map_partitions(
        encode_transactions,
        meta=pd.DataFrame(columns=global_te.columns_, dtype=bool)
    ).compute()

    # 合并结果
    sparse_matrix = csr_matrix(np.vstack(encoded_parts.values), dtype=np.bool_)

    # 创建DataFrame
    df = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        columns=global_te.columns_
    )

    # 后续处理
    frequent_itemsets = fpgrowth(
        df,
        min_support=0.02,
        use_colnames=True,
        max_len=5
    )

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

    electronics_rules = rules[
        rules['antecedents'].apply(lambda x: '电子产品' in x) |
        rules['consequents'].apply(lambda x: '电子产品' in x)
        ]

    plt.figure(figsize=(16, 10))
    ax = sns.scatterplot(
        data=rules,
        x='support',
        y='confidence',
        size='lift',
        hue='lift',
        palette='viridis',
        sizes=(50, 500)  # 控制气泡大小范围
    )
    # 筛选TOP10高价值规则（按提升度排序）
    top_rules = rules.nlargest(10, 'lift')
    # 添加标签
    for line in top_rules.itertuples():
        # 将frozenset转换为可读字符串
        antecedents = ", ".join(line.antecedents)
        consequents = ", ".join(line.consequents)
        label = f"{antecedents} → {consequents}"
        ax.annotate(
            label,
            xy=(line.support, line.confidence),
            xytext=(10, 10),  # 标签偏移量
            textcoords='offset points',
            fontsize=10,
            arrowprops=dict(
                arrowstyle="->",
                color='gray',
                lw=1
            ),
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.8
            )
        )
    # 添加辅助线
    plt.axhline(y=0.7, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=0.05, color='gray', linestyle='--', alpha=0.3)
    plt.title("商品类别关联规则（标注TOP10高提升度规则）", fontsize=14)
    plt.xlabel("支持度", fontsize=12)
    plt.ylabel("置信度", fontsize=12)
    plt.legend(title='提升度', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('task1_rules_labeled.png', dpi=300)
    return electronics_rules.sort_values('lift', ascending=False)


# ==================== 任务2：支付方式分析 ====================
def task2_payment_analysis(processed_ddf):
    high_value = processed_ddf[processed_ddf['is_high_value']]
    payment_dist = high_value['payment_method'].value_counts(normalize=True).compute()
    plot_data = payment_dist.reset_index()
    plot_data.columns = ['payment_method', 'percentage']

    plt.figure(figsize=(10, 7))
    plt.pie(
        plot_data['percentage'],
        labels=plot_data['payment_method'],
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("高价值商品支付方式分布")
    plt.tight_layout()
    plt.savefig('task2_payment.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ==================== 任务2.2: 支付方式与商品类别的关联规则 ====================
    # 第一步：获取所有可能的支付方式和商品类别（全局统一）
    all_payments = processed_ddf['payment_method'].unique().compute().tolist()
    all_categories = processed_ddf['item_category'].unique().compute().tolist()

    # 创建全局编码器（包含支付方式和商品类别）
    global_te = TransactionEncoder()
    global_te.fit([[f"payment_{p}"] + [c] for p in all_payments for c in all_categories])

    # 第二步：准备并行处理的数据
    filtered = processed_ddf[processed_ddf['category_count'] > 0].persist()

    # 并行编码函数
    def encode_payment_transactions(df):
        # 按订单分组生成事务
        transactions = df.groupby('order_id').apply(
            lambda x: [f"payment_{x['payment_method'].iloc[0]}"] + x['item_category'].tolist()
        ).tolist()

        # 使用全局编码器
        te = TransactionEncoder()
        te.columns_ = global_te.columns_
        te.columns_mapping_ = {k: v for v, k in enumerate(te.columns_)}

        return pd.DataFrame(
            te.transform(transactions),
            columns=te.columns_,
            dtype=bool
        )

    # 并行编码
    encoded_parts = filtered.map_partitions(
        encode_payment_transactions,
        meta=pd.DataFrame(columns=global_te.columns_, dtype=bool)
    ).compute()

    # 合并结果（稀疏矩阵优化）
    sparse_matrix = csr_matrix(np.vstack(encoded_parts.values), dtype=np.bool_)
    df_trans = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        columns=global_te.columns_
    )

    # 后续处理（保持不变）
    frequent_itemsets = fpgrowth(df_trans, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    # 筛选规则（优化版）
    payment_rules = rules[
        rules['antecedents'].apply(
            lambda x: any('payment_' in item for item in x) and
                      not any('payment_' in item for item in rules['consequents'])
        )
    ].sort_values('lift', ascending=False)

    return {
        'high_value_payment_dist': payment_dist.to_dict(),
        'payment_category_rules': payment_rules
    }


# ==================== 任务3：时间序列分析 ====================
def task3_time_analysis(processed_ddf):
    time_analysis = (
        processed_ddf.groupby([
            'item_category',
            processed_ddf['purchase_date'].dt.year.rename('year'),
            processed_ddf['purchase_date'].dt.month.rename('month')
        ])
        .size()
        .unstack(['year', 'month'])
        .compute()
    )

    # 2. 格式化列标签
    time_analysis.columns = [f"{y}-{m:02d}" for y, m in time_analysis.columns]

    # 3. 筛选Top5品类
    top_categories = processed_ddf['item_category'].value_counts().nlargest(5).index.compute()

    # 4. 绘图
    plt.figure(figsize=(16, 6))
    sns.heatmap(
        time_analysis.loc[top_categories].fillna(0),
        cmap='YlOrRd',
        annot=True,
        fmt='.0f',
        linewidths=0.5
    )
    plt.title('Top5品类销售季节性热力图（按年月）')
    plt.xticks(rotation=45, ha='right')  # 标签旋转45度
    plt.tight_layout()
    plt.savefig('task3_seasonality.png', dpi=300)

    # ==================== 3.3 时序模式挖掘 ====================
    # 改进的序列模式挖掘（考虑时间窗口）
    def extract_sequences(user_group):
        user_group = user_group.sort_values('purchase_date')
        sequences = []
        for i in range(len(user_group) - 1):
            time_diff = (user_group['purchase_date'].iloc[i + 1] - user_group['purchase_date'].iloc[i]).days
            if time_diff <= 30:  # 30天内的连续购买
                sequences.append((
                    user_group['item_category'].iloc[i],
                    user_group['item_category'].iloc[i + 1],
                    time_diff
                ))
        return pd.DataFrame(sequences, columns=['antecedent', 'consequent', 'days_diff'])

    seq_rules = (processed_ddf.groupby('user_name')
                 .apply(extract_sequences,
                        meta={'antecedent': 'object', 'consequent': 'object', 'days_diff': 'int64'})
                 .compute())
    #print(seq_rules.head())
    seq_rules['days_diff'] = pd.to_numeric(seq_rules['days_diff'], errors='coerce').fillna(0)
    # 规则统计与筛选
    seq_stats = (
        seq_rules.reset_index(drop=True)  # 去掉 user_name 索引
        .groupby(['antecedent', 'consequent'])
        .agg(
            count=('days_diff', 'size'),
            avg_days=('days_diff', 'mean')
        )
        .reset_index()
    )

    # 计算支持度（总订单数为分母）
    print(processed_ddf.head())
    total_orders = processed_ddf.index.nunique().compute()
    seq_stats['support'] = seq_stats['count'] / total_orders
    seq_stats = seq_stats.sort_values('count', ascending=False)

    # ==================== 可视化 ====================

    # 3. 时序规则网络图（筛选前10%规则）
    top_rules = seq_stats[seq_stats['count'] > seq_stats['count'].quantile(0.9)]
    plt.figure(figsize=(10, 8))
    G = nx.from_pandas_edgelist(
        top_rules.head(20),  # 仅展示前20条规则
        source='antecedent',
        target='consequent',
        edge_attr='count'
    )
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=700,
        node_color='skyblue',
        edge_color='gray',
        width=[d['count'] / 50 for (u, v, d) in G.edges(data=True)]
    )
    plt.title('Top20购物序列模式')
    plt.savefig('task3_sequences.png')

    return {
        'sequence_rules': seq_stats.to_dict('records')
    }


# ==================== 任务4：退款分析 ====================
def task4_refund_analysis(processed_ddf):
    # ==================== 4.1 数据准备 ====================
    refund = processed_ddf[processed_ddf['payment_status'].isin(['已退款', '部分退款'])].persist()

    # 获取全局商品类别（用于统一编码）
    all_categories = processed_ddf['item_category'].unique().compute().tolist()

    # ==================== 4.2 并行事务编码 ====================
    # 创建全局事务编码器
    global_te = TransactionEncoder()
    global_te.fit([all_categories])  # 仅需编码商品类别

    # 定义分区处理函数
    def encode_refund_transactions(df):
        # 按订单分组生成事务
        transactions = df.groupby('order_id')['item_category'].apply(list).tolist()

        # 使用全局编码器
        te = TransactionEncoder()
        te.columns_ = global_te.columns_
        te.columns_mapping_ = {k: v for v, k in enumerate(te.columns_)}

        return pd.DataFrame(
            te.transform(transactions),
            columns=te.columns_,
            dtype=bool
        )

    # 并行编码
    encoded_parts = refund.map_partitions(
        encode_refund_transactions,
        meta=pd.DataFrame(columns=global_te.columns_, dtype=bool)
    ).compute()

    # ==================== 4.3 关联规则挖掘 ====================
    # 合并稀疏矩阵
    sparse_matrix = csr_matrix(np.vstack(encoded_parts.values), dtype=np.bool_)
    df_trans = pd.DataFrame.sparse.from_spmatrix(
        sparse_matrix,
        columns=global_te.columns_
    )

    # 挖掘频繁项集（使用FP-Growth加速）
    frequent_itemsets = fpgrowth(
        df_trans,
        min_support=0.005,
        use_colnames=True,
        max_len=3  # 限制规则长度
    )

    # 生成关联规则
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.4
    ).sort_values('lift', ascending=False)

    # ==================== 4.4 增强可视化 ====================
    # 1. 退款类别分布热力图
    refund_status = refund.compute()
    cross_tab = pd.crosstab(
        refund_status['item_category'],
        refund_status['payment_status'],
        normalize='index'  # 按行标准化
    )

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cross_tab,
        cmap='YlOrRd',
        annot=True,
        fmt='.2%',
        linewidths=0.5
    )
    plt.title('各品类退款率分布')
    plt.savefig('task4_refund_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 添加退款规则网络图
    top_rules = rules.head(10)
    plt.figure(figsize=(10, 6))
    G = nx.from_pandas_edgelist(
        top_rules,
        source='antecedents',
        target='consequents',
        edge_attr='lift',
        edge_key='rule'
    )
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=500)
    nx.draw_networkx_edges(
        G, pos,
        width=[d['lift'] / 2 for (u, v, d) in G.edges(data=True)],
        edge_color='gray'
    )
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title('Top10退款关联规则')
    plt.savefig('task4_refund_rules.png')
    plt.close()

    return {
        'refund_rules': rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        .to_dict('records'),
        'refund_rates': cross_tab.to_dict()
    }


# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 数据预处理
    processed_ddf = process_data('30G_data_new/')
    print(processed_ddf.head())
    # 执行任务
    print('开始执行任务1...')
    task1_results = task1_association_rules(processed_ddf)
    task1_results.to_csv('task1_electronics_rules.csv')
    print('开始执行任务2...')
    task2_results = task2_payment_analysis(processed_ddf)
    print('开始执行任务3...')
    task3_results = task3_time_analysis(processed_ddf)
    print('开始执行任务4...')
    task4_results = task4_refund_analysis(processed_ddf)
    
    # 保存所有结果
    with pd.ExcelWriter('analysis_results.xlsx') as writer:
    
        task1_results.to_excel(writer, sheet_name="电子产品关联规则")
    
        task2_results['payment_category_rules'].to_excel(
            writer,
            sheet_name="支付方式关联规则"
        )
    
        pd.DataFrame(task3_results['sequence_rules']).to_excel(
            writer,
            sheet_name="序列规则"
        )
    
        pd.DataFrame(task4_results['refund_rules']).to_excel(
            writer,
            sheet_name="退款关联规则"
        )
