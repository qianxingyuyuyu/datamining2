import pandas as pd
import json
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from scipy.sparse import csr_matrix
from collections import defaultdict
from matplotlib.ticker import MaxNLocator

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
def process_data(df):
    order_records = []
    for idx, row in df.iterrows():
        try:
            history = json.loads(row['purchase_history'])
            purchase_date = pd.to_datetime(history.get('purchase_date', ''))
            purchase_date_str = purchase_date.strftime("%Y%m%d%H%M%S")
            user_prefix = row['user_name'][:4].upper()
            order_id = f"{user_prefix}_{purchase_date_str}"

            # 初始化订单级变量
            categories = set()
            high_value_categories = set()
            item_count = 0
            max_price = 0.0

            # 遍历订单中的商品
            for item in history.get('items', []):
                item_id = item.get('id')
                product = product_info.get(item_id, {'category': '未知类别', 'price': 0.0})
                main_category = category_mapping.get(product['category'], "未知类别")
                categories.add(main_category)

                # 记录高价值类别
                if product['price'] > 5000:
                    high_value_categories.add(main_category)

                # 记录最高单价
                if product['price'] > max_price:
                    max_price = product['price']

                item_count += 1

            # 构建订单记录
            order_record = {
                'user_name': row['user_name'],
                'order_id': order_id,
                'item_categories': list(categories),
                'high_value_categories': list(high_value_categories),
                'payment_method': history.get('payment_method', ''),
                'payment_status': history.get('payment_status', ''),
                'purchase_date': pd.to_datetime(history.get('purchase_date', '')),
                'max_price': max_price,  # 用于高价值判断
                'item_count': item_count
            }
            order_records.append(order_record)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    return pd.DataFrame(order_records)


# ==================== 任务1：商品关联规则 ====================
def task1_association_rules(df):
    filtered = df[df['item_count'] > 1]
    # 获取所有事务数据（列表的列表）
    transactions = filtered['item_categories'].tolist()
    transactions = [ast.literal_eval(s) if isinstance(s, str) else s for s in transactions if s]
    print(transactions[:5])

    # 初始化 TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘频繁项集
    frequent_itemsets = fpgrowth(
        df_trans,
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
    print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
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
def task2_payment_analysis(df):
    high_value = df[df['max_price'] > 5000]
    payment_dist = high_value['payment_method'].value_counts(normalize=True)
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
    transactions = []
    for _, row in df.iterrows():
        payment = f"payment_{row.payment_method}"
        # 确保 item_categories 是列表格式
        categories = ast.literal_eval(row.item_categories) if isinstance(row.item_categories,
                                                                         str) else row.item_categories
        # 为每个类别生成独立事务
        for category in categories:
            transactions.append([payment, category])

    print("示例事务数据:", transactions[:5])
    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘频繁项集
    frequent_itemsets = fpgrowth(df_trans, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

    print("事务矩阵列名:", df_trans.columns)
    print("频繁项集数量:", len(frequent_itemsets))
    if not frequent_itemsets.empty:
        print("二元项集样例:")
        print(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 2].head())
    print(rules.head(10))
    return {
        'high_value_payment_dist': payment_dist.to_dict(),
        'payment_category_rules': rules
    }


# ==================== 任务3：时间序列分析 ====================
def task3_time_analysis(df):
    df['item_categories'] = df['item_categories'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 2. 生成序列（按用户分组）
    def generate_sequences(user_group):
        # 确保item_categories是列表类型
        user_group = user_group.sort_values('purchase_date')
        user_group['item_categories'] = user_group['item_categories'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # 收集所有结果
        antecedents = []
        consequents = []
        days_diffs = []

        # 获取所有购买记录
        records = list(zip(
            user_group['item_categories'],
            user_group['purchase_date']
        ))

        # 遍历每对购买记录
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                cats_i, date_i = records[i]
                cats_j, date_j = records[j]

                # 为当前购买对生成所有组合
                for cat_i in cats_i:
                    for cat_j in cats_j:
                        antecedents.append(cat_i)
                        consequents.append(cat_j)
                        days_diffs.append((date_j - date_i).days)

        return pd.DataFrame({
            'antecedent': antecedents,
            'consequent': consequents,
            'days_diff': days_diffs
        })

    # 使用groupby.apply处理
    seq_rules = df.groupby('user_name').apply(generate_sequences).reset_index(drop=True)
    print(seq_rules.head())

    # 计算基础指标
    total_users = df['user_name'].nunique()

    item_category_counts = df.explode('item_categories')['item_categories'].value_counts()

    seq_rules['days_diff'] = seq_rules['days_diff'].astype(int)
    seq_stats = (
        seq_rules.groupby(['antecedent', 'consequent'])
        .agg(count=('days_diff', 'size'),
             avg_days=('days_diff', 'mean'))
        .reset_index()
    )

    # 计算高级指标
    seq_stats['support'] = seq_stats['count'] / total_users
    seq_stats['antecedent_support'] = seq_stats['antecedent'].map(item_category_counts) / total_users
    seq_stats['consequent_support'] = seq_stats['consequent'].map(item_category_counts) / total_users
    seq_stats['confidence'] = seq_stats['support'] / seq_stats['antecedent_support']
    seq_stats['lift'] = seq_stats['support'] / (seq_stats['antecedent_support'] * seq_stats['consequent_support'])

    return seq_stats.sort_values('count', ascending=False)


# ==================== 任务4：退款分析 ====================
def task4_refund_analysis(df):
    refund = df[df['payment_status'].isin(['已退款', '部分退款'])]

    transactions = refund['item_categories'].tolist()
    transactions = [ast.literal_eval(s) if isinstance(s, str) else s for s in transactions if s]
    print(transactions[:5])

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    # 挖掘频繁项集
    frequent_itemsets = fpgrowth(
        df_trans,
        min_support=0.005,
        use_colnames=True,
        max_len=3
    )

    # 生成关联规则
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.4
    ).sort_values('lift', ascending=False)
    print(rules.head(10))
    return {
        'refund_rules': rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    }


# ==================== 主执行流程 ====================
if __name__ == "__main__":
    # 读取数据
    df = pd.read_parquet('10G_data_new/', columns=['user_name', 'purchase_history'])

    # 数据预处理
    processed_df = process_data(df)
    print('----------------------')
    print(processed_df.head())

    # 执行任务
    print('开始执行任务1...')
    task1_results = task1_association_rules(processed_df)
    task1_results.to_csv('task1_electronics_rules.csv')

    print('开始执行任务2...')
    task2_results = task2_payment_analysis(processed_df)
    task2_results['payment_category_rules'].to_csv('task2_payment_rules.csv')

    print('开始执行任务3...')
    task3_results = task3_time_analysis(processed_df)
    task3_results.to_csv('task3_sequence_rules.csv')

    print('开始执行任务4...')
    task4_results = task4_refund_analysis(processed_df)
    task4_results['refund_rules'].to_csv('task4_refund_rules.csv')