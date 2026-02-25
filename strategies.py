"""
选股策略模块
包含多种常见选股策略，可自由组合
"""
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta


def get_all_stocks():
    """获取所有A股股票列表（排除ST、科创板等）"""
    # 获取所有A股实时行情
    df = ak.stock_zh_a_spot_em()
    
    # 基础过滤
    df = df[~df['名称'].str.contains('ST|退|N ', na=False)]  # 排除ST、退市、新股
    df = df[~df['代码'].str.startswith('688')]  # 排除科创板（可选）
    df = df[~df['代码'].str.startswith('8')]    # 排除北交所（可选）
    
    # 清洗数据：把 '-' 替换为 NaN，然后转为数值
    numeric_cols = ['最新价', '涨跌幅', '涨跌额', '成交量', '成交额',
                    '振幅', '最高', '最低', '今开', '昨收',
                    '量比', '换手率', '市盈率-动态', '市净率',
                    '总市值', '流通市值', '60日涨跌幅', '年初至今涨跌幅']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def strategy_low_pe_pb(df, pe_max=20, pb_max=2, min_market_cap=50e8):
    """
    策略1：低估值选股
    - 市盈率 < pe_max
    - 市净率 < pb_max
    - 市盈率 > 0（排除亏损股）
    - 总市值 > min_market_cap
    """
    result = df[
        (df['市盈率-动态'] > 0) &
        (df['市盈率-动态'] < pe_max) &
        (df['市净率'] > 0) &
        (df['市净率'] < pb_max) &
        (df['总市值'] > min_market_cap)
    ].copy()
    
    # 按市盈率排序
    result = result.sort_values('市盈率-动态', ascending=True)
    return result


def strategy_volume_breakout(df, min_turnover=3, min_change=3, max_change=9.5):
    """
    策略2：放量突破选股
    - 换手率 > min_turnover%
    - 涨幅在 min_change% ~ max_change% 之间
    - 量比 > 1.5
    """
    result = df[
        (df['换手率'] > min_turnover) &
        (df['涨跌幅'] > min_change) &
        (df['涨跌幅'] < max_change) &
        (df['量比'] > 1.5)
    ].copy()
    
    result = result.sort_values('量比', ascending=False)
    return result


def strategy_strong_trend(df, min_market_cap=100e8):
    """
    策略3：强势趋势股
    - 60日涨跌幅 > 20%
    - 当日涨幅 > 0
    - 换手率在 1%~10% 之间（不能太低也不能太高）
    - 总市值 > 100亿
    """
    result = df[
        (df['60日涨跌幅'] > 20) &
        (df['涨跌幅'] > 0) &
        (df['换手率'] > 1) &
        (df['换手率'] < 10) &
        (df['总市值'] > min_market_cap)
    ].copy()
    
    result = result.sort_values('60日涨跌幅', ascending=False)
    return result


def strategy_ma_golden_cross(stock_code, days=60):
    """
    策略4：均线金叉选股（需要逐个检查，较慢）
    - MA5 上穿 MA20
    - 成交量放大
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
            end_date=datetime.now().strftime('%Y%m%d'),
            adjust="qfq"
        )
        
        if len(df) < 20:
            return False
        
        df['MA5'] = df['收盘'].rolling(5).mean()
        df['MA20'] = df['收盘'].rolling(20).mean()
        df['VOL_MA5'] = df['成交量'].rolling(5).mean()
        
        # 判断金叉：前一天 MA5 < MA20，今天 MA5 >= MA20
        if len(df) >= 2:
            today = df.iloc[-1]
            yesterday = df.iloc[-2]
            
            golden_cross = (
                yesterday['MA5'] < yesterday['MA20'] and
                today['MA5'] >= today['MA20'] and
                today['成交量'] > today['VOL_MA5'] * 1.2  # 成交量放大20%
            )
            return golden_cross
    except Exception:
        pass
    
    return False