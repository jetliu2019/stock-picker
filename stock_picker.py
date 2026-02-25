"""
A股自动选股主程序
每个交易日收盘后运行，筛选符合条件的股票并推送到手机
"""
import pandas as pd
from datetime import datetime, timezone, timedelta
from strategies import (
    get_all_stocks,
    strategy_low_pe_pb,
    strategy_volume_breakout,
    strategy_strong_trend,
)
from notifier import Notifier


def is_trading_day():
    """简单判断是否为交易日（排除周末，节假日需额外处理）"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    # 周一到周五
    return now.weekday() < 5


def format_number(num):
    """格式化大数字"""
    if pd.isna(num):
        return '-'
    if abs(num) >= 1e8:
        return f'{num/1e8:.1f}亿'
    elif abs(num) >= 1e4:
        return f'{num/1e4:.1f}万'
    else:
        return f'{num:.2f}'


def format_results(name, df, max_count=10):
    """将选股结果格式化为可读文本"""
    if df.empty:
        return f"\n### {name}\n暂无符合条件的股票\n"
    
    display_df = df.head(max_count)
    lines = [f"\n### {name}（共{len(df)}只，显示前{min(max_count, len(df))}只）\n"]
    lines.append("| 代码 | 名称 | 现价 | 涨跌幅 | PE | PB | 总市值 | 换手率 |")
    lines.append("|------|------|------|--------|-----|-----|--------|--------|")
    
    for _, row in display_df.iterrows():
        lines.append(
            f"| {row['代码']} | {row['名称']} | "
            f"{row.get('最新价', '-')} | "
            f"{row.get('涨跌幅', '-'):.2f}% | "
            f"{row.get('市盈率-动态', '-')} | "
            f"{row.get('市净率', '-')} | "
            f"{format_number(row.get('总市值', 0))} | "
            f"{row.get('换手率', '-')}% |"
        )
    
    return '\n'.join(lines)


def main():
    print("=" * 60)
    print(f"📊 A股自动选股程序启动")
    
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    print(f"⏰ 当前时间（北京时间）: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 判断交易日（简单版，仅排除周末）
    if not is_trading_day():
        print("📅 今天不是交易日（周末），跳过选股")
        return
    
    # ========== 1. 获取全市场数据 ==========
    print("\n🔄 正在获取全市场行情数据...")
    try:
        all_stocks = get_all_stocks()
        print(f"✅ 共获取 {len(all_stocks)} 只股票数据")
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        notifier = Notifier()
        notifier.send("选股程序异常", f"获取行情数据失败: {e}")
        return
    
    # ========== 2. 运行各选股策略 ==========
    results = []
    
    # 策略1：低估值
    print("\n📈 运行策略1：低估值选股...")
    low_pe = strategy_low_pe_pb(all_stocks, pe_max=15, pb_max=1.5, min_market_cap=100e8)
    results.append(format_results("📊 低估值策略 (PE<15, PB<1.5, 市值>100亿)", low_pe))
    print(f"  → 筛选出 {len(low_pe)} 只")
    
    # 策略2：放量突破
    print("📈 运行策略2：放量突破选股...")
    vol_break = strategy_volume_breakout(all_stocks, min_turnover=3, min_change=3)
    results.append(format_results("🚀 放量突破策略 (换手>3%, 涨幅3~9.5%, 量比>1.5)", vol_break))
    print(f"  → 筛选出 {len(vol_break)} 只")
    
    # 策略3：强势趋势
    print("📈 运行策略3：强势趋势选股...")
    strong = strategy_strong_trend(all_stocks, min_market_cap=200e8)
    results.append(format_results("💪 强势趋势策略 (60日涨幅>20%, 市值>200亿)", strong))
    print(f"  → 筛选出 {len(strong)} 只")
    
    # ========== 3. 汇总结果 ==========
    date_str = now.strftime('%Y-%m-%d')
    
    # 构建推送内容
    title = f"📊 {date_str} A股选股报告"
    
    summary_lines = [
        f"**选股时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**全市场股票数**: {len(all_stocks)}",
        f"**上证指数涨跌**: 请查看行情软件",
        "",
        "---",
    ]
    
    content = '\n'.join(summary_lines) + '\n'.join(results)
    content += "\n\n---\n⚠️ *以上数据仅供参考，不构成投资建议*"
    
    print("\n" + "=" * 60)
    print(content)
    print("=" * 60)
    
    # ========== 4. 推送到手机 ==========
    print("\n📱 正在推送选股结果...")
    notifier = Notifier()
    notifier.send(title, content)
    
    print("\n✅ 选股程序运行完毕！")


if __name__ == '__main__':
    main()