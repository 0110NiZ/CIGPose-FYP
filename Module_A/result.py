"""用中文打印训练结果，也可以读取已有 history.json，不需要重新训练。"""
import argparse
import json
from pathlib import Path

import config


# 显示用 0 到 100 标度，JSON 文件仍保留原来的 0 到 1 数值。
METRICS = [
    ('coco/AP', '整体 AP'),
    ('coco/AP .5', 'AP50'),
    ('coco/AP .75', 'AP75'),
    ('coco/AP (M)', '中等人体 AP'),
    ('coco/AP (L)', '大型人体 AP'),
    ('coco/AR', '整体 AR'),
    ('coco/AR .5', 'AR50'),
    ('coco/AR .75', 'AR75'),
    ('coco/AR (M)', '中等人体 AR'),
    ('coco/AR (L)', '大型人体 AR'),
]


def metric_text(metrics):
    lines = ['指标采用 0 到 100 标度，越高越好']
    for key, name in METRICS:
        if key in metrics:
            lines.append(f'{name}：{metrics[key] * 100:.4f}')
    return '\n'.join(lines)


def print_epoch(row, total_epochs, improved=False):
    print(f'\n第 {row["epoch"]} 轮完成，共 {total_epochs} 轮', flush=True)
    print(f'训练人数：{row["count"]:,}')
    print(f'训练损失：{row["loss"]:.6f}')
    print(f'收益预测平均绝对误差：{row["train_gain_mae"]:.6f}')
    print(f'本轮训练耗时：{row["train_seconds"]:.1f} 秒，不含验证')
    if 'validation' in row:
        print('本轮验证结果')
        print(metric_text(row['validation']))
        if improved:
            print('本轮 AP 创下当前最高值，已保存 best.pt')
    else:
        print('本轮未安排验证')
    print('本轮进度已保存到 last.pt 和 history.json', flush=True)


def summary_text(history, output, total_epochs=None):
    if not history:
        return '还没有已完成的训练轮次\n'
    output = Path(output).resolve()
    last = history[-1]
    complete = total_epochs is not None and last['epoch'] >= total_epochs
    lines = ['训练最终结果' if complete else '当前已记录的训练结果',
             f'已完成轮数：{last["epoch"]}']
    if total_epochs is not None:
        lines.append(f'计划训练轮数：{total_epochs}')
    lines += ['', '每轮训练记录']
    for row in history:
        lines.append(f'第 {row["epoch"]} 轮   损失 {row["loss"]:.6f}'
                     f'   收益预测误差 {row["train_gain_mae"]:.6f}'
                     f'   训练耗时 {row["train_seconds"]:.1f} 秒')
    validated = [row for row in history if 'coco/AP' in row.get('validation', {})]
    if validated:
        # AP 相同保留最早的一轮，与训练时保存 best.pt 的规则一致。
        best = max(validated, key=lambda row: row['validation']['coco/AP'])
        lines += ['', '各次验证结果，指标采用 0 到 100 标度']
        for row in validated:
            metrics = row['validation']
            line = f'第 {row["epoch"]} 轮   AP {metrics["coco/AP"] * 100:.4f}'
            if 'coco/AR' in metrics:
                line += f'   AR {metrics["coco/AR"] * 100:.4f}'
            lines.append(line)
        lines += ['', '最终选用的最佳模型', f'最佳轮次：第 {best["epoch"]} 轮',
                  metric_text(best['validation']),
                  f'最佳模型文件：{output / "best.pt"}']
    else:
        lines += ['', '尚无验证结果，暂时不能选择最佳模型']
    lines += ['', f'最后完成的一轮：第 {last["epoch"]} 轮',
              f'最后一轮训练损失：{last["loss"]:.6f}',
              f'最后一轮收益预测误差：{last["train_gain_mae"]:.6f}']
    if 'validation' in last:
        lines += [f'最后一轮 AP：{last["validation"]["coco/AP"] * 100:.4f}']
        if 'coco/AR' in last['validation']:
            lines += [f'最后一轮 AR：{last["validation"]["coco/AR"] * 100:.4f}']
    lines += ['', f'结果目录：{output}', f'文字总结：{output / "result.txt"}',
              '原始完整记录：history.json',
              '是否提升需与相同设置下的 Top-1 和 P0 基线比较，仅凭训练损失不能判断。']
    return '\n'.join(lines) + '\n'


def show_summary(history, output, total_epochs=None):
    text = summary_text(history, output, total_epochs)
    output = Path(output)
    temp = output / 'result.txt.tmp'
    temp.write_text(text, encoding='utf-8')
    temp.replace(output / 'result.txt')
    print('\n' + text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=config.OUTPUT)
    args = parser.parse_args()
    history = json.loads((args.output / 'history.json').read_text())
    settings_path = args.output / 'settings.json'
    epochs = json.loads(settings_path.read_text()).get('epochs') if settings_path.exists() else None
    show_summary(history, args.output, epochs)


if __name__ == '__main__':
    main()
