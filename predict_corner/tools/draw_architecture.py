from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    root = Path(__file__).resolve().parents[2]
    output = root / 'predict_corner/docs/figures'
    output.mkdir(exist_ok=True)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig, ax = plt.subplots(figsize=(12.5, 11.25), dpi=160)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 2000), ylim=(1800, 0))
    ax.axis('off')
    fig.patch.set_facecolor('#f7f9fc')
    blue, orange, green, ink = '#3167a7', '#ba6a28', '#26725c', '#172a3a'

    def text(x, y, value, size=13, color=ink, weight='normal', align='center'):
        ax.text(x, y, value, fontproperties=font, fontsize=size, color=color,
                fontweight=weight, ha=align, va='center', linespacing=1.45)

    def box(x, y, w, h, face, edge, title, body, title_size=14, body_size=12.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=16',
                                   facecolor=face, edgecolor=edge, linewidth=1.4))
        text(x + w / 2, y + 34, title, title_size, edge, 'bold')
        text(x + w / 2, y + 34 + (h - 40) / 2, body, body_size)

    def arrow(x1, y1, x2, y2, color='#718297', dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=15, linewidth=1.6, color=color,
                                     linestyle='--' if dashed else '-'))

    text(65, 65, '当前角点预测网络 · DINOv2 + Transformer', 23, weight='bold', align='left')
    text(65, 118, '单张 RGB 输入  |  DINOv2 通用预训练特征固定  |  角点 decoder 随机初始化、自行训练',
         12.8, '#52667a', align='left')
    box(65, 180, 280, 178, '#ffffff', '#718297', 'RGB + Padding',
        '480 × 640\n↓\n490 × 644', body_size=12.5)
    box(400, 180, 430, 178, '#e9f1fb', blue, '冻结 DINOv2 ViT-B/14 reg',
        '图像 → 1610 个 patch 特征\n(N, 1610, 768)')
    box(885, 180, 570, 178, '#fff0e2', orange, '可训练 Transformer 热图 decoder',
        '12 层 · 768 维 · 8 heads\n输出八通道像素 logits')
    box(1510, 180, 425, 178, '#e7f4ee', green, '热图与二维角点',
        '8 张热图：(N, 8, 480, 640)\n8 个角点：(N, 8, 2)')
    for a, b in [(345, 400), (830, 885), (1455, 1510)]:
        arrow(a + 5, 270, b - 5, 270)

    ax.add_patch(FancyBboxPatch((65, 415), 890, 1190, boxstyle='round,pad=0,rounding_size=20',
                               facecolor='#f0f5fc', edgecolor='#bad0e9', linewidth=1.2))
    ax.add_patch(FancyBboxPatch((1005, 415), 930, 1190, boxstyle='round,pad=0,rounding_size=20',
                               facecolor='#fff7ee', edgecolor='#eccdae', linewidth=1.2))
    text(510, 460, 'DINOv2 ViT-B/14 内部结构', 17, blue, 'bold')
    text(1470, 460, '角点 decoder 内部结构', 17, orange, 'bold')
    text(510, 508, '约 8658 万参数 · eval · 无梯度', 12, blue)
    text(1470, 508, '约 8745 万参数 · 由合成数据训练', 12, orange)

    box(110, 555, 800, 150, '#ffffff', blue, 'Patch Embedding',
        'Conv2d：3 → 768，kernel = stride = 14\n35 × 46 网格 → (N, 1610, 768)')
    arrow(510, 710, 510, 750, blue)
    box(110, 755, 800, 155, '#ffffff', blue, '位置与附加 tokens',
        'patch 位置编码 + 1 个 CLS + 4 个 register tokens\n进入 ViT 的序列：(N, 1615, 768)')
    arrow(510, 915, 510, 955, blue)
    box(110, 960, 800, 355, '#e2edfb', blue, '12 × ViT Encoder Block',
        'LayerNorm\nMulti-Head Self-Attention：12 heads × 64 维\nLayerScale + Residual\n'
        'LayerNorm\nMLP：768 → 3072 → 768，GELU\nLayerScale + Residual', body_size=13)
    arrow(510, 1320, 510, 1360, blue)
    box(110, 1365, 800, 170, '#ffffff', blue, '取图像 patch 特征',
        '最终 LayerNorm → x_norm_patchtokens\n取 1610 个 patch；CLS / registers 不传入 decoder')

    box(1050, 555, 840, 185, '#ffffff', orange, '任务特征与位置',
        'MLP adapter + LayerNorm\n加共享的 learned query 向量与 2D sin/cos 位置编码\n(N, 1610, 768)', body_size=12.5)
    arrow(1470, 745, 1470, 785, orange)
    box(1050, 790, 840, 355, '#ffead7', orange, '12 × Decoder Self-Attention Block',
        'LayerNorm → Q / K / V，Q/K RMSNorm\nSelf-Attention：8 heads × 96 维\nResidual\n'
        'LayerNorm\nMLP：768 → 3072 → 768，GELU\nResidual', body_size=13)
    arrow(1470, 1150, 1470, 1190, orange)
    box(1050, 1195, 840, 155, '#ffffff', orange, '像素预测头',
        'LayerNorm + Linear：768 → 1568\n每个 token 输出 14 × 14 × 8 个 logits')
    arrow(1470, 1355, 1470, 1395, orange)
    box(1050, 1400, 840, 155, '#ffffff', orange, 'Unpatchify + Crop',
        '拼接 35 × 46 个块 → (N, 8, 490, 644)\n裁去右侧 4 像素与底部 10 像素 → (N, 8, 480, 640)')

    text(65, 1650, '训练：DINOv2 的 float32 特征只计算一次并缓存；梯度更新橙色 decoder。',
         13, color=ink, align='left')
    text(65, 1702, '监督：Sigmoid 热图 → coarse；logits 的 soft-argmax（T = 0.25）→ fine；总 loss = coarse + 2 × fine。',
         12.5, color=ink, align='left')
    text(65, 1751, '当前为单图适配：不输入参考图；使用 DINOv2 通用预训练，未加载 BoxDreamer 任务权重。',
         12.3, color='#52667a', align='left')
    fig.savefig(output / 'dino_transformer_architecture.png', dpi=160)
    fig.savefig(output / 'dino_transformer_architecture.svg')
    plt.close(fig)
    print(output / 'dino_transformer_architecture.png')


if __name__ == '__main__':
    main()
