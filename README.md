# VAE 图像超分辨率（×2）实验项目

本项目使用 **Conditional VAE（条件变分自编码器，CVAE）** 完成单幅图像 ×2 超分辨率重建。输入低分辨率图像（例如 128×128），输出 256×256 图像。

## 1. 环境

建议 Python 3.10/3.11，Windows、Linux、macOS 均可。

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

安装 PyTorch 时，如果需要 CUDA，请按 PyTorch 官网安装选择器选择与你的显卡驱动匹配的命令：
https://pytorch.org/get-started/locally/

## 2. 下载 DIV2K

DIV2K 官方数据集页面：https://data.vision.ee.ethz.ch/cvl/DIV2K/

```bash
python download_data.py --split train
python download_data.py --split valid
```

训练集压缩包较大（数 GB），请预留足够磁盘空间和网络流量。下载脚本会将图像解压到：

```text
data/DIV2K_train_HR/
data/DIV2K_valid_HR/
```

如果你已从官方页面手动下载并解压，请把训练图像放入 `data/DIV2K_train_HR/`，无需再次下载。

## 3. 训练

先用少量 epoch 验证环境和数据路径：

```bash
python train.py --data_dir data/DIV2K_train_HR --epochs 2 --batch_size 4 --crop_size 256
```

正式训练示例：

```bash
python train.py --data_dir data/DIV2K_train_HR --epochs 50 --batch_size 8 --crop_size 256 --lr 1e-4 --beta 1e-4
```

若显存不足，降低 `--batch_size`（例如 2 或 4），或降低 `--crop_size`。模型默认自动使用 CUDA（若可用），否则使用 CPU。

训练输出：
- `outputs/checkpoints/best.pt`：训练损失最低的权重
- `outputs/checkpoints/last.pt`：最后一个 epoch 权重
- `outputs/loss_curve.png`：训练损失曲线
- `outputs/train_log.csv`：每轮损失记录

## 4. 测试与可视化

准备任意 RGB 图片，运行：

```bash
python test.py --checkpoint outputs/checkpoints/best.pt --input path/to/your_image.jpg --output_dir outputs/test
```

测试脚本会把原图缩小一半，再将双三次放大图分块送入模型（默认 256 像素 tile、32 像素重叠）以降低显存占用，并保存：
- `*_lr.png`：低分辨率输入
- `*_bicubic.png`：双三次插值基线
- `*_vae.png`：CVAE 输出
- `*_comparison.png`：低分辨率最近邻放大、双三次插值、CVAE 输出三种结果拼图

如果测试图片的原始宽高不是偶数，脚本会裁掉最右侧/最下侧多出的像素，使 ×2 尺寸严格对应。

### PSNR / SSIM

若输入的是 DIV2K 验证集中的原图，可以与其下采样后的图像配对评估：

```bash
python evaluate.py --checkpoint outputs/checkpoints/best.pt \
  --hr_dir data/DIV2K_valid_HR --output_dir outputs/evaluation
```

评估会对每张验证图像计算 bicubic 和 CVAE 的 PSNR、SSIM、MAE，并保存 `metrics.csv` 和均值汇总。**只有存在对应高分辨率真值（HR ground truth）时，PSNR/SSIM 才有意义。** 对任意网络图片进行推理时不能凭空计算这两个指标。

## 5. 模型说明

模型是**空间隐变量条件 VAE**（SpatialCVAE）。关键设计：隐变量不是单个全局向量，而是空间图 `z ∈ R^{latent_dim × H/8 × W/8}`，因此能携带超分所需的高频细节。

训练时：
1. 从 HR 图像随机裁剪 `crop_size × crop_size` 图块。
2. 将 HR 图块下采样 2 倍，得到 LR。
3. 将 LR 双三次上采样回 HR 尺寸，作为条件输入 `c`。
4. 先验网络只读 `c`，输出 `p(z|c)` 的 `(mu_p, logvar_p)`；后验网络读 `[c, HR]`，输出 `q(z|c,HR)` 的 `(mu_q, logvar_q)`。
5. 从 `q(z|c,HR)` 采样 `z`，解码器读取 `[c, z]`，重建 HR。

损失函数：

`L = MSE(reconstruction, HR) + beta * KL(q(z|c,HR) || p(z|c))`

- 重建项用 **MSE**：PSNR 是 MSE 的单调变换，用 MSE 训练就是在直接优化 PSNR。
- KL 项对齐**条件先验 `p(z|c)`** 而不是 `N(0,I)`：这样测试时先验仍有意义，不会因 `z=0` 丢失信息。

测试时没有 HR 目标，因此：
- **确定性重建**（默认，PSNR 最高）：取先验均值 `z = mu_p(c)`。
- **生成式重建**：从 `p(z|c)` 采样 `z`，一张 LR 可生成多张不同的 HR（见 `demo_generate.py`）。

## 6. 目录结构

```text
vae_sr_x2_project/
├── data/
├── outputs/
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── utils.py
├── download_data.py
├── train.py
├── test.py
├── evaluate.py
├── requirements.txt
└── README.md
```
