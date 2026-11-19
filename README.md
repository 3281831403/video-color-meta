# video-color-meta

## 题面

把 HDR 素材批量转 SDR 后，颜色整体发灰偏绿：① `bt709` 与 `bt.709` 被当成两种，兼容判断失败；② 兼容判断忽略 transfer，HDR 素材被当 SDR 直接拷贝；③ 矩阵按 BT.601 系数硬编码，BT.709 转换偏色；④ limited/full 不区分，暗部被抬亮 16/255；⑤ tone map 不钳制，高光溢出后回绕；⑥ 色域映射在 RGB 空间线性插值，超出目标色域；⑦ 转换顺序先 matrix 后 range，整幅偏 6/255；⑧ 批量遇未知枚举只跳过，输出里留下混色帧；⑨ 未知 transfer 静默当成 sRGB；⑩ 浮点输出不量化，8 bit 管道里出现 16 位值。color/metadata.py 是核心，tests/test_metadata.py 15个红。要修的是：normalize 的别名归一与未知值报错、compatible 的 primaries/transfer/matrix 三要素判断、BT.601/BT.709 矩阵、limited/full 范围、tone map 的三种曲线与上限、XYZ 色域映射与钳制、convert_pixel 的 range→matrix→gamut 串联、convert_batch 的原子回滚与顺序、未知 transfer 报错、8 bit 量化。

## 运行方式

```powershell
cd D:\gsb-20261119\video-color-meta
python -m pytest tests -q
```

初始状态输出恰好为 `15 failed`，没有 collection/import/syntax error；15 条失败都来自 `color/metadata.py` 中 8 个桩的 `NotImplementedError`。修复核心模块后测试应转绿。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_metadata.py` 是评测基准，**不可修改**；只允许修改 `color/metadata.py`。实现只使用 Python 标准库（`math`、`json`）与 pytest，离线运行，不使用随机数。
