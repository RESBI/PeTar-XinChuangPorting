# LA664（龙芯 3A6000）LSX/LASX 树力内核——基准与报告

本目录收录 PeTar 的 128 位 LSX 与 256 位 LASX 树力内核（`src/force_loongarch.hpp`）
移植报告。构建时通过以下选项选择向量宽度：

```shell
./configure --with-arch=loongarch --with-simd=lasx    # 或 lsx / no
```

也可以在 make 阶段覆盖，例如 `make LARCH_SIMD=lsx`（或 `make LARCH_SIMD=none`）；
可执行文件名跟随 configure 时的选择（`petar.mpi.omp.lasx`、`petar.mpi.omp.lsx`
或 `petar.mpi.omp`）。

目录内容：

- `REPORT.md`——完整报告（英文）：移植设计、四级数值验证、内核与端到端基准、
  扩展性、每步耗时构成、守恒性、生产运行与跨平台定位。
- `REPORT_CN.md`——报告中文版。
- `REPORT_RSQRT_CN.md` / `REPORT_RSQRT.md`——反比开方专项报告（中文 / 英文）：
  平台精确指令（默认，保留设计）与软件修正链（可选）的 A/B 实测、守恒性与设计取舍。
- `figs/`——标量 / LSX / LASX 构建的图。
- `figs_cross/`——跨平台对比与方法论图。
- `figs_rsqrt/`——反比开方专项报告插图。
- 本目录不存放原始测试数据；报告中的每张图与每张表都可由正文描述的测量复现。

3A6000（4 物理核 × 2 SMT，2.5 GHz 实测 2.50 GHz，GCC 15.3）关键结果：

| 负载 | 标量（F64） | LSX（128 位） | LASX（256 位） |
|---|---|---|---|
| EP-EP 内核 [ns/交互] | 8.99 | 2.80（3.2×） | **1.42（6.3×）** |
| 单核，N=2000 demo，512 步 [s] | 38.86 | 18.70（2.08×） | **14.10（2.76×）** |
| 最优布局（1 进程 × 4 OMP 线程）[s] | 12.49 | 6.04 | **4.53** |
| N=8000，4×1，每步 [ms] | 213.5 | 85.7 | **57.1** |
| 100 Myr 生产样例，8 OMP 线程 | 28 分 16 秒 | **10 分 54 秒** | 11 分 39 秒 |

数值验证（对标 `petar.simd.test`，默认构建）：最大力相对误差 3.6e-4（阈值 7e-3），
邻居计数逐粒子一致。反比开方默认采用平台精确指令 `xvfrsqrt.s`（误差 0）：
力误差比软件链改善 16%～21%，大 N 守恒性更好，而生产性能差异 ≤1%（详见
`REPORT_RSQRT_CN.md`）；`make LARCH_RSQRT=sw` 可切回软件修正链。
