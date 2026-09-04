# 02 · 模型架构(Qwen3.5-27B)

> 来源:`model_config/config.json` + `model_config/README.md`。
> 本文件为冷数据档案,涉及算子/模型判断时按需读取。

## 2.1 架构表

| 项 | 值 |
|---|---|
| 总层数 | 64 |
| 层布局 | `16 × (3 × (GDN → FFN) + 1 × (FullAttn → FFN))` ⇒ **48 层 GDN + 16 层 FullAttn** |
| FFN | **DENSE**(intermediate=17408,silu),无专家。README 第26行"sparse MoE"是系列通用营销文案,27B 这款无 MoE |
| GDN(linear_attention) | QK heads=16 / V heads=48,head_dim=128,conv_kernel=4 |
| FullAttn | Q heads=24 / KV heads=4,head_dim=256,partial_rotary=0.25(rotary_dim=64) |
| hidden=5120,vocab=248320,mrope(section [11,11,10]),max_pos=262144 |
| 视觉(ViT) | depth=27,hidden=1152,heads=16(head_dim=72,pad 到 96) |

## 2.2 关键结论

**FFN 是稠密的(非 MoE)。** config.json 无 `num_experts`/`moe_intermediate_size`/`router` 字段,`intermediate_size=17408` + `hidden_act=silu` 是标准稠密 FFN。README 第 26 行 "sparse Mixture-of-Experts" 是系列通用营销文案,27B 这款无 MoE。这直接影响 DeepGEMM 的可用性——见 `01_constraints_env.md` §1.3。

**注意力是混合的(非纯全注意力)。** `layer_types` 为 64 元素数组 = `[linear_attention ×3, full_attention ×1] × 16`,即 **48 层 Gated DeltaNet(线性注意力)+ 16 层全注意力**(`full_attention_interval=4`)。线性注意力层用 GDN(QK heads=16/V heads=48/head_dim=128),全注意力层用 GQA(Q heads=24/KV heads=4/head_dim=256/partial_rotary=0.25)。两种层共享同一套稠密 FFN。对 decode 而言:48 层走线性注意力递推路径(状态更新 + 投影 GEMM),仅 16 层走 KV-cache 全注意力——这是 P2-decode 算子拆解的架构前提(见 `09_cpu_sched_overhead_design.md` / `10_gdn_gemm_design.md`)。
