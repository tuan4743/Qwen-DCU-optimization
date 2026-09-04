# DCU 容器访问链路(MCP ssh-sessions)

> 本文档记录唯一**已验证可稳定复用**的 worker 容器访问路径。
> 旧路径(docker exec 进容器、本地 `ssh PRA26-worker` 直连 173.0.8.2)已被组委会修复/权限收紧,不可靠,详见文末"已废弃路径"。

---

## 0. 适用前提

- 你用的 worker 容器 hostname=`worker-0`,容器内 IP 随作业变化(曾用 `173.0.8.2`、`173.0.58.3`、`173.0.59.7`……),用户 `root`,sshd 在 22 端口(`SSH-2.0-OpenSSH_8.9p1`)。**每次开工从 `squeue` 取节点名 + 进容器后 `hostname -I` 确认当前容器 IP**(见 §2.6)。下文示例统一用 `173.0.59.7`,**实际用时务必替换成当次取到的值**。
- 计算节点名随作业变化(曾用 `e03r1n07`、`h14r1n09`……),**每次开工必须重取**(容器随时超时关机,重启后 PID/作业/节点/容器名全变)。
- 通过 Claude Code 的 **MCP `ssh-sessions` 工具**操作,而非本地 `ssh` 命令。原因:本地 ssh 直连 173.0.8.2 三把密钥都 `Permission denied`,而 MCP 链从计算节点嵌套 `ssh root@173.0.8.2` 可成功(靠计算节点转发,无需显式密钥)。

---

## 1. 三条会话

| MCP session 名 | 主机 | 用户 | 角色 |
|---|---|---|---|
| `login` | `zzeshell.scnet.cn:65032` | `xdzs2026_c150` | 登录跳板 |
| `compute` | `h14r1n09:22`(节点名会变) | `xdzs2026_c150` | 计算节点 |
| `worker` | `<容器IP>:22`(随作业变,本次 `173.0.59.7`) | `root` | **你的 vllm 容器(worker-0)** |

密钥:`~/.ssh/InstanceKey.txt`(登录节点密钥,对 login/compute 生效;worker 段靠 compute 转发,不需显式密钥)。

---

## 2. 标准连接流程(每次开工按此走)

### 2.1 看现有会话
```
ssh_sessions
```
连着的复用,断的 reconnect。

### 2.2 连/重连 login(基础跳板)
```
ssh_reconnect login
```
若首次:
```
ssh_connect name=login host=zzeshell.scnet.cn port=65032 username=xdzs2026_c150 key_path=~/.ssh/InstanceKey.txt
```

### 2.3 取当前作业的计算节点名(关键:节点名会变)
在 login 上跑:
```
ssh_exec login "/opt/gridview/slurm/bin/squeue -u xdzs2026_c150 -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R'"
```
- `squeue` 在登录节点 non-interactive shell 的 PATH 不含,**必须用全路径** `/opt/gridview/slurm/bin/squeue`。
- 输出最后一列 `%R` 形如 `h14r1n09`,即计算节点名。记下它,后续命令里替换。

### 2.4 进 worker 容器(核心一步:嵌套 ssh)
**worker 段不必单独 ssh_connect**,直接从 login 嵌套两层 ssh 进容器:
```
ssh_exec login "ssh <节点名> \"ssh root@173.0.8.2 'bash -lc \\\"<容器内命令>\\\"'\""
```

例(取 hostname 确认进对容器):
```
ssh_exec login "ssh h14r1n09 \"ssh root@173.0.8.2 'bash -lc \\\"hostname\\\"'\""
```
期望输出:`worker-0`。

> 也可以 `ssh_connect name=compute host=<节点名> username=xdzs2026_c150 key_path=~/.ssh/InstanceKey.txt` + `ssh_connect name=worker host=173.0.8.2 username=root`,再 `ssh_shell session=worker "..."`。但嵌套 `ssh_exec` 一行更省事,推荐。

### 2.5 命令必须用 `bash -lc` 包裹
非交互 ssh 进容器**不加载 `~/.bashrc`**(第 6 行 `[ -z "$PS1" ] && return`)→ 不设 `LD_LIBRARY_PATH` → `import vllm` 报 `ImportError: libgalaxyhip.so.5: cannot open shared object file`。**所有容器内命令一律 `bash -lc '...'`**,让登录 shell 加载 DCU 环境。

### 2.6 取当次容器 IP(每次开工必做)
§2.3 的 `squeue` 只给计算节点名,**不给容器 IP**。容器 IP 每次作业都变,进容器后用 base64 法(见 §3.1)一次取齐:
```
B64=$(echo -n 'echo "节点:$(hostname)"; echo "容器IP:$(hostname -I | awk "{print \$1}")"; echo "vllm进程:"; pgrep -fa "vllm|EngineCore" | grep -v pgrep | head -3' | base64 -w0)
ssh_exec login "ssh <节点名> \"ssh root@<猜的容器IP> \\\"echo $B64 | base64 -d | bash -l\\\"\""
```
> 首次不知道容器 IP 时,先从 login `ssh <节点名>` 进计算节点,再 `docker ps` 或 `cat /proc/*/cgroup` 找 worker-0 容器的 veth IP;或直接 `ssh <节点名> "hostname; hostname -I"` 看计算节点上是否有容器网桥信息。拿到后填进后续命令。**一旦本次会话取到 IP,整段会话内固定用它**(容器不重启就不变)。

---

## 3. 引号转义速查(嵌套 ssh 的痛点)

最外层是 `ssh_exec` 的 command 字符串,里面三层 ssh,引号层层嵌套。规则:

- 最外层用双引号包整条命令。
- login 上的 `ssh <节点名>` 后用双引号。
- `ssh root@<容器IP>` 后用单引号。
- 容器内 `bash -lc` 后用转义双引号 `\\\"`。

模板(把 `<CMD>` 换成容器内命令):
```
ssh h14r1n07 "ssh root@173.0.59.7 'bash -lc \"<CMD>\"'"
```
作为 `ssh_exec login` 的 command:
```
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 'bash -lc \\\"<CMD>\\\"'\""
```

容器内命令若含双引号,再继续转义。

### 3.1 引号地狱的根治:base64 传命令(强烈推荐)

一旦容器内命令出现**多行 / heredoc / 单双引号混用 / Python 三引号**,上面层层转义必然崩(`syntax error`、`unmatched quote`、`command not found` 把命令名切碎)。**别和转义搏斗,直接 base64:**

思路:本地把脚本 base64 编码 → ssh 链路原样透传 → 容器内 `base64 -d | bash -l` 解码执行。base64 串只含 `[A-Za-z0-9+/=]`,**对任何 shell 引号都免疫**,内层脚本想怎么写就怎么写(单引号、双引号、heredoc、`$()`、反斜杠都照常)。

模板(MCP `ssh_exec login`,把脚本写进本地变量):
```
B64=$(echo -n 'set -e
echo "===== ps vllm ====="
ps -ef | grep -E "vllm|EngineCore" | grep -v grep | head
# 随便用引号,不用转义
curl -sS -m 5 --noproxy "*" http://127.0.0.1:8001/health
' | base64 -w0)
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 \\\"echo $B64 | base64 -d | bash -l\\\"\""
```

要点:
- `echo -n '脚本' | base64 -w0`:`-n` 去尾换行,`-w0` 不折行(折行会被 shell 当多条)。
- 最内层是 `echo $B64 | base64 -d | bash -l`:**用 `bash -l` 不是 `bash -lc`**,因为是管道喂 stdin,且 `-l` 加载 DCU 环境(等价 §2.5 的 `bash -lc` 效果)。
- `$B64` 在本地展开成纯 base64 串后,整条 ssh 命令里就只剩一层双引号、一层单引号、base64 串,**零嵌套引号**。
- 脚本里 `set -e` 让任何命令失败立即退出,exit code 能透传回 MCP(`ssh_exec` 返回 EXIT=N)。
- 这套写法**全程不用 `ssh_shell session=worker` 逐条跑**,一条 MCP 调用就能跑完整段多行脚本,实测稳定。

> 只有单行、无引号歧义的简单命令才用 §3 的逐层转义模板。**多行/含引号一律 base64**,别犹豫。

---

## 4. 文件读写(本地 ↔ 容器)

容器内文件操作分两种:小文件读内容、大文件本地分析;以及本地改文件推回容器。

### 4.1 读容器文件内容(小文件 / 片段)

**优先用 base64 把目标内容打回来**,不要靠 `cat` 经三层 ssh 回传(`cat` 的输出会被 ssh 的 stderr/warning 污染,且多行难解析):

```
# 读整个小文件
B64=$(echo -n 'cat /usr/local/lib/python3.10/dist-packages/gemm_probe.py' | base64 -w0)
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 \\\"echo $B64 | base64 -d | bash -l\\\"\""
```
文件内容直接出现在 MCP 返回的 STDOUT 里。

**读指定行段**(核查源码桩最常用):
```
B64=$(echo -n 'sed -n "630,705p" /usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_next.py' | base64 -w0)
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 \\\"echo $B64 | base64 -d | bash -l\\\"\""
```

**只 grep 关键字计数**(核查桩是否在位):
```
B64=$(echo -n 'grep -c "from gemm_probe import" /usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_next.py' | base64 -w0)
...
```

### 4.2 大文件拉回本地分析(trace、日志)

trace `.json.gz` 动辄几十 MB,经三层 ssh `cat` 回传会超时/截断。**用 sftp 经 login→compute 链路下载,或先在容器内解压后只回传分析结果:**

推荐做法 —— **容器内就地分析,只回传 JSON/数字结果**(本次会话验证过,401MB 的 trace 在容器内用 python3 分析,只把统计 Counter 回传):
```
B64=$(echo -n 'set +e
cd /public/home/xdzs2026_c150/zya/profile_traces
zcat rank0.XXXX.pt.trace.json.gz > /tmp/trace.json
python3 - <<PY
import json
from collections import Counter
d=json.load(open("/tmp/trace.json"))
evs=d["traceEvents"]
print(Counter(e.get("cat","") for e in evs).most_common(10))
print("GEMM_PROBE count:", sum(1 for e in evs if "GEMM_PROBE" in e.get("name","")))
PY
' | base64 -w0)
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 \\\"echo $B64 | base64 -d | bash -l\\\"\""
```
要点:
- 容器内 `python3 - <<PY ... PY` heredoc 任意写,base64 透传无压力。
- **坑**:`python3 - <<PY` 里的 f-string 不能含 `\"` 转义(Python 3.10 f-string expression 不允许反斜杠),报 `SyntaxError: f-string expression part cannot include a backslash`。解决:f-string 里别写字典下标 `v["x"]`,改用 `v.get("x")` 或先把值取到变量;或干脆不用 f-string,用 `"...%s..." % (...)` 格式化。
- 容器内 `/tmp` 是 overlay 盘,放几百 MB 临时文件没问题,分析完即弃。

若必须把文件拉回本地(如需本地工具处理):用 MCP `ssh_download` 经 login session 下,路径是 login 上看到的路径(容器路径在 login 不可见,需先 `scp` 到 login 节点再下,通常不值得,优先容器内分析)。

### 4.3 本地改文件推回容器

vllm 源码改动的标准流程是**改 zya 源码树 → 编译 wheel → pip 安装覆盖 site-packages**(见任务 10 §5),不要直接手改 site-packages 单文件。但**辅助脚本(非 vllm 包内容,wheel 不含的)**需手动 cp 到 site-packages,本次会话的三个探针文件就是:

```
B64=$(echo -n 'set -e
# 从源码树 cp 到 site-packages 根 + /usr/local 双保险
# (容器 python3.10 sys.path[0]=/usr/local, 顶层 import 要放这两处)
for f in gemm_probe.py fill_alloc_probe.py fill_capture_hook.py; do
  cp /public/home/xdzs2026_c150/zya/$f /usr/local/lib/python3.10/dist-packages/$f
  cp /public/home/xdzs2026_c150/zya/$f /usr/local/$f
done
# 验证 import
python3 -c "import gemm_probe, fill_alloc_probe, fill_capture_hook; print(gemm_probe.__file__)"
' | base64 -w0)
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 \\\"echo $B64 | base64 -d | bash -l\\\"\""
```

关键经验:
- **顶层 `import xxx`(如 qwen3_next.py 里 `from gemm_probe import`)**,模块文件必须放 `sys.path` 上的顶层目录 —— 容器里是 `/usr/local`(sys.path[0])和 `/usr/local/lib/python3.10/dist-packages`,**不能塞 vllm/model_executor/models/ 子目录**(那不是顶层,import 不到)。
- 改完文件若该文件已被运行中的 vllm 进程加载,需**重启 vllm**才生效;若涉及编译产物还要清缓存(见 §4.4)。
- 多文件 cp 用 `for f in ...; do cp ...; done` 一把梭,base64 透传无引号问题。

### 4.4 改源码后必清的缓存(否则装的旧版)

vllm + inductor 有三层编译缓存,改源码重新装 wheel 后若不清,vllm 仍用旧编译产物(症状:trace 里看不到新桩、行为不变):

```
# 1. 停 vllm(否则 triton_cache 内 .so 被 EngineCore 占用, rm 报 .nfs Device busy)
pkill -9 -f start_vllm.sh; pkill -9 -f EngineCore; pkill -9 -f VLLM
# 确认无残留
pgrep -fa vllm | grep -v pgrep

# 2. 清三处缓存
rm -rf /public/home/xdzs2026_c150/zya/triton_cache
rm -rf /public/home/xdzs2026_c150/zya/vllm_cache/torch_compile_cache
rm -rf /tmp/torchinductor_root
# 3. 清 site-packages 的 vllm __pycache__(防止旧 .pyc 生效)
find /usr/local/lib/python3.10/dist-packages/vllm -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 4. 重启 vllm(见 §5)
```

**坑**:`rm -rf triton_cache` 在 EngineCore 还活着时报 `.nfs000000xxxx: Device or resource busy` —— 因为进程仍 mmap 着里面的 .so。必须先 `pkill` 确认进程死透再 rm。

### 4.5 核实"装对了"的标准核查

改源码 + 编译安装 + 清缓存 + 重启后,用这套确认 site-packages 里的 vllm 确实是新码(本次会话验证流程):
```
B64=$(echo -n 'F=/usr/local/lib/python3.10/dist-packages/vllm/model_executor/models/qwen3_next.py
echo "桩数:"; grep -c "from gemm_probe import" $F
echo "disable:"; grep -n "torch.compiler.disable" $F
echo "桩位置:"; grep -n "label_in_proj_qkvz\|label_in_proj_ba\|label_out_proj" $F
echo "辅助文件:"; ls -la /usr/local/lib/python3.10/dist-packages/gemm_probe.py /usr/local/gemm_probe.py
' | base64 -w0)
ssh_exec login "ssh h14r1n07 \"ssh root@173.0.59.7 \\\"echo $B64 | base64 -d | bash -l\\\"\""
```

---

## 5. 常用核查命令(进 worker-0 后)

```
hostname                                    # 期望 worker-0
df -h /                                     # overlay 盘,关注 Avail(曾满到 0)
pgrep -fa 'vllm' | grep -v pgrep            # vllm 是否在跑
ls /public/home/xdzs2026_c150/zya/triton_autotune_cache/*.autotune.json 2>/dev/null | wc -l  # 候选1 是否执行过
grep -c fill_alloc_probe /usr/local/lib/python3.10/dist-packages/vllm/v1/worker/gpu_model_runner.py  # probe 桩(期望 0)
sed -n '718,722p' /usr/local/lib/python3.10/dist-packages/triton/backends/amd/driver.py   # cache_size 期望 256*1024*1024
```

启动 vllm(设 TMPDIR 防 /tmp 写满):
```
mkdir -p /public/home/xdzs2026_c150/zya/tmp
export TMPDIR=/public/home/xdzs2026_c150/zya/tmp
cd /public/home/xdzs2026_c150/zya && nohup bash start_vllm.sh > logs/start.log 2>&1 &
```
> **日志文件名坑**:`start_vllm.sh` 内部用 `tee -a "$LOG_FILE"` 把 stdout 再写到 `logs/vllm_start.log`(脚本里 `LOG_FILE=logs/vllm_start.log`)。所以 vllm 真正的启动日志看 **`logs/vllm_start.log`**,不是上面 nohup 重定向的 `logs/start.log`(后者只有 nohup 层的少量输出)。排查启动卡住/报错一律 `tail -f logs/vllm_start.log`。

### 5.1 访问 8001 端口必须 `--noproxy "*"`(Squid 代理坑)
容器内有 Squid 代理环境变量:
```
http_proxy=http://preset:6e298f07@10.13.17.166:3128
```
**直接 `curl http://127.0.0.1:8001/health` 会被 Squid 拦截**,返回代理错误页(`Connection refused` / 502),完全不是 vllm 的响应。**容器内所有访问本地 8001 的 curl 都要加 `--noproxy "*"`**:
```
# 健康检查
curl -sS -m 5 --noproxy "*" http://127.0.0.1:8001/health
# 发 decode 请求
curl -sS -m 120 --noproxy "*" http://127.0.0.1:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-27B","prompt":"Hello","max_tokens":16,"temperature":0}'
# profiler 接口(start_profile / stop_profile 同样要 --noproxy "*")
curl -sS -m 10 --noproxy "*" -X POST http://127.0.0.1:8001/start_profile
curl -sS -m 180 --noproxy "*" -X POST http://127.0.0.1:8001/stop_profile
```
> `stop_profile` 接口本身慢(要等 trace flush 落盘),`-m 180` 给足时间,超时不代表失败 —— 单独去 `profile_traces/` 目录看新 trace 文件是否生成确认成功(见 §4.2 的容器内分析法)。

---

## 5. 已废弃路径(别再用)

- **本地 `ssh PRA26-worker`**:`~/.ssh/config` 里 ProxyJump 经 PRA26-compute(用 `squeue` 全路径解析节点名)→ `root@173.0.8.2`。实测三把本地密钥全 `Permission denied`(InstanceKey 是登录节点密钥,非 worker root 的)。**直连不可用**,改走 MCP 嵌套 ssh(§2.4)。本地 config 里 PRA26-worker/PRA26-compute 条目可保留作记录,但不要直接 `ssh PRA26-worker`。
- **docker exec 进容器**:组委会已修复,docker exec 不再能进用户的 worker 容器(或权限被收)。统一用 §2.4 嵌套 ssh。
- **`UserKnownHostsFile=none` / `GlobalKnownHostsFile=none`**:macOS 空设备写法,Windows OpenSSH 把 `none`/`NUL` 当文件路径 → `Host key verification failed`。改用默认 known_hosts + `StrictHostKeyChecking no`。
- host key 变更警告:`ssh-keygen -R <host>` 清掉即可(h14r1n09/worker 容器每次重建 key 都变)。
