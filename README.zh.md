# football-predictor

[English](README.md) · [中文](README.zh.md)

球队级足球比赛预测器。Dixon-Coles 双变量泊松模型 + Elo 修正先验，
基于免费公开数据源。Web UI 完整支持双语（中文 / English）。三种用法：

- **Web UI** — `python predict.py serve` → 打开 <http://localhost:8001>
  （单页应用：比赛预测 / 世界杯 / 联赛 / 关于；无需 build）
- **CLI** — `python predict.py predict "Arsenal" "Chelsea" --league 英超`
- **HTTP API** — `python predict.py serve` 在 :8001 暴露 FastAPI

当前预测模型停在球队层。球员表已经建好，付费数据源能往里塞阵容年龄、上场分钟、
首发、进球、助攻、评分，但首版生产模型故意停留在球队级 —— 因为校准更容易，
也不容易过拟合。

读 [ARCHITECTURE.md](ARCHITECTURE.md) 看项目边界、数据契约和基础规则。

## 数据覆盖

| 档位 | 数据 | 联赛 |
|------|------|------|
| **Tier 1** | 比赛结果 + ClubElo（+ 可选 xG） | 英超、英冠、西甲、西乙、意甲/意乙、德甲/德乙、法甲/法乙、荷甲、葡超、比甲 |
| **Tier 2** | 比赛结果 + ClubElo | 英甲 |
| **Tier 3** | 仅比赛结果或部分数据 | 沙特、J1、K1、中超、MLS、巴甲、阿甲、墨超、葡乙 |

Tier 3 联赛没有免费的 ClubElo 或 xG。预测仍能跑（用纯进球的 Dixon-Coles 拟合 +
内置 Elo 构造器），但校准比 Tier 1 差大约 5-8 个百分点。

API-Football 是可选的。设置 `FOOTBALL_API_KEY` 或 `API_FOOTBALL_KEY` 之后，
更新器可以拉取已配置联赛的赛程和当季球员数据。

**世界杯 2026** 用 eloratings.net 的国家队 Elo。完整 48 队蒙特卡洛
（12 组 × 4 + 8 个最佳第三 → R32 → 决赛）。

## 安装

```bash
cd ~/Documents/football-predictor
python3.11 -m venv .venv         # scipy>=1.14 需要 Python 3.10+
source .venv/bin/activate
pip install -r requirements.txt
python predict.py init-db
```

## 拉取数据

```bash
# 全部数据（5 年历史比赛 + 今日俱乐部 Elo + 国家队 Elo）：
python predict.py update

# 单个联赛。`--league` 同时接受英文 key 和中文别名：
python predict.py update --league 英超
python predict.py update --league premier_league
python predict.py update --league 沙特       # = saudi_pro
python predict.py update --league 中超       # = chinese_super_league

# 可选 API-Football 数据路径：
cp .env.example .env
# 编辑 FOOTBALL_API_KEY
python predict.py doctor --live
python predict.py update --league 沙特 --include-api-football
python predict.py update --league 中超 --include-api-football
python predict.py update --league 英超 --include-api-football --include-players

# xG 补充（看下面 "xG 数据源" 小节 — FBref 路径已挂，用 API-Football）：
python predict.py backfill-api-xg --league 英超 --season 2024 --limit 90

# 在依赖某个 API-Football league ID 之前先核对一下：
python predict.py api-football-leagues --country China --search "Super League"
```

数据落到 `data/football.sqlite3`。原始 CSV 缓存到 `data/cache/`。

HTTP API 同时暴露 `POST /update`（后台任务）、`GET /leagues`、`GET /coverage`、
`GET /doctor`、`POST /backtest`、
`GET /export/{matches|ratings|players|player_season_stats|update_state}`。

## xG 数据源

Dixon-Coles 模型可以吃单场 xG（`home_xg`、`away_xg`）。两条爬虫路径，
其中一条当前不可用：

| 数据源 | 命令 | 状态 |
|--------|------|------|
| API-Football | `predict.py backfill-api-xg --league <key> --season <year>` | **可用** |
| FBref | `predict.py update --league <key> --include-xg` | **被封** — FBref 对我们的 IP 返回 403 |

**推荐**用 `backfill-api-xg`。API-Football 的 `/fixtures/statistics` 端点
按球队返回 `expected_goals`。每场比赛 1 个 HTTP 请求，免费档 100 req/天的
配额下，一个 EPL 整赛季（~380 场）需要约 4 天慢慢补 —— 升级付费档可以一次
跑完。仓库里有个 launchd plist
（`deploy/com.aaronsun.football-predictor-claude.api-xg-backfill.plist`），
每天 CST 8:01 自动跑一批 90 场，实现免维护。

FBref 爬虫代码仍在 `scrape/fbref.py`，但 FBref 现在在网络层把我们 IP 封了
（所有请求 403，连首页都进不去；不是 UA 或限速问题）。重新启用需要住宅
代理服务或 headless-browser 绕过 —— 都不在 scope 内。要验证某联赛的 xG
是不是真的填上了，调 `/diagnostics/ablation` 看返回里的 `silent_features`
警告数组。

## 中文别名

`--league` 标志接受中文名替代英文 key。完整列表在 `scrape/registry.py::EXTRA_ALIASES`。

| 别名         | League key             |
|--------------|------------------------|
| `英超`       | `premier_league`       |
| `西甲`       | `la_liga`              |
| `德甲`       | `bundesliga`           |
| `意甲`       | `serie_a`              |
| `法甲`       | `ligue_1`              |
| `中超`       | `chinese_super_league` |
| `沙特`       | `saudi_pro`            |
| `世界杯`     | `world_cup`            |

Web UI 的联赛下拉框会根据语言切换自动显示中英文。

## 预测一场比赛

```bash
# 俱乐部比赛。联赛是可选的，但能改善拟合。
python predict.py predict "Arsenal" "Chelsea" --league 英超

# 中立场地，淘汰赛（输出晋级概率而不是单纯的平局概率）。
python predict.py predict "Real Madrid" "Bayern Munich" \
  --neutral-site --stage "quarter-final"

# 国家队比赛（自动用国家队 Elo）。
python predict.py predict "Brazil" "Argentina" --league 世界杯
```

CLI 输出：包含 `probabilities {home_win, draw, away_win}`、`expected_goals`、
`most_likely_scores`、完整 `score_matrix` 以及训练元数据的 JSON。

当 ClubElo / 国家队 Elo 缺失时，服务会从已经在 SQLite 里的历史比赛
构造一个无前瞻泄漏的内置 Elo。

## 回测

```bash
# 单个联赛的滚动回测。
python predict.py backtest --league 英超 --min-train-matches 120 --refit-every 5
python predict.py backtest --league 英超 --include-predictions  # verbose 输出

# 看当前哪些联赛有可用数据。
python predict.py coverage
python predict.py coverage --only-empty
python predict.py doctor
python predict.py doctor --live

# 把原始表导出来做 notebook 分析。
python predict.py export matches -o data/exports/matches.csv
python predict.py export player_season_stats -o data/exports/player_season_stats.csv
```

回测输出：1X2 准确率、multi-class Brier score、multi-class log loss，
以及实际的主胜/平/客胜分布。

## 世界杯 2026

```bash
# 合成 top-48 抽签（确定性）。
python worldcup.py --n-sims 20000

# 真实抽签公布后用真实分组。draw.json 示例：
# { "A": ["United States", "Mexico", "Egypt", "Iran"], "B": [...] }
python worldcup.py --groups data/wc2026_draw.json
```

输出每支球队的 R16 / 8 强 / 半决赛 / 决赛 / 夺冠概率。
名字需要匹配 `eloratings.net` 的拼写（"United States"、"South Korea"）。

## 跑成服务

```bash
python predict.py serve --port 8001
```

打开 <http://localhost:8001> 看 Web UI，或者直接打 JSON 接口：

| Method | Path                          | 用途                                       |
|--------|-------------------------------|--------------------------------------------|
| GET    | `/health`                     | 存活检查 + 定时器状态                      |
| GET    | `/stats`                      | 比赛 / 球队 / Elo 计数                     |
| GET    | `/coverage`                   | 各联赛数据覆盖报告                         |
| GET    | `/doctor`                     | 数据源就绪检查 + 下一步命令建议            |
| GET    | `/leagues`                    | 联赛注册表                                 |
| GET    | `/teams?league=<key>`         | 去重球队列表（前端自动补全用）             |
| GET    | `/recent?league=<key>&limit=` | 最近比赛                                   |
| POST   | `/predict`                    | 单场 Dixon-Coles + Elo 预测                |
| GET    | `/worldcup/forecast?n_sims=`  | 世界杯 2026 蒙特卡洛（带缓存）             |
| POST   | `/backtest`                   | 滚动起点回测                               |
| POST   | `/update`                     | 触发增量数据更新（后台任务）               |
| GET    | `/export/{table}`             | SQLite 表的 CSV 导出                       |
| GET    | `/api-football/leagues`       | 查询 API-Football 的 league ID             |

服务会同时启动一个 APScheduler 任务，每天 03:30 Asia/Shanghai 跑 `update_all`。
用 `FOOTBALL_PREDICTOR_ENABLE_SCHEDULER=false` 关掉。
设 `FOOTBALL_PREDICTOR_DAILY_API_FOOTBALL=true` 让每日任务也拉 API-Football 赛程
（除非你的配额充裕，建议保持关闭）。
`FOOTBALL_PREDICTOR_DAILY_FBREF_XG=true` 曾经用于每日 FBref xG 补充，
但 FBref 现在已经封了我们的 IP —— 看上面"xG 数据源"小节。改用
`predict.py backfill-api-xg`。

加新模型特性之前，`python predict.py doctor` 要保持绿色。它会检查 SQLite schema、
配置的数据源、覆盖缺口以及下一步要跑的命令。

## 测试

```bash
python -m pytest
```

测试覆盖：中文联赛别名、SQLite upsert/去重、Dixon-Coles 概率归一化、xG 混合、
FBref xG merge 行为、内置 Elo 生成、滚动回测指标、API-Football 球员数据归一化。

## 它做不了的事

- **预测球员个人表现。** 存储已经建好了，但当前模型还没用球员特征。
- **自动处理伤停、停赛、换帅。** 强度评分滞后 4-6 场。
- **稳定打赢盘口收盘价。** 一个好的 Dixon-Coles 模型在顶级联赛对三选一的准确率
  在 52-55% —— 跟有效市场差不多。用概率找 value，不是用它求确定性。

## 文件

```
config/leagues.yaml         联赛注册表：代码、档位、中文别名。
data/database.py            SQLAlchemy ORM（matches、ratings、players、update state）。
models/elo.py               内置无前瞻泄漏的 Elo 构造器。
models/backtest.py          滚动 1X2 回测指标。
models/dixon_coles.py       Dixon-Coles + Elo 调整的双变量泊松。
scrape/registry.py          LeagueRegistry + EXTRA_ALIASES（中文别名）。
scrape/update.py            IncrementalUpdater 编排器。
scrape/api_football.py      可选 API-Football 赛程 + 球员数据。
scrape/clubelo.py           ClubElo 每日快照。
scrape/football_data.py     历史结果（football-data.co.uk）。
scrape/eloratings.py        国家队 Elo。
scrape/fbref.py             可选 FBref xG 增强（带限速）。
predict.py                  FastAPI app + typer CLI（update / predict / serve / export）。
worldcup.py                 48 队蒙特卡洛模拟器。
```

## 礼貌

- ClubElo 按日缓存，同一天绝不重复抓。
- football-data.co.uk 赛季缓存：首次抓取后只重拉当前进行中的赛季。
- FBref 限速很严（10 req/min）。爬虫每次请求之间 sleep 6.5s，被封就静默跳过。

不要在循环里跑 `update`。一天一次足够了。
