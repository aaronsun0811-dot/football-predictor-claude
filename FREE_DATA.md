# 临时用免费数据测试模型

Tier 3 联赛（中超/巴甲/阿甲/沙特/J1/K1/MLS/墨超）和洲际杯赛（亚冠、解放者、世俱杯等）
目前没数据，模型预测会报"No domestic matches"。这是我们已经探明的免费源 + 怎么用。

## 三个免费选项一览

| 源 | 注册 | 限速 | 实际免费档覆盖 | 推荐度 |
|---|---|---|---|---|
| **football-data.org** ⭐ | 邮箱注册，30 秒 | 10 次/分钟 | 12 个 TIER_ONE：**巴甲、欧冠、欧洲杯、世界杯**，及 5 大欧洲联赛（已有） | 巴甲必备 |
| **API-Football** ⭐⭐ | RapidAPI 注册 | 100 次/天（免费档） | **几乎所有联赛**：中超、沙特、J1、K1、MLS、墨超、亚冠、解放者、CAF、CONCACAF | 中超 / 亚洲 / 美洲必备 |
| **TheSportsDB** | 无需 key | ~每秒一次 | 只能拉**每季前 15 场**（鸡肋） | 不推荐 |

—— 这台机器从外网测了 5 个源，**football-data.org / TheSportsDB / OpenLigaDB / Understat 可直连**，
API-Football（v3.football.api-sports.io）**403**（需要 RapidAPI 头）。
ClubElo 和 FBref **网络超时**（可能是 IP 段被封）。

## 推荐流程：football-data.org（5 分钟）

### 1. 注册拿 key

打开 <https://www.football-data.org/client/register>，
填邮箱 + 名字 + "use case"（写 hobby 即可），秒级发邮件到你的邮箱。

### 2. 设置环境变量

```bash
export FOOTBALL_DATA_ORG_KEY=your_actual_key_here
```

或写进 `.env` 持久化：

```bash
echo "FOOTBALL_DATA_ORG_KEY=your_key" >> ~/Documents/football-predictor-claude/.env
```

### 3. 直接拉数据（Python 一行调用）

```bash
cd ~/Documents/football-predictor-claude
source .venv/bin/activate
python -c "
from pathlib import Path
from data.database import Database
from scrape.football_data_org import upsert_matches_into_db, LEAGUE_KEY_TO_CODE

db = Database('data/football.sqlite3')
# 拉中超 + 巴甲 + 解放者 + 欧冠 4 个 — 大概 30 秒（含限速等待）
for league_key in ['chinese_super_league', 'brasileirao', 'copa_libertadores', 'champions_league']:
    code = LEAGUE_KEY_TO_CODE[league_key]
    n = upsert_matches_into_db(db, code, league_key,
                                cache_dir=Path('data/cache/football-data-org'),
                                seasons=[2024, 2025])
    print(f'  {league_key}: {n} matches inserted')
"
```

### 4. 立刻预测

```bash
# 中超对阵
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"home_team":"Shanghai Port","away_team":"Beijing Guoan","league":"中超"}'

# 真实欧冠（不再走跨联赛拟合）
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"home_team":"Real Madrid","away_team":"Bayern Munich","league":"欧冠"}'

# 解放者杯
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"home_team":"Flamengo","away_team":"River Plate","league":"解放者杯"}'
```

## football-data.org 免费档（TIER_ONE）实际覆盖

**真正能免费拿的（TIER_ONE）**：

```
code  league_key            实际意义
─────────────────────────────────────────
BSA   brasileirao           ← 巴甲，本项目无替代源
CL    champions_league      ← 真实 UCL，替代跨联赛拟合
EC    euro                  ← 欧洲杯
WC    world_cup             ← 世界杯
PL    premier_league        （football-data.co.uk 已覆盖，可选）
PD    la_liga                同上
BL1   bundesliga             同上
FL1   ligue_1                同上
SA    serie_a                同上
PPL   primeira               同上
DED   eredivisie             同上
ELC   championship           同上
```

**需付费档（不要被代码列误导）**：

```
code  league_key            实际档位
─────────────────────────────────────────
EL    europa_league         TIER_TWO（€20+/月）
CLI   copa_libertadores     TIER_FOUR（€50+/月）
CSL   chinese_super_league  TIER_FOUR（€50+/月）⚠️ 之前误传
```

也就是说，**注册免费 key 实际能补的最关键空白是巴甲**（其它都已有或仍需付费）。
中超、解放者杯还是要走 API-Football 路径。

完整映射在 `scrape/football_data_org.py` 的 `LEAGUE_KEY_TO_CODE`。

## 免费档拿不到的联赛

football-data.org 免费档**没有**：
- 沙特职业联赛（Saudi Pro）
- J1 / K1（日韩顶级）
- MLS / Liga MX（北美）
- 阿甲 Liga Profesional（有公开 code 但是 TIER_THREE，需付费档）
- 葡甲、英甲、所有第二档
- 亚冠 / CAF CL / CONCACAF CC / 大洋洲 / 南美超级杯 / 欧超杯 / 世俱杯

要补这些只能：
- 付 **football-data.org TIER_TWO/THREE 订阅**（€20-50/月）
- 注册 **API-Football** 免费档（100 次/天，够测试不够长期跑）
- 找其它源（unofficial scrape）

## API-Football 路径（中超 + 亚冠 + 解放者 必备）

football-data.org 免费档拿不到中超 / 亚冠 / 解放者杯（都是 TIER_FOUR）。
API-Football 100 req/day 免费档**全有**，本项目已经接好了。

### 1. 注册免费 key

去 <https://dashboard.api-football.com/register>，邮箱秒发 key。

> 注：api-football.com 和 api-sports.io 是同一家公司，注册后看 dashboard，
> "x-apisports-key" 那个就是要用的 key。

### 2. 设置环境变量

```bash
export API_FOOTBALL_KEY=your_actual_key_here
# 或 FOOTBALL_API_KEY=your_key（两个名字都认）
```

### 3. 检查配额（这一步只花 1 个请求）

```bash
cd ~/Documents/football-predictor-claude && source .venv/bin/activate
python -c "from scrape.api_football import quota_status; print(quota_status())"
# 期望输出: {'plan': 'Free', 'requests_today': 1, 'requests_limit_day': 100}
```

### 4. 拉中超 + 亚冠 + 解放者杯（3 个联赛 × 3 季 = 9 个请求）

```bash
python -c "
from pathlib import Path
from data.database import Database
from scrape.api_football import upsert_matches_into_db

db = Database('data/football.sqlite3')
cache = Path('data/cache/api-football')

for league_key in ['chinese_super_league', 'afc_champions_league', 'copa_libertadores']:
    n = upsert_matches_into_db(db, league_key, cache_dir=cache, years_back=3)
    print(f'  {league_key:25s} → {n} matches inserted')
"
```

预计耗时约 1 分钟（含限速等待 — 10 req/min）。耗 9 个请求，剩余 91 个/天。

### 5. 立刻预测

```bash
# 中超对阵 — 注意球队名用 API-Football 的标准名（"Shanghai Port FC", "Beijing Guoan F.C." 等）
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"home_team":"Shanghai Port FC","away_team":"Beijing Guoan F.C.","league":"中超"}'

# 亚冠
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"home_team":"Al-Hilal Saudi FC","away_team":"Yokohama F. Marinos","league":"亚冠"}'

# 解放者杯
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"home_team":"Flamengo","away_team":"River Plate","league":"解放者杯"}'
```

### 6. 想顺便补全所有亚洲/美洲缺口？

加几个就好（一共 18 个请求，配额够）：

```bash
python -c "
from pathlib import Path
from data.database import Database
from scrape.api_football import upsert_matches_into_db
db = Database('data/football.sqlite3')
cache = Path('data/cache/api-football')
# 中超 + 亚冠 + 亚冠二级 + 沙特 + J1 + K1 + MLS + 墨超 + 阿甲 + 解放者 + 南美杯
for k in ['chinese_super_league','afc_champions_league','afc_champions_league_two',
          'saudi_pro','j1','k1','mls','liga_mx','primera_argentina',
          'copa_libertadores','copa_sudamericana']:
    n = upsert_matches_into_db(db, k, cache_dir=cache, years_back=2)
    print(f'  {k:25s} → {n} matches')
"
```

22 个请求 × 7 秒限速 = 约 3 分钟拉完。补全后 web UI 的"洲际赛事" tab 里的亚冠 / 解放者 / 南美杯都能直接预测。

## 已知不可达

这台 / 这个网络对以下源无法 reach（可能是地理 IP 段被封）：

- `api.clubelo.com` — 俱乐部 Elo（超时）
- `fbref.com` — xG 数据（403）

如果你换个网络（手机热点 / VPN）这两个也能用，scraper 代码已经写好了：
`scrape/clubelo.py` 和 `scrape/fbref.py`。
