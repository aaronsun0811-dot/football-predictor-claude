"""Bulk i18n migration script. Idempotent — running twice is a no-op for
already-replaced strings. Adds new key pairs to both zh + en blocks in app.js.

Patterns handled:
  1. ``<tag attrs>ZH</tag>``  — simple text-only content
  2. ``<tag attrs>ZH</tag>`` with internal whitespace/newlines
  3. JS string literals: ``'ZH'`` inside attribute expressions
  4. ``{ status_a: 'ZH-A', status_b: 'ZH-B' }`` mapping objects

For more complex cases (inline ``<strong>`` tags, multi-line prose with
embedded variables), edit manually after running this.

Run from project root: ``python tools/i18n_migrate.py``.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "static" / "index.html"
APPJS = ROOT / "static" / "app.js"


# ===========================================================================
# Translation table — (key, zh, en)
# ===========================================================================
TEXT_REPLACEMENTS: list[tuple[str, str, str]] = [
    # --- Match tab ----------------------------------------------------------
    ("model_dc_elo_label",    "DC + Elo（默认，但过信）",       "DC + Elo (default, over-confident)"),
    ("model_ensemble_label",  "Ensemble（3 模型平均）",        "Ensemble (3-model avg)"),
    ("model_market_fused_label","Market-Fused（理论最稳）",     "Market-Fused (most stable)"),
    ("model_dc_pblog_label",  "Dixon-Coles（penaltyblog）",   "Dixon-Coles (penaltyblog)"),
    ("model_poisson_label",   "Poisson（基线）",               "Poisson (baseline)"),

    # --- Worldcup -----------------------------------------------------------
    ("worldcup_title",        "世界杯 2026 蒙特卡洛",          "World Cup 2026 Monte Carlo"),
    ("worldcup_n_sims",       "模拟次数",                       "Simulations"),
    ("worldcup_sims_fast",    "1,000（快）",                    "1,000 (fast)"),
    ("worldcup_sims_default", "5,000（默认）",                  "5,000 (default)"),
    ("worldcup_sims_precise", "20,000（精）",                   "20,000 (precise)"),

    # --- Data health (the big one) ------------------------------------------
    ("health_title",          "数据健康仪表盘",                 "Data Health Dashboard"),
    ("health_loading",        "加载中…",                       "Loading…"),
    ("health_total_matches",  "比赛总数",                       "Total matches"),
    ("health_leagues_card",   "联赛",                          "Leagues"),
    ("health_sources_card",   "数据源",                         "Sources"),
    ("health_fresh_card",     "≤7 天新",                       "≤7 days fresh"),
    ("health_stale_card",     ">30 天旧",                      ">30 days stale"),
    ("health_per_league",     "每联赛状态",                     "Per-league status"),
    ("health_per_source",     "每数据源",                       "Per source"),
    ("health_th_league",      "联赛",                          "League"),
    ("health_th_matches",     "场数",                          "Matches"),
    ("health_th_earliest",    "最早",                          "Earliest"),
    ("health_th_latest",      "最新",                          "Latest"),
    ("health_th_days_stale",  "旧 (天)",                       "Stale (days)"),
    ("health_th_primary_src", "主源",                          "Primary source"),
    ("health_th_status",      "状态",                          "Status"),
    ("health_th_source",      "源",                            "Source"),
    ("health_th_latest_match","最新比赛",                       "Latest match"),
    ("health_th_service",     "服务",                          "Service"),
    ("health_freshness_fresh","新鲜",                          "Fresh"),
    ("health_freshness_recent","近期",                         "Recent"),
    ("health_freshness_stale","滞后",                          "Stale"),
    ("health_freshness_old",  "历史",                          "Historical"),
    ("health_configured",     "已配置",                         "Configured"),
    ("health_not_configured", "未配置",                         "Not configured"),
    ("health_register",       "注册",                          "Sign up"),
    ("health_sort_hint",      "按\"新鲜度\"排序，最旧的在底部",  "Sorted by freshness; oldest at the bottom"),

    # --- About --------------------------------------------------------------
    ("about_why_not_high_heading", "为什么准确率不会高于 55%", "Why accuracy won't exceed 55%"),

    # --- Leagues ------------------------------------------------------------
    ("leagues_filter_label",  "联赛",                          "League"),
    ("leagues_tier_label",    "档",                            "Tier"),

    # --- Strengths ----------------------------------------------------------
    ("strengths_sort_label",  "排序",                          "Sort by"),

    # --- Common: many uses of "加载中" as a placeholder ---------------------
    ("loading_short",         "加载中…",                       "Loading…"),

    # --- Reuse existing keys for common verbs ------------------------------
    # (These keys already exist in i18n; we just wrap the HTML literals)
    ("refresh",               "刷新",                          "Refresh"),
    ("run",                   "运行",                          "Run"),

    # --- Data health: cache section -----------------------------------------
    ("health_caches_heading", "磁盘缓存",                       "Disk caches"),
    ("health_caches_hint",
     "每个数据源的本地缓存目录。空的表示这个源没启用或没缓存。",
     "Per-source on-disk cache directories. Empty rows mean the source isn't configured or hasn't cached anything yet."),
    ("health_th_cache",       "缓存",                          "Cache"),
    ("health_th_files",       "文件数",                         "Files"),
    ("health_th_size",        "总大小",                         "Total size"),
    ("health_th_newest_file", "最新文件",                       "Newest file"),
    ("health_th_path",        "路径",                          "Path"),

    # --- Data health: audit section -----------------------------------------
    ("audit_heading",         "实测准确率（已结果的 /upcoming 预测）",
                              "Realized accuracy (resolved /upcoming predictions)"),
    ("audit_intro",
     "把\"本周赛程\"里历史预测和实际比分对账。这是真实场景下的准确率，比合成回测更诚实。",
     "Reconciles past /upcoming predictions with actual scores. This is real-world accuracy — more honest than synthetic backtests."),
    ("audit_empty",
     "暂无已结果的预测。\"本周赛程\"里有预测过的比赛踢完之后会自动结算。",
     "No resolved predictions yet. Fixtures predicted on the 'Upcoming' tab will be reconciled once they finish."),
    ("audit_resolved_count",  "已结果",                         "Resolved"),
    ("audit_wdl_hit_rate",    "胜平负命中率",                   "W/D/L accuracy"),
    ("audit_n_scored",        "已记预测比分",                   "Recorded score predictions"),
    ("audit_n_scored_note",   "(round 18 之后才开始记)",         "(recorded since round 18)"),
    ("audit_score_hit_rate",  "比分命中率",                     "Exact-score accuracy"),
    ("audit_score_hit_note",  "猜对几比几的比例（足球普遍 8-12%）",
                              "Fraction of exact-score predictions correct (football typically 8-12%)"),
    ("audit_goal_distance",   "进球差均值",                     "Mean goal distance"),
    ("audit_per_league",      "按联赛拆分",                     "Per league"),
    ("audit_per_league_note", "(同口径于 /backtest 的 by_league — 可直接比对)",
                              "(same shape as /backtest's by_league — directly comparable)"),
    ("audit_th_wdl",          "胜平负",                         "W/D/L"),
    ("audit_th_score",        "比分",                          "Score"),
    ("audit_th_goal_dist",    "进球差",                         "Goal distance"),

    # --- Backtest tab ------------------------------------------------------
    ("bt_subtitle_line1",
     "每场比赛只用之前的数据训练，预测当场——评估真实泛化能力。",
     "Walk-forward: train on prior matches only, predict the current one — measures real generalization."),
    ("bt_subtitle_line2",
     "每个联赛 ~30 秒。Accuracy = 1X2 命中率，Brier ↓ 越准，Log loss ↓ 越准。",
     "~30 seconds per league. Accuracy = 1X2 hit rate. Brier ↓ better. Log loss ↓ better."),
    ("bt_league_auto",        "— 自动（合并所有）—",               "— Auto (all merged) —"),
    ("bt_min_train",          "最少训练样本",                       "Min training matches"),
    ("bt_min_train_hint",     "越小越快但越早期的预测越噪。100-200 通常合理。",
                              "Smaller = faster but earlier predictions are noisier. 100-200 is usually reasonable."),
    ("bt_refit_every",        "每 N 场重新拟合",                    "Refit every N matches"),
    ("bt_refit_every_hint",   "refit_every=1 最准但慢，10-25 兼顾速度。",
                              "refit_every=1 is most accurate but slow. 10-25 balances speed."),
    ("bt_run",                "运行回测",                          "Run backtest"),
    ("bt_running",            "计算中…可能 30-60 秒",               "Computing… 30-60 seconds"),
    ("bt_pick_league",        "选联赛后点「运行回测」。预计 30-60 秒。",
                              "Pick a league, then click Run. Expect 30-60 seconds."),
    ("bt_walking",            "滚动训练 + 预测中…",                 "Walk-forward training + predicting…"),
    ("bt_sample_count",       "样本数",                            "Sample size"),
    ("lower_better",          "越低越好",                          "Lower = better"),
    ("bt_score_metrics",      "比分级别指标",                       "Score-level metrics"),
    ("bt_scored_count",       "参与评分的预测",                     "Scored predictions"),
    ("bt_score_acc_note",     "几比几猜对的比例（足球普遍 8-12%）",
                              "Fraction of exact-score predictions correct (football typically 8-12%)"),
    ("bt_goal_dist_note",     "|预测 H − 实际 H| + |预测 A − 实际 A|",
                              "|predicted H − actual H| + |predicted A − actual A|"),
    ("bt_by_league_hint",
     "仅显示样本数 ≥ 30 的联赛 — 样本太小的数字不可靠。按 N 倒序。",
     "Only leagues with ≥ 30 samples shown — smaller samples are unreliable. Sorted by N desc."),
    ("bt_fit_health",         "拟合健康度",                         "Fit health"),
    ("bt_fit_all_ok",         "全部成功",                          "All ok"),
    ("bt_fit_high_fail",      "失败率偏高",                         "High failure rate"),
    ("bt_fit_few_fail",       "少量失败",                          "Some failures"),
    ("bt_fit_attempts",       "尝试拟合",                          "Fit attempts"),
    ("bt_fit_success",        "拟合成功",                          "Successful fits"),
    ("bt_fit_failures",       "失败次数",                          "Failures"),
    ("bt_fit_fresh_pct",      "新鲜拟合比例",                       "Fresh-fit %"),
    ("bt_fit_fresh_note",     "每次预测时正好刚 refit",              "Predictions made on the same iteration as a refit"),

    # --- ROI tab -----------------------------------------------------------
    ("roi_intro_para",
     "上面的「价值发现」给出 EV 和 Kelly 建议，听着很美好。",
     "The 'Value Finder' tab gives EV and Kelly suggestions, which sounds great in theory."),
    ("roi_intro_para2",
     "这一页做的事很简单:过去几年我们如果真的按那套规则下注,钱包是涨还是缩?",
     "This page does one thing: if we'd actually bet on those rules over the past few years, would the bankroll go up or down?"),
    ("roi_min_edge",          "最小 edge (模型 - 隐含)",            "Min edge (model − implied)"),
    ("roi_min_edge_hint",     "只下 模型概率 - 盘口隐含 ≥ 这个值的注。", "Only bet when model_prob − implied_prob ≥ this threshold."),
    ("roi_min_ev",            "最小 EV",                            "Min EV"),
    ("roi_min_ev_hint",       "期望价值阈值。EV<0 永远不下。",       "Expected-value threshold. Never bet EV<0."),
    ("roi_kelly_mult",        "Kelly 乘数",                         "Kelly multiplier"),
    ("roi_kelly_mult_hint",   "0.5 = 半 Kelly,稳一点。1.0 = 全 Kelly,激进。", "0.5 = half-Kelly (safer). 1.0 = full Kelly (aggressive)."),
    ("roi_implied_method",    "盘口隐含方法",                       "Implied-prob method"),
    ("roi_implied_shin",      "Shin (推荐)",                        "Shin (recommended)"),
    ("roi_run",               "运行 ROI 模拟",                      "Run ROI simulation"),
    ("roi_running",           "回测中…可能 10-30 秒",                "Simulating… 10-30 seconds"),
    ("roi_pick_league",       "选联赛后点「运行」。预计 10-30 秒。",  "Pick a league, then click Run. Expect 10-30 seconds."),
    ("roi_starting",          "初始资金",                           "Starting bankroll"),
    ("roi_final",             "最终资金",                           "Final bankroll"),
    ("roi_roi",               "ROI",                                "ROI"),
    ("roi_n_bets",            "下注次数",                           "Bets placed"),
    ("roi_n_total",           "总比赛数",                           "Total matches"),
    ("roi_win_rate",          "命中率",                             "Hit rate"),

    # --- Value tab ---------------------------------------------------------
    ("value_home_odds",       "主胜赔率",                           "Home odds"),
    ("value_draw_odds",       "平局赔率",                           "Draw odds"),
    ("value_away_odds",       "客胜赔率",                           "Away odds"),
    ("value_calc",            "计算价值",                           "Compute value"),
    ("value_no_value",        "无价值",                             "No value"),
    ("value_has_value",       "有价值",                             "Value!"),
    ("value_model_says",      "模型概率",                           "Model prob"),
    ("value_implied",         "隐含概率",                           "Implied prob"),
    ("value_edge_pp",         "edge (pp)",                          "Edge (pp)"),
    ("value_ev_per_unit",     "EV / 单位",                           "EV / unit"),
    ("value_kelly_pct",       "Kelly 比例",                         "Kelly fraction"),
    ("value_pick_first",      "先在「比赛」tab 跑一次预测,把概率搬过来。", "Run a prediction in the Match tab first; copy the probabilities here."),

    # --- Continental tab ---------------------------------------------------
    ("continental_intro",
     "欧冠 / 欧联 / 欧会杯 / 解放者杯 / 亚冠 + 国家队大赛 (欧洲杯 / 美洲杯 / 亚洲杯 / 非洲杯)。",
     "UCL / Europa / Conference / Libertadores / AFC CL + national-team tournaments (Euro / Copa America / Asian Cup / AFCON)."),
    ("continental_neutral",   "默认中立场(开关在「比赛」tab)。",       "Neutral venue by default (toggle on the Match tab)."),
    ("continental_pick",      "选一场",                            "Pick a match"),

    # --- In-play tab -------------------------------------------------------
    ("inplay_subtitle_para",
     "填入当前比分 + 当前分钟,重新计算最终结果概率。",
     "Enter the current score + minute. Recomputes final-result probabilities."),
    ("inplay_current_home",   "当前主队进球",                       "Current home goals"),
    ("inplay_current_away",   "当前客队进球",                       "Current away goals"),
    ("inplay_minute",         "已踢分钟",                           "Minutes elapsed"),
    ("inplay_recompute",      "重新计算",                           "Recompute"),
    ("inplay_pre_match",      "赛前预测",                           "Pre-match"),
    ("inplay_now",            "当前局势",                           "Current state"),
    ("inplay_chasing_mult",   "追分加成 (默认 1.15)",                "Chasing multiplier (default 1.15)"),
    ("inplay_leading_mult",   "领先减成 (默认 0.92)",                "Leading multiplier (default 0.92)"),

    # --- Strengths tab -----------------------------------------------------
    ("strengths_intro",
     "基于 Dixon-Coles 拟合的攻防参数 + 近 5 场战绩。",
     "Dixon-Coles attack/defense parameters + last-5 form."),
    ("strengths_attack_note", "攻 越高 = 进球能力越强",              "Attack: higher = better at scoring"),
    ("strengths_defense_note","防 越高 = 失球越少 (已翻转,统一越高越好)。",
                              "Defense: higher = fewer conceded (inverted, so higher is always better)."),
    ("strengths_loading",     "拟合中…",                            "Fitting…"),
    ("strengths_compare",     "两队对比",                           "Compare two teams"),

    # --- Diagnostics tab ---------------------------------------------------
    ("diag_subtitle",
     "光看 accuracy 不够。这里给三个更深的视角:",
     "Accuracy isn't enough. Three deeper views:"),
    ("diag_run",              "运行诊断",                           "Run diagnostics"),
    ("diag_running",          "诊断中…",                            "Running…"),
    ("diag_calibration",      "校准图",                             "Calibration curve"),
    ("diag_calibration_perfect", "完美校准线",                       "Perfect calibration"),
    ("diag_confidence_ladder","信心阶梯",                           "Confidence ladder"),

    # --- Replay tab --------------------------------------------------------
    ("replay_loading",        "复盘中…",                            "Replaying…"),
    ("replay_predict_was",    "模型预测",                           "Model predicted"),
    ("replay_actual_was",     "实际结果",                           "Actual result"),
    ("replay_filter_league",  "限定联赛",                           "Filter by league"),
    ("replay_filter_all",     "— 全部 —",                          "— All —"),

    # --- Match tab: result cards -------------------------------------------
    ("match_most_likely_score","最可能比分",                        "Most likely score"),
    ("match_score_model_prob","模型给这个比分的概率",                "Model probability for this score"),
    ("match_xg_label",        "预期进球（xG）",                     "Expected goals (xG)"),
    ("match_elo_label",       "Elo 评级",                          "Elo rating"),
    ("match_top5_scores",     "最可能比分（top 5）",                "Most likely scores (top 5)"),
    ("match_score_matrix",    "比分概率矩阵",                       "Score probability matrix"),
    ("match_matrix_corner",   "主\\客",                            "H\\A"),
    ("match_matrix_legend",   "单位 %，行=主队进球，列=客队进球",   "Values in %. Rows = home goals, columns = away goals."),
    ("h2h_win_legend",        "胜",                                "W"),
    ("h2h_draw_legend",       "平",                                "D"),
    ("h2h_loss_legend",       "负",                                "L"),

    # --- Upcoming tab ------------------------------------------------------
    ("upcoming_loading",      "加载中...",                         "Loading…"),
    ("upcoming_fetch_first",  "加载近期对阵...第一次较慢（含 fd.org 日期核对）",
                              "Loading upcoming fixtures… first load is slower (cross-checks fd.org dates)"),
    ("upcoming_scope_hint",   "展示范围：欧洲顶级联赛 + 中日韩 + 沙特 + 美洲 + 洲际杯",
                              "Scope: top European leagues + China/Japan/Korea + Saudi + Americas + continental cups"),
    ("upcoming_expand_hint",  "点击展开完整预测",                   "Click to expand full prediction"),

    # --- Worldcup ----------------------------------------------------------
    ("wc_computing",          "计算中…",                           "Computing…"),
    ("wc_from_cache",         "缓存命中",                          "Cache hit"),
    ("wc_instant",            "即时计算",                          "Just computed"),
    ("wc_th_team",            "球队",                              "Team"),
    ("wc_th_final",           "决赛",                              "Final"),
    ("wc_th_champion",        "夺冠",                              "Champion"),
    ("wc_chart_title",        "夺冠概率（top 10）",                "Champion probability (top 10)"),
    ("wc_run_hint",           "点击「运行」开始模拟。首次约 5 秒，之后从缓存读取。",
                              "Click Run to start the simulation. First run ~5s, then served from cache."),

    # --- Data coverage tab -------------------------------------------------
    ("coverage_intro",        "每个联赛的覆盖等级、数据状态 + 最近 30 场比赛。",
                              "Per-league coverage tier, data state, and the last 30 matches."),
    ("coverage_th_data",      "数据",                              "Data"),
    ("coverage_pick_left",    "左侧选择一个联赛",                   "Pick a league on the left"),
    ("coverage_th_date",      "日期",                              "Date"),
    ("coverage_th_home",      "主队",                              "Home"),
    ("coverage_th_away",      "客队",                              "Away"),
    ("coverage_empty",        "该联赛暂无数据。可能是 Tier 3 联赛——football-data.co.uk 没覆盖，需要 API-Football 拉取。",
                              "No data for this league yet. Likely a Tier 3 league — not covered by football-data.co.uk, needs API-Football."),

    # --- Backtest tab: more details ----------------------------------------
    ("bt_max_stale",          "最大过期",                          "Max staleness"),
    ("bt_max_stale_unit",     "场",                                "matches"),
    ("bt_max_stale_hint",     "场（最长一次预测时模型已 N 场没重训）",
                              "matches (longest gap between refit and the prediction)"),
    ("bt_avg_stale",          "平均过期",                          "Mean staleness"),
    ("bt_real_dist",          "真实结果分布",                       "Actual outcome distribution"),
    ("bt_baseline_ref",       "基线参考：随机三选一=33%，\"永远选主\"=",
                              "Baselines: random 1-of-3 = 33%; \"always pick home\" ="),
    ("bt_market_close",       "盘口收盘=53-55%。",                 "; closing odds ≈ 53–55%."),
    ("bt_how_to_read",        "怎么读这个数字？",                   "How to read this"),
    ("bt_acc_53",             "Accuracy ≥ 53% = 接近盘口水平，可挖 value。",
                              "Accuracy ≥ 53% = near closing-odds level; value-bet candidates exist."),
    ("bt_acc_48_52",          "48-52% = 比\"永远选主\"略好，模型还有调参空间。",
                              "48–52% = slightly better than always-home; room to tune."),
    ("bt_acc_lt48",           "< 48% = 数据太少或联赛方差太大（Tier 3 常见）。",
                              "< 48% = too little data or high variance (common for Tier 3)."),
    ("bt_xg_note",            "xG 已经可以通过 python predict.py update --include-xg 接入；有 xG 的联赛会在拟合时混合实际进球和 xG。",
                              "xG can be ingested via python predict.py update --include-xg; leagues with xG mix actual goals + xG during fit."),
    ("bt_fit_failed_pts",     "失败的拟合点 (前",                   "Failed fit points (first"),
    ("bt_fit_failed_pts_2",   "个)",                              ")"),

    # --- Diagnostics tab ---------------------------------------------------
    ("diag_auto_option",      "— 自动 —",                          "— Auto —"),
    ("diag_n_hint",           "50 通常 30 秒以内出结果。",          "n=50 usually completes within 30 seconds."),
    ("diag_pick_first",       "选择联赛点「运行诊断」",              "Pick a league, then click Run."),
    ("diag_walking",          "滚动回测 + 诊断中…",                 "Walk-forward + diagnostics…"),
    ("diag_ece_excellent",    "< 0.04 优秀",                       "< 0.04 excellent"),
    ("diag_ece_ok",           "0.04-0.08 一般",                    "0.04–0.08 ok"),
    ("diag_ece_poor",         "> 0.08 差",                         "> 0.08 poor"),
    ("diag_curve_title",      "校准曲线 · 预测概率 vs 实际发生率",   "Calibration curve · predicted vs actual frequency"),
    ("diag_curve_legend",     "虚线 y=x 是完美校准。点高于虚线=模型保守（实际比预测的更频繁）；点低于=模型过信。",
                              "The dashed y=x line is perfect calibration. Above it = model conservative (reality more frequent); below = over-confident."),
    ("diag_ladder_title",     "信心阶梯 · 模型最高概率档的实际命中率",
                              "Confidence ladder · realized hit rate per top-probability band"),
    ("diag_ladder_th_band",   "信心档",                            "Band"),
    ("diag_ladder_th_hit",    "实际命中",                          "Realized hit"),
    ("diag_ladder_th_pred",   "平均预测",                          "Mean predicted"),
    ("diag_ladder_th_bias",   "偏差",                              "Bias"),
    ("diag_ladder_note",      "红色行 = 高信心档过信（应该慎下注）。绿色行 = 模型反而保守，可挖 value。",
                              "Red row = high-confidence band over-confident (bet cautiously). Green row = model is conservative; value-bet candidate."),

    # --- Value tab ---------------------------------------------------------
    ("value_odds_heading",    "盘口赔率（十进制）",                "Bookmaker odds (decimal)"),
    ("value_home_short",      "主胜",                              "Home"),
    ("value_draw_short",      "平",                                "Draw"),
    ("value_away_short",      "客胜",                              "Away"),
    ("value_analyze",         "分析价值",                          "Analyze value"),
    ("value_computing",       "计算中…",                           "Computing…"),
    ("value_pick_hint",       "填好两队 + 三个赔率，点击「分析价值」",
                              "Fill in both teams + three odds, then click Analyze."),
    ("value_compare_heading", "盘口隐含概率 vs 模型概率",            "Implied probability vs model probability"),
    ("value_edge_short",      "边际",                              "Edge"),
    ("value_detected",        "检测到价值",                        "Value detected"),
    ("value_none",            "无明显价值",                        "No clear value"),
    ("value_kelly_advice",    "Kelly 建议",                        "Kelly suggestion"),
    ("value_no_value_note",   "所有三个结果的盘口都比模型预测的更准或几乎相等。这通常是市场对的，看看下一场吧。",
                              "On all three outcomes the bookmaker's implied probabilities match or beat the model. Usually means the market is right — try another match."),

    # --- ROI tab: controls + warning ---------------------------------------
    ("roi_min_edge_label",    "最少边际 (pp)",                     "Min edge (pp)"),
    ("roi_min_edge_hint2",    "5pp = 模型概率比盘口隐含高 5 个百分点才下。",
                              "5pp = only bet when model prob ≥ implied prob + 5 percentage points."),
    ("roi_min_ev_label",      "最少 EV",                           "Min EV"),
    ("roi_min_ev_hint2",      "5% = 期望每元至少赚 5 分才下。",     "5% = only bet when EV ≥ 5%."),
    ("roi_kelly_label",       "Kelly 系数",                        "Kelly factor"),
    ("roi_kelly_hint2",       "0.5 = 半 Kelly（默认，更稳）。",     "0.5 = half-Kelly (default, safer)."),
    ("roi_model_label",       "模型",                              "Model"),
    ("roi_model_dc_default",  "DC + Elo（默认）",                  "DC + Elo (default)"),
    ("roi_model_market",      "Market-Fused（推荐：最稳）",        "Market-Fused (recommended: safest)"),
    ("roi_model_hint",        "Market-Fused 在 La Liga / Primeira 实测亏损最小。",
                              "Market-Fused has the smallest realized loss on La Liga / Primeira."),
    ("roi_run_label",         "运行模拟",                          "Run simulation"),
    ("roi_running_label",     "回测中…30-60 秒",                   "Backtesting… 30–60 seconds"),
    ("roi_pick_first",        "选择联赛与阈值，点「运行模拟」",      "Pick a league + thresholds, then click Run."),
    ("roi_walking_label",     "滚动回测 + 模拟下注中…",            "Walk-forward + simulated betting…"),
    ("roi_known_result_h",    "已知结果（剧透）",                   "Known result (spoiler)"),

    # --- Continental tab: model panel + form -------------------------------
    ("cont_pick_match",       "选择赛事",                          "Pick a match"),
    ("cont_home_label",       "主队（中立场可视为 Team A）",        "Home (treat as Team A if neutral)"),
    ("cont_knockout",         "淘汰赛",                            "Knockout"),
    ("cont_neutral",          "中立场",                            "Neutral"),
    ("cont_predict",          "预测",                              "Predict"),
    ("cont_common_matchups",  "常见对阵（点击填入）",                "Common matchups (click to fill)"),
    ("cont_pick_first",       "选择赛事和两支队伍，点「预测」",      "Pick a match and two teams, then click Predict."),
    ("cont_neutral_site",     "中立场",                            "Neutral venue"),
    ("cont_home_advantage",   "主场优势",                          "Home advantage"),
    ("cont_knockout_short",   " · 淘汰赛",                         " · knockout"),
    ("cont_advance_prob",     "晋级概率（含加时+点球）",            "Advance probability (incl. ET + pens)"),
    ("cont_model_source_h",   "模型来源",                          "Model source"),
    ("cont_model_cross_league","跨联赛 Dixon-Coles",                "Cross-league Dixon-Coles"),
    ("cont_model_matches",    "场比赛，",                           "matches,"),
    ("cont_model_merged_prefix","合并",                            "merged"),
    ("cont_model_merged_suffix","个域内顶级联赛",                    "top leagues from the region"),
    ("cont_elo_home",         "主",                                "home"),
    ("cont_elo_away",         "客",                                "away"),
    ("cont_model_national_elo","纯 Elo（国家队）",                  "Pure Elo (national teams)"),
    ("cont_model_national_elo_desc",
     "基于 eloratings.net 的国家队评级，配合自适应平局比例和加时点球的简单晋级概率。",
     "Based on eloratings.net national-team ratings, with an adaptive draw rate plus a simple ET/penalties advance model."),
    ("cont_elo_diff",         "Elo 差",                            "Elo diff"),
    ("cont_draw_share_adapted","平局比例自适应到",                   "draw rate adapts to"),
    ("cont_default_filter",   "— 全部 / 自动 —",                   "— All / Auto —"),

    # --- In-play tab -------------------------------------------------------
    ("inplay_current_score",  "当前比分",                          "Current score"),
    ("inplay_kickoff",        "开场 0'",                           "Kickoff 0'"),
    ("inplay_halftime",       "半场 45'",                          "Half 45'"),
    ("inplay_fulltime",       "结束 90'",                          "End 90'"),
    ("inplay_pick_first",     "填入两队 + 当前比分 + 分钟，点「重新计算」",
                              "Enter both teams + current score + minute, then click Recompute."),
    ("inplay_final_probs",    "最终结果概率",                       "Final-result probabilities"),
    ("inplay_expected_final", "预期终盘比分",                       "Expected final score"),
    ("inplay_state_adjust",   "状态修正",                          "Game-state adjustment"),
    ("inplay_state_legend",   "落后方 ×1.15、领先方 ×0.92（基于多年实证）",
                              "Trailing side ×1.15, leading side ×0.92 (based on multi-year empirical evidence)"),
    ("inplay_top_final",      "最可能的终盘比分",                   "Most likely final scores"),
    ("inplay_caveat",
     "模型不感知红牌、伤停、换人战术调整、定位球数量等。是个数学基线，不是\"通灵预测\"。比如真实场景里 75 分钟领先 1-0 守不住的概率，要在这个数字上加几个百分点的\"人为因素\"。",
     "The model doesn't see red cards, injuries, tactical subs, or set-piece counts. It's a math baseline, not a 'psychic prediction'. For real-world scenarios — like the chance a 1-0 lead at 75' is blown — add a few percentage points of 'human factor' on top."),

    # --- ROI result cards --------------------------------------------------
    ("roi_final_roi",         "最终 ROI",                          "Final ROI"),
    ("roi_final_bankroll",    "终值",                              "Final bankroll"),
    ("roi_total_staked",      "总投注",                            "Total staked"),
    ("roi_max_drawdown",      "最大回撤",                          "Max drawdown"),
    ("roi_bankroll_curve",    "资金曲线",                          "Bankroll curve"),

    # --- Backtest interpretation (new keys) --------------------------------
    ("bt_how_to_read_html",
     "Accuracy ≥ 53% = 接近盘口水平,可挖 value。48-52% = 比\"永远选主\"略好,模型还有调参空间。&lt; 48% = 数据太少或联赛方差太大(Tier 3 常见)。",
     "Accuracy ≥ 53% = near closing-odds level; value-bet candidates exist. 48–52% = slightly better than always-home; room to tune. &lt; 48% = too little data or high variance (common for Tier 3)."),
    ("bt_xg_note_html",
     "xG 已经可以通过 <code>python predict.py update --include-xg</code> 接入;有 xG 的联赛会在拟合时混合实际进球和 xG。",
     "xG can be ingested via <code>python predict.py update --include-xg</code>; leagues with xG mix actual goals + xG during fit."),

    # --- Strengths form ----------------------------------------------------
    ("str_sort_overall",      "综合（默认）",                       "Overall (default)"),
    ("str_sort_attack",       "进攻",                              "Attack"),
    ("str_sort_defense",      "防守",                              "Defense"),
    ("str_sort_recent_gd",    "近 5 场净胜球",                     "Last-5 GD"),
    ("str_sort_team",         "球队名",                            "Team name"),
    ("str_fitting",           "拟合 Dixon-Coles 中…",              "Fitting Dixon-Coles…"),
    ("str_th_attack",         "攻",                                "Atk"),

    # --- Diagnostics pick + walk -------------------------------------------
    ("diag_pick_run",         "选择联赛点击「运行诊断」",            "Pick a league, then click Run diagnostics."),

    # --- Misc / various small ----------------------------------------------
    ("ui_lang_zh_label",      "中",                                "中"),
    ("upcoming_click_to_expand","点击展开完整预测",                  "Click to expand the full prediction"),
    ("match_score_model_prob","模型给这个比分的概率",                "Model probability for this score"),

    # --- ROI value mid-sentence -------------------------------------------
    ("roi_dot_edge",          " · 边际",                           " · edge"),
    ("roi_half_kelly_note",   "（半 Kelly 更稳：",                  " (half-Kelly is safer: "),

    # --- Backtest fit-detail row ------------------------------------------
    ("bt_train_set",          " 训练集 ",                          " training set "),
    ("bt_matches_dot",        " 场 · ",                            " matches · "),
]


# JS-expression string literal replacements. Each entry:
#   (key_to_call, zh_literal_in_quotes)
# All occurrences of ``'ZH'`` and ``"ZH"`` inside the HTML get rewritten to
# ``$t('key_to_call')``. Use this for strings inside Alpine x-text / :title /
# mapping objects / ternaries — places the tag-content regex can't reach.
JS_LITERAL_REPLACEMENTS: list[tuple[str, str]] = [
    # Reuse existing health_freshness_* keys for the status mapping object
    ("health_freshness_fresh",  "新鲜"),
    ("health_freshness_recent", "近期"),
    ("health_freshness_stale",  "滞后"),
    ("health_freshness_old",    "历史"),
    ("health_configured",       "已配置"),
    ("health_not_configured",   "未配置"),
]


# ===========================================================================
# Migration logic
# ===========================================================================

# Anchors we use to inject new i18n keys at the bottom of each I18N block
ZH_INSERT_ANCHOR = '    date_confirmed_title: "fd.org 也把这场比赛标在这天",'
EN_INSERT_ANCHOR = '    date_confirmed_title: "fd.org also lists this match on this date",'


def js_escape(s: str) -> str:
    """Escape a string for safe inclusion in a JS double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    html = HTML.read_text()
    appjs = APPJS.read_text()

    n_html = 0
    new_zh_lines: list[str] = []
    new_en_lines: list[str] = []
    already_have_keys = set(re.findall(r"^    (\w+):", appjs, re.M))

    for key, zh, en in TEXT_REPLACEMENTS:
        target = zh.strip()
        # Pattern: an HTML tag pair where the inner text equals our target.
        # Allow attributes on the open tag — but skip if x-text= is already there.
        pattern = re.compile(
            r"<(\w+)((?:(?!x-text=)[^>])*)>(\s*)" + re.escape(target) + r"(\s*)</\1>",
            re.DOTALL,
        )

        def _sub(m, _key=key):
            tag = m.group(1)
            attrs = m.group(2).rstrip()
            ws_before = m.group(3)
            ws_after = m.group(4)
            # Always need a space between tag-name and x-text=. If attrs is empty,
            # the regex matched "<tag>" so we need to insert one. If attrs is
            # non-empty, ensure we have a single space before x-text.
            sep = " "
            return f"<{tag}{attrs}{sep}x-text=\"$t('{_key}')\">{ws_before}{ws_after}</{tag}>"

        new_html, n = pattern.subn(_sub, html, count=10)  # up to 10 occurrences per key
        if n > 0:
            html = new_html
            n_html += n
            if key not in already_have_keys:
                new_zh_lines.append(f'    {key}: "{js_escape(zh)}",')
                new_en_lines.append(f'    {key}: "{js_escape(en)}",')
                already_have_keys.add(key)

    # Pass 2: JS-expression string-literal rewrites.
    # ``'ZH'`` (or ``"ZH"``) inside any HTML attribute value becomes
    # ``$t('key')``. We don't try to be smart about context — if the literal
    # appears anywhere it gets swapped, and Alpine evaluates ``$t(...)`` in JS.
    n_js = 0
    for key, zh in JS_LITERAL_REPLACEMENTS:
        # Single-quote form
        before = html
        html = html.replace(f"'{zh}'", f"$t('{key}')")
        if html != before:
            n_js += before.count(f"'{zh}'") - html.count(f"'{zh}'")
        # Double-quote form
        before = html
        html = html.replace(f'"{zh}"', f"$t('{key}')")
        if html != before:
            n_js += before.count(f'"{zh}"') - html.count(f'"{zh}"')

    if new_zh_lines:
        appjs = appjs.replace(
            ZH_INSERT_ANCHOR,
            ZH_INSERT_ANCHOR + "\n" + "\n".join(new_zh_lines),
            1,
        )
        appjs = appjs.replace(
            EN_INSERT_ANCHOR,
            EN_INSERT_ANCHOR + "\n" + "\n".join(new_en_lines),
            1,
        )

    HTML.write_text(html)
    APPJS.write_text(appjs)
    print(f"Made {n_html} text-content replacements in index.html")
    print(f"Made {n_js} JS-literal replacements in index.html")
    print(f"Added {len(new_zh_lines)} new key pairs to app.js")


if __name__ == "__main__":
    main()
