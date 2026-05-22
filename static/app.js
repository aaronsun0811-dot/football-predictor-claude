/* Football Predictor — Alpine.js app glue.
 *
 * One global function: app() returns the reactive state + methods that
 * static/index.html binds against. No build step; runs straight in the browser.
 */

// Bilingual UI strings. Keep keys short and stable; HTML uses $t('key').
// Only translate visible/headline strings — verbose help text stays in zh for now.
const I18N = {
  zh: {
    // tabs
    tab_match: "比赛预测",
    tab_worldcup: "世界杯 2026",
    tab_leagues: "联赛浏览",
    tab_backtest: "回测",
    tab_diagnostics: "模型诊断",
    tab_value: "价值发现",
    tab_roi: "ROI 验证",
    tab_continental: "洲际赛事",
    tab_inplay: "实时进行中",
    tab_strengths: "球队强度",
    tab_replay: "比赛复盘",
    tab_upcoming: "本周赛程",
    tab_health: "数据健康",
    tab_about: "关于",
    // common
    home_team: "主队",
    away_team: "客队",
    league: "联赛",
    model: "模型",
    home_win: "主胜",
    draw: "平",
    away_win: "客胜",
    predict: "预测",
    calculating: "计算中…",
    run: "运行",
    refresh: "刷新",
    neutral_site: "中立场",
    knockout: "淘汰赛",
    expected_goals: "预期进球",
    training_rows: "训练样本",
    most_likely_scores: "最可能比分（top 5）",
    score_matrix: "比分概率矩阵",
    leagues_count: "联赛",
    matches_count: "场",
    teams_count: "队",
    online: "在线",
    offline: "离线",
    // tab subtitles / hints
    match_subtitle: "Dixon-Coles + Elo 修正。胜/平/负概率、预期进球、最可能比分。",
    choose_teams: "选择两队后点击「预测」",
    all_auto: "— 全部 / 自动 —",
    manual_predict_heading: "手动预测一场",
    upcoming_board_heading: "近期对阵 · 实时预测",
    more_options: "更多选项",
    featured_only_note: "仅显示重要联赛 + 有完整预测的对阵",
    // /upcoming tab strings (R42 i18n pass)
    upcoming_title: "本周赛程 + 实时预测",
    upcoming_subtitle: "从 TheSportsDB（免费、无 key）拉取未来几天的真实赛程，每一场都用我们训练好的 Dixon-Coles 模型实时跑出胜平负概率。中超、J1、MLS、欧洲五大、英冠、比甲、荷甲 都覆盖。",
    days_ahead_label: "未来天数",
    days_today_plus_3: "今天 + 3 天",
    days_next_7: "未来 7 天",
    days_next_14: "未来 14 天",
    days_next_30: "未来 30 天",
    upcoming_loading: "拉取中…",
    fixtures_count_suffix: "场比赛 · 数据来源 TheSportsDB",
    upcoming_loading_detail: "拉赛程 + 逐场跑模型…可能 10-30 秒",
    upcoming_not_loaded: "还没拉过赛程。",
    upcoming_load_now: "立即拉取",
    upcoming_empty_prefix: "TheSportsDB 没有返回未来",
    upcoming_empty_suffix: "天内的赛程。",
    upcoming_empty_hint: "免费档每个联赛通常只返回 1-2 场。试着把天数拉大到 14 或 30 天。",
    upcoming_try_30: "改为 30 天重试",
    alt_scores_label: "备选",
    total_goals_label: "总进球",
    delta_h_ago: "h前",
    upcoming_howto_heading: "怎么用",
    upcoming_howto_1: "赛程来源是 TheSportsDB（免费，无 key），所以每个联赛通常只返回 1-2 场最近的",
    upcoming_howto_2: "每条预测用的是赛前训练的 Dixon-Coles + Elo 模型，n= 是训练样本数",
    upcoming_howto_3: '状态 "Not Started" / "P" = 未开赛；"Match Finished" = 已踢完（可对比预测 vs 实际）',
    upcoming_howto_4: "缓存 6 小时,避免对 TheSportsDB 频繁打扰",
    metadata_only_warn: "⚠ 暂无法预测",
    metadata_only_fallback_detail: "该联赛缺历史数据",
    h2h_other_leagues: "过去交锋（其他赛事的历史数据）",
    draw_short: "平",
    date_warn_title_prefix: "fd.org 显示这场比赛在",
    date_confirmed_title: "fd.org 也把这场比赛标在这天",
    market_letter: "盘",
    model_word: "模型",
    market_implied: "盘口隐含",
    bet_rate_suffix: "下注率",
    roi_red_zone_a: "红色区域 = 资金在 starting bankroll",
    roi_red_zone_b: "以下。",
    club_short: "俱乐部",
    national_short: "国家队",
    cont_pick_match_placeholder: "请选择赛事",
    inplay_current_minute: "当前分钟",
    remaining_word: "剩余",
    minutes_unit: "分钟",
    pre_match_was: "赛前为",
    home_win_short: "主胜",
    away_win_short: "客胜",
    draw_word: "平局",
    train_samples: "训练样本",
    h2h_recent_prefix: "近",
    h2h_recent_suffix: "次交锋",
    perspective: "视角",
    venue_home: "主场",
    venue_away: "客场",
    h2h_win_short: "胜",
    h2h_draw_short: "平",
    h2h_loss_short: "负",
    matches_unit: "场",
    days_unit: "天",
    upcoming_summary_a: "已预测",
    upcoming_summary_b: "全部",
    upcoming_summary_c: "未来",
    no_prediction: "无预测",
    upcoming_no_pred_a: "未来",
    upcoming_no_pred_b: "天没有覆盖联赛的预测对阵",
    view_all_prefix: "查看全部",
    view_all_with_minor: "场(含小联赛)",
    wc_n_sims_suffix: "次模拟",
    coverage_history_matches: "场历史比赛",
    latest_label: "最新",
    no_data: "无数据",
    bt_train_set: "训练集",
    value_model_letter: "模",
    inplay_away_letter: "客",
    audit_of_which_prefix: "其中",
    audit_of_which_suffix: "个当时有日期警告",
    load_failed: "加载失败",
    avb_compared_prefix: "对比了",
    avb_compared_suffix: "个联赛",
    avb_skipped_prefix: "跳过",
    avb_skipped_reason: "backtest 数据不足",
    diff_prefix: "差",
    balls_unit: "球",
    predicted_label: "预测",
    unmatched_warn_a: "还有",
    unmatched_warn_b: "个预测对不上",
    unmatched_warn_c: "近 30 天里",
    unmatched_warn_d: "对得上",
    venues_flipped: "主客互换",
    nearby_candidate_label: "最近候选",
    suggested_alias: "建议 alias",
    backfill_window: "窗口",
    backfill_done_at: "跑完",
    overlap_no_conflict: "跨源重叠但比分一致的",
    total_short: "共",
    collapse_action: "收起",
    expand_action: "展开",
    submitting: "提交中…",
    submit: "提交",
    snapshot_time: "快照时间",
    status_ready: "可用",
    status_sparse: "稀疏",
    status_empty: "缺数据",
    status_unknown: "未知",
    bt_subtitle_line1: "每场比赛只用之前的数据训练,预测当场—评估真实泛化能力。",
    bt_subtitle_line2: "每个联赛 ~30 秒。Accuracy = 1X2 命中率,Brier ↓ 越准,Log loss ↓ 越准。",
    bt_fit_failed_pts_2: "个)",
    h2h_win_legend: "胜",
    h2h_loss_legend: "负",
    manual_entry_league_ph: "英超 / premier_league / EPL",
    api_endpoints_listing: "POST /predict     单场预测\nPOST /backtest    滚动回测\nPOST /update      后台增量更新\nGET  /doctor      数据源诊断\nGET  /worldcup/forecast   世界杯模拟\nGET  /teams       球队列表\nGET  /leagues     联赛列表\nGET  /coverage    联赛覆盖率\nGET  /recent      最近比赛\nGET  /stats       数据库统计\nGET  /health      健康检查\nGET  /api-football/leagues  API-Football 联赛 ID 查询\nGET  /export/{matches|ratings|players|player_season_stats|update_state}  导出 CSV",
    footer_latest_data: "最新数据",
    match_score_prob_inline: "模型给这个比分的概率",
    upcoming_click_to_expand_title: "点击展开完整预测",
    bt_fit_all_ok_label: "全部成功",
    bt_fit_high_fail_label: "失败率偏高",
    bt_fit_few_fail_label: "少量失败",
    value_dot_edge: " · 边际",
    value_half_kelly_inline: "(半 Kelly 更稳:",
    cont_knockout_opt: "淘汰赛",
    cont_neutral_opt: "中立场",
    health_30_days_old: ">30 天旧",
    audit_date_mismatch_tooltip: "当时 fd.org 已经标注这些比赛的日期可能错位 — 模型本身不背锅",
    audit_per_league_h: "按联赛拆分",
    avb_compare_h: "🆚 audit vs backtest 对比 (点击展开 — 首次会跑 30-60s)",
    avb_running: "跑 backtest 中... 每个联赛 5-15 秒",
    avb_th_audit: "Audit (实战)",
    avb_th_backtest: "Backtest (回测)",
    avb_delta_legend_html: "Δ 解读:命中率<span class='font-mono'>&gt; +5pp</span> = backtest 明显比 audit 强 → 可能数据漂移;<span class='font-mono'>&lt; -5pp</span> = audit 反而比 backtest 强(多半是小样本噪音)。",
    avb_empty_html: "暂无可比较的联赛 — audit 需要单联赛 ≥3 个 resolved 才会出现在 <code>audit.by_league</code> 里。等定时 snapshot 多跑几天就有数据。",
    audit_best_calls_h: "✓ 最稳的命中(按胜平负)",
    audit_pred_label: "预测",
    audit_score_matched: "⚡ 比分对",
    audit_none_yet: "(还没有)",
    audit_worst_misses_h: "✗ 最大冷门 / 跌眼镜",
    audit_date_warn_tooltip: "预测时 fd.org 就提示这场比赛日期可能错位 — 不算模型的锅",
    audit_none_yet_2: "(还没有)",
    audit_reason_date_mismatch: "日期错位 (TheSportsDB 把日期标错了 — 同两队在 ±7 天内有真比赛)",
    audit_reason_no_db: "该日联赛无数据 (backfill 没覆盖到 / 配额没了)",
    audit_reason_name: "球队名对不上 (需要加 alias)",
    audit_reason_no_close: "同日有数据但球队不像同一场",
    audit_first_5_samples: "前 5 个样本",
    audit_actual_at: "→ 实际在",
    audit_exact_score_h: "⚡ 比分完全猜中",
    audit_model_prob_for_score: "模型给这个比分概率",
    backfill_recent_h: "最近一次回填",
    backfill_leagues_reached: "联赛触达",
    backfill_new_matches: "新增比赛",
    backfill_errors: "错误数",
    backfill_duration: "耗时",
    backfill_league_details: "联赛详情 (",
    backfill_no_source_collapsed: "跳过的 no_source 已折叠)",
    source_consistency_h: "数据源一致性",
    source_consistency_intro: "三个数据源 (football-data.co.uk / fd.org / api-football) 默认覆盖不同联赛、互不重叠。",
    source_consistency_intro2: "如果同一场比赛出现在多个源,下面会标出来;如果比分不一致,会按字段优先级 (fd.org &gt; api-football &gt; fd.co.uk) 选出\"权威\"版本。",
    source_cross_coverage: "跨源覆盖",
    source_score_conflicts: "比分冲突",
    source_conflicts_h: "⚠️ 比分冲突 (按字段优先级 resolve)",
    source_th_chosen: "采纳",
    source_status_conflict: "比分冲突",
    source_status_overlap: "有重叠",
    source_status_ok: "正常",
    manual_entry_h: "手动录入赛果",
    manual_entry_intro_p1: "fd.org / api-football / fd.co.uk 都没拉到,或者拉错了的比分,手动补一条。",
    manual_entry_intro_p2: "自动覆盖同一场比赛的其它源数据(按字段优先级)。",
    manual_entry_stage_opt: "阶段 (可选)",
    manual_entry_neutral_site: "中立场地",
    coverage_empty_league_short: "· 空联赛",
    cont_pick_match: "选择赛事",
    cont_home_label: "主队（中立场可视为 Team A）",
    cont_neutral: "中立场",
    cont_predict: "预测",
    cont_common_matchups: "常见对阵（点击填入）",
    cont_pick_first: "选择赛事和两支队伍，点「预测」",
    cont_home_advantage: "主场优势",
    cont_knockout_short: " · 淘汰赛",
    cont_advance_prob: "晋级概率（含加时+点球）",
    cont_default_filter: "— 全部 / 自动 —",
    inplay_current_score: "当前比分",
    inplay_kickoff: "开场 0'",
    inplay_halftime: "半场 45'",
    inplay_fulltime: "结束 90'",
    inplay_pick_first: "填入两队 + 当前比分 + 分钟，点「重新计算」",
    inplay_final_probs: "最终结果概率",
    inplay_expected_final: "预期终盘比分",
    inplay_state_adjust: "状态修正",
    inplay_state_legend: "落后方 ×1.15、领先方 ×0.92（基于多年实证）",
    inplay_top_final: "最可能的终盘比分",
    roi_total_staked: "总投注",
    roi_max_drawdown: "最大回撤",
    roi_bankroll_curve: "资金曲线",
    str_sort_overall: "综合（默认）",
    str_sort_attack: "进攻",
    str_sort_defense: "防守",
    str_sort_recent_gd: "近 5 场净胜球",
    str_sort_team: "球队名",
    str_fitting: "拟合 Dixon-Coles 中…",
    str_th_attack: "攻",
    bt_avg_stale: "平均过期",
    bt_baseline_ref: "基线参考:随机三选一=33%,\"永远选主\"=",
    bt_fit_failed_pts: "失败的拟合点 (前",
    bt_how_to_read_html: "Accuracy ≥ 53% = 接近盘口水平,可挖 value。48-52% = 比\"永远选主\"略好,模型还有调参空间。&lt; 48% = 数据太少或联赛方差太大(Tier 3 常见)。",
    bt_market_close: ",盘口收盘=53-55%。",
    bt_max_stale: "最大过期",
    bt_max_stale_hint: "场(最长一次预测时模型已 N 场没重训)",
    bt_max_stale_unit: "场",
    bt_xg_note_html: "xG 已经可以通过 <code>python predict.py update --include-xg</code> 接入;有 xG 的联赛会在拟合时混合实际进球和 xG。",
    cont_draw_share_adapted: "平局比例自适应到",
    cont_elo_away: "客",
    cont_elo_diff: "Elo 差",
    cont_elo_home: "主",
    cont_model_cross_league: "跨联赛 Dixon-Coles",
    cont_model_matches: "场比赛,",
    cont_model_merged_prefix: "合并",
    cont_model_merged_suffix: "个域内顶级联赛",
    cont_model_national_elo: "纯 Elo(国家队)",
    cont_model_national_elo_desc: "基于 eloratings.net 的国家队评级,配合自适应平局比例和加时点球的简单晋级概率。",
    cont_model_source_h: "模型来源",
    diag_ece_excellent: "< 0.04 优秀",
    diag_ece_ok: "0.04-0.08 一般",
    diag_ece_poor: "> 0.08 差",
    health_configured: "已配置",
    health_freshness_fresh: "新鲜",
    health_freshness_old: "历史",
    health_freshness_recent: "近期",
    health_freshness_stale: "滞后",
    health_not_configured: "未配置",
    inplay_caveat: "模型不感知红牌、伤停、换人战术调整、定位球数量等。是个数学基线,不是\"通灵预测\"。比如真实场景里 75 分钟领先 1-0 守不住的概率,要在这个数字上加几个百分点的\"人为因素\"。",
    roi_final_bankroll: "终值",
    roi_final_roi: "最终 ROI",
    str_th_defense: "防",
    str_th_overall: "综合",
    str_th_last5: "近 5 场",
    str_th_gd: "净胜",
    str_sort_overall_short: "综合",
    str_compare_hint: "点击左边表格选 2 支球队比较攻防。",
    str_compare_none: "未选择",
    str_compare_panel_h: "攻防对比",
    str_compare_legend: "绿=第一支队,紫=第二支队。宽度按强度绝对值归一。",
    replay_scan_all: "扫描整个联赛找惊喜",
    replay_scanning: "扫描中… 30-90 秒",
    replay_recent_30: "最近 30 场(点击复盘)",
    replay_pick_first: "左侧选一场比赛开始复盘",
    replay_retraining: "重训 Dixon-Coles 到比赛前一天…",
    replay_trained_prefix: "用",
    replay_trained_suffix: "场赛前数据训练",
    replay_model_correct: "模型预测正确",
    replay_model_wrong: "模型预测错误",
    replay_pre_match_probs: "赛前模型概率(粗体 = 实际结果)",
    replay_biggest_upsets_h: "最跌眼镜的 15 场",
    replay_biggest_upsets_sub: "模型说\"几乎不可能\",结果却发生了",
    replay_best_calls_h: "最神预测的 15 场",
    replay_best_calls_sub: "模型高信心、命中",
    roi_principal_label: "本金",
    diag_pick_run: "选择联赛点击「运行诊断」",
    ui_lang_zh_label: "中",
    value_legend_h: "看懂这些数字",
    value_legend_html: "<li><strong>边际</strong>:模型概率 − 盘口隐含概率。正数=模型觉得低估了。</li><li><strong>EV</strong>:每 1 元本金的期望回报。+5% = 长期平均赚 5 分钱。</li><li><strong>Kelly</strong>:理论最优仓位比例。<strong>实战推荐用 1/2 Kelly</strong>,因为模型概率本身也有误差。</li><li>本工具不下单不联通盘口,只是给参考。运气波动远比 EV 大。</li>",
    roi_known_result_html: "这个 Dixon-Coles 模型在本数据集所有联赛、所有阈值组合下,长期 ROI 都是<strong>负数</strong>。最好的 EPL 配置是 -3.6%。市场收盘赔率太效率,模型概率不够锐。",
    roi_why_lose_h: "为什么大概率亏?",
    roi_why_lose_p1_html: "Bet365 收盘赔率经过整个市场的「调价」过程,把模糊的信息都吸进去了。一个普通的 Dixon-Coles 模型校准虽好(ECE &lt; 0.03),但概率估计的<strong>锐度</strong>不够——当模型说\"主胜 70%\"时,在收盘赔口袋里,市场可能已经定价为 60-65%。模型以为找到了 5pp 的边际,其实只是在和「市场+vig」对赌。",
    roi_why_lose_p2_html: "要赚钱需要:<strong>对市场不知道的信息</strong>(伤停、阵容、天气)+ <strong>更精的概率模型</strong>(xG、玩家级、贝叶斯收缩)。这都不在本项目的当前范围。",
    match_most_likely_score: "最可能比分",
    match_xg_label: "预期进球（xG）",
    match_elo_label: "Elo 评级",
    match_top5_scores: "最可能比分（top 5）",
    match_score_matrix: "比分概率矩阵",
    match_matrix_corner: "主\\客",
    match_matrix_legend: "单位 %，行=主队进球，列=客队进球",
    h2h_draw_legend: "平",
    upcoming_fetch_first: "加载近期对阵...第一次较慢（含 fd.org 日期核对）",
    upcoming_scope_hint: "展示范围：欧洲顶级联赛 + 中日韩 + 沙特 + 美洲 + 洲际杯",
    wc_computing: "计算中…",
    wc_from_cache: "缓存命中",
    wc_instant: "即时计算",
    wc_th_team: "球队",
    wc_th_final: "决赛",
    wc_th_champion: "夺冠",
    wc_chart_title: "夺冠概率（top 10）",
    wc_run_hint: "点击「运行」开始模拟。首次约 5 秒，之后从缓存读取。",
    coverage_intro: "每个联赛的覆盖等级、数据状态 + 最近 30 场比赛。",
    coverage_th_data: "数据",
    coverage_pick_left: "左侧选择一个联赛",
    coverage_th_date: "日期",
    coverage_th_home: "主队",
    coverage_th_away: "客队",
    coverage_empty: "该联赛暂无数据。可能是 Tier 3 联赛——football-data.co.uk 没覆盖，需要 API-Football 拉取。",
    bt_real_dist: "真实结果分布",
    bt_how_to_read: "怎么读这个数字？",
    diag_auto_option: "— 自动 —",
    diag_n_hint: "50 通常 30 秒以内出结果。",
    diag_walking: "滚动回测 + 诊断中…",
    diag_curve_title: "校准曲线 · 预测概率 vs 实际发生率",
    diag_curve_legend: "虚线 y=x 是完美校准。点高于虚线=模型保守（实际比预测的更频繁）；点低于=模型过信。",
    diag_ladder_title: "信心阶梯 · 模型最高概率档的实际命中率",
    diag_ladder_th_band: "信心档",
    diag_ladder_th_hit: "实际命中",
    diag_ladder_th_pred: "平均预测",
    diag_ladder_th_bias: "偏差",
    diag_ladder_note: "红色行 = 高信心档过信（应该慎下注）。绿色行 = 模型反而保守，可挖 value。",
    value_odds_heading: "盘口赔率（十进制）",
    value_home_short: "主胜",
    value_away_short: "客胜",
    value_analyze: "分析价值",
    value_pick_hint: "填好两队 + 三个赔率，点击「分析价值」",
    value_compare_heading: "盘口隐含概率 vs 模型概率",
    value_edge_short: "边际",
    value_detected: "检测到价值",
    value_none: "无明显价值",
    value_kelly_advice: "Kelly 建议",
    value_no_value_note: "所有三个结果的盘口都比模型预测的更准或几乎相等。这通常是市场对的，看看下一场吧。",
    roi_min_edge_label: "最少边际 (pp)",
    roi_min_edge_hint2: "5pp = 模型概率比盘口隐含高 5 个百分点才下。",
    roi_min_ev_label: "最少 EV",
    roi_min_ev_hint2: "5% = 期望每元至少赚 5 分才下。",
    roi_kelly_label: "Kelly 系数",
    roi_kelly_hint2: "0.5 = 半 Kelly（默认，更稳）。",
    roi_model_label: "模型",
    roi_model_dc_default: "DC + Elo（默认）",
    roi_model_market: "Market-Fused（推荐：最稳）",
    roi_model_hint: "Market-Fused 在 La Liga / Primeira 实测亏损最小。",
    roi_run_label: "运行模拟",
    roi_running_label: "回测中…30-60 秒",
    roi_pick_first: "选择联赛与阈值，点「运行模拟」",
    roi_walking_label: "滚动回测 + 模拟下注中…",
    roi_known_result_h: "已知结果（剧透）",
    roi_n_bets: "下注次数",
    roi_win_rate: "命中率",
    inplay_recompute: "重新计算",
    strengths_compare: "两队对比",
    diag_run: "运行诊断",
    bt_league_auto: "— 自动（合并所有）—",
    bt_min_train: "最少训练样本",
    bt_min_train_hint: "越小越快但越早期的预测越噪。100-200 通常合理。",
    bt_refit_every: "每 N 场重新拟合",
    bt_refit_every_hint: "refit_every=1 最准但慢，10-25 兼顾速度。",
    bt_run: "运行回测",
    bt_running: "计算中…可能 30-60 秒",
    bt_pick_league: "选联赛后点「运行回测」。预计 30-60 秒。",
    bt_walking: "滚动训练 + 预测中…",
    bt_sample_count: "样本数",
    bt_score_metrics: "比分级别指标",
    bt_scored_count: "参与评分的预测",
    bt_score_acc_note: "几比几猜对的比例（足球普遍 8-12%）",
    bt_goal_dist_note: "|预测 H − 实际 H| + |预测 A − 实际 A|",
    bt_by_league_hint: "仅显示样本数 ≥ 30 的联赛 — 样本太小的数字不可靠。按 N 倒序。",
    bt_fit_health: "拟合健康度",
    bt_fit_attempts: "尝试拟合",
    bt_fit_success: "拟合成功",
    bt_fit_failures: "失败次数",
    bt_fit_fresh_pct: "新鲜拟合比例",
    bt_fit_fresh_note: "每次预测时正好刚 refit",
    // About tab (R43 hand-pass)
    about_para1: "球队层面的足球比赛预测器。Dixon-Coles 双变量泊松模型 + Elo 修正先验。数据源：football-data.co.uk（历史结果）、ClubElo（俱乐部 Elo）、eloratings.net（国家队 Elo）、可选 API-Football/FBref。",
    about_why_text: "这是足球本身的方差——单场比赛随机性极大。Closing 盘口的 3 分类预测准确率也就 53-55%。这个模型的目的是找<strong>值</strong>,不是确定性:当模型给出 60% 主胜而盘口隐含 45%,那才是信号。",
    about_coverage_heading: "数据覆盖等级",
    about_coverage_t1: "<strong>T1</strong>：欧洲主流联赛——结果 + ClubElo + xG（可选）。准确率 52-55%。",
    about_coverage_t2: "<strong>T2</strong>：英冠/英甲。结果 + ClubElo,无 xG。",
    about_coverage_t3: "<strong>T3</strong>：沙特/J1/K1/中超/MLS/巴甲/阿甲/墨超/葡甲——免费源覆盖不稳定,需要 API-Football key 才能拉。配置已就绪。",
    about_limits_heading: "不能做的",
    about_limit_1: "不能预测球员个人表现（需要付费球员数据）",
    about_limit_2: "不能感知伤停、换帅、阵容变化（评级有 4-6 场滞后）",
    about_limit_3: "不能稳定跑赢盘口",
    about_doctor_heading: "数据源诊断",
    about_doctor_schema_check: "检查",
    // R44 — prose paragraphs (x-html in HTML so inline <strong> renders)
    roi_tab_title: "ROI 验证 — \"价值发现\"真能赚钱吗?",
    roi_intro_p1: "上面的 <strong>价值发现</strong> tab 给出 EV +5%、+10% 听着很美好。这一页做的事很简单:<strong>过去几年我们如果真的按那套规则下注,钱包是涨还是缩?</strong>",
    roi_intro_p2: "走法:滚动训练 Dixon-Coles → 计算每场 EV → 满足 edge + EV 阈值就用半 Kelly 下注 → 用<strong>真实的 Bet365 收盘赔率</strong>结算 → 看终盘资金。",
    value_tab_title: "价值发现 — EV 和 Kelly 仓位建议",
    value_intro_p: "输入一场比赛的预测和你看到的盘口赔率,计算预期价值 (EV) 和仓位建议 (Kelly)。EV>0 说明模型认为这个赔率被低估了。",
    value_warning_html: "<strong>⚠ 重要:</strong>本模型在历史回测中长期亏损 (ROI 模拟器证明)。这个工具只是把模型自己的判断翻译成赌注语言,不代表能赚钱。",
    continental_tab_title: "洲际赛事 — UCL/Europa/Libertadores/AFC CL/...",
    continental_intro_p1: "欧冠 / 欧联 / 欧会杯 / 解放者杯 / 亚冠 + 国家队大赛(欧洲杯 / 美洲杯 / 亚洲杯 / 非洲杯)。",
    continental_intro_p2: "俱乐部赛事用<strong>跨联赛拟合</strong>(皇马 vs 拜仁 → 西甲+德甲+其他欧洲顶级联赛合并训练),国家队赛事直接用国家队 Elo。",
    inplay_tab_title: "实时进行中 — 重新计算最终结果概率",
    inplay_intro_p: "填入当前比分 + 当前分钟,重新计算最终结果概率。",
    inplay_math_explain: "数学上:先用赛前模型算 xG,然后剩余时间按状态修正(追分队 ×1.15,领先队 ×0.92),用泊松分布卷积剩余进球,加上当前比分。",
    strengths_tab_title: "球队强度 — 攻防参数 + 近 5 场战绩",
    strengths_intro_html: "基于 Dixon-Coles 拟合的攻防参数 + 近 5 场战绩。<strong>攻</strong> 越高 = 进球能力越强;<strong>防</strong> 越高 = 失球越少(已翻转,统一\"越高越好\")。",
    diag_tab_title: "模型诊断 — 三个深视角",
    diag_intro_html: "光看 accuracy 不够。这里给三个更深的视角:<strong>校准图</strong>(模型说 70% 实际真的 70% 吗?)、<strong>ECE</strong>(一行总结校准误差)、<strong>信心阶梯</strong>(高/中/低置信度各自命中率)。",
    replay_tab_title: "比赛复盘 — 模型当时怎么看",
    replay_intro_html: "点任何历史比赛,把模型倒回那场比赛<strong>之前</strong>(重新拟合),看它当时预测什么 vs 实际什么。下方两个榜单展示了模型<strong>看走眼</strong>和<strong>神预测</strong>的赛事,是 ROI 验证的可视化补充。",
    health_caches_heading: "磁盘缓存",
    health_caches_hint: "每个数据源的本地缓存目录。空的表示这个源没启用或没缓存。",
    health_th_cache: "缓存",
    health_th_files: "文件数",
    health_th_size: "总大小",
    health_th_newest_file: "最新文件",
    health_th_path: "路径",
    audit_heading: "实测准确率（已结果的 /upcoming 预测）",
    audit_intro: "把\"本周赛程\"里历史预测和实际比分对账。这是真实场景下的准确率，比合成回测更诚实。",
    audit_empty: "暂无已结果的预测。\"本周赛程\"里有预测过的比赛踢完之后会自动结算。",
    audit_resolved_count: "已结果",
    audit_wdl_hit_rate: "胜平负命中率",
    audit_n_scored: "已记预测比分",
    audit_n_scored_note: "(round 18 之后才开始记)",
    audit_score_hit_rate: "比分命中率",
    audit_score_hit_note: "猜对几比几的比例（足球普遍 8-12%）",
    audit_goal_distance: "进球差均值",
    audit_per_league: "按联赛拆分",
    audit_per_league_note: "(同口径于 /backtest 的 by_league — 可直接比对)",
    audit_th_wdl: "胜平负",
    audit_th_score: "比分",
    audit_th_goal_dist: "进球差",
    health_loading: "加载中…",
    health_total_matches: "比赛总数",
    health_leagues_card: "联赛",
    health_sources_card: "数据源",
    health_fresh_card: "≤7 天新",
    health_per_league: "每联赛状态",
    health_per_source: "每数据源",
    health_th_matches: "场数",
    health_th_earliest: "最早",
    health_th_latest: "最新",
    health_th_days_stale: "旧 (天)",
    health_th_primary_src: "主源",
    health_th_status: "状态",
    health_th_source: "源",
    health_th_latest_match: "最新比赛",
    health_th_service: "服务",
    health_register: "注册",
    health_sort_hint: "按\"新鲜度\"排序，最旧的在底部",
    model_ensemble_label: "Ensemble（3 模型平均）",
    model_dc_pblog_label: "Dixon-Coles（penaltyblog）",
    worldcup_title: "世界杯 2026 蒙特卡洛",
    worldcup_subtitle: "48 队，国家队 Elo 驱动。模拟整个分组+淘汰阶段。",
    worldcup_n_sims: "模拟次数",
    worldcup_sims_fast: "1,000（快）",
    worldcup_sims_default: "5,000（默认）",
    worldcup_sims_precise: "20,000（精）",
    health_title: "数据健康仪表盘",
    health_subtitle: "每个联赛 + 数据源 + API key + 缓存的当前状态。",
    about_why_not_high_heading: "为什么准确率不会高于 55%",
    leagues_filter_label: "联赛",
    leagues_tier_label: "档",
    strengths_sort_label: "排序",
    accuracy: "Accuracy",
    rps: "RPS",
    brier: "Brier",
    log_loss: "Log loss",
    lower_better: "越低越好",
    // replay
    replay_subtitle: "点任何历史比赛，把模型倒回那场比赛之前重新拟合，看它当时预测什么 vs 实际什么。",
    replay_recent: "最近 30 场（点击复盘）",
    replay_choose: "左侧选一场比赛开始复盘",
    replay_correct: "✓ 模型预测正确",
    replay_wrong: "✗ 模型预测错误",
    replay_scan: "扫描整个联赛找惊喜",
    replay_scanning: "扫描中… 30-90 秒",
    replay_upsets: "最跌眼镜的 15 场",
    replay_best: "最神预测的 15 场",
    // strengths
    strengths_subtitle: "基于 Dixon-Coles 拟合的攻防参数 + 近 5 场战绩。攻 越高 = 进球能力越强；防 越高 = 失球越少（已翻转，统一\"越高越好\"）。",
    strengths_sort: "排序",
    sort_overall: "综合（默认）",
    sort_attack: "进攻",
    sort_defense: "防守",
    sort_recent_gd: "近 5 场净胜球",
    sort_elo: "ClubElo",
    sort_team: "球队名",
    home_advantage: "主场优势",
    compare_two_teams: "两队对比",
    attack: "攻",
    defense: "防",
    overall: "综合",
    last5: "近 5 场",
    net_gd: "净胜",
  },
  en: {
    // tabs
    tab_match: "Match",
    tab_worldcup: "World Cup 2026",
    tab_leagues: "Leagues",
    tab_backtest: "Backtest",
    tab_diagnostics: "Diagnostics",
    tab_value: "Value Finder",
    tab_roi: "ROI Audit",
    tab_continental: "Continental",
    tab_inplay: "Live In-Play",
    tab_strengths: "Strengths",
    tab_replay: "Replay",
    tab_upcoming: "This Week",
    tab_health: "Data Health",
    tab_about: "About",
    // common
    home_team: "Home",
    away_team: "Away",
    league: "League",
    model: "Model",
    home_win: "Home Win",
    draw: "Draw",
    away_win: "Away Win",
    predict: "Predict",
    calculating: "Calculating…",
    run: "Run",
    refresh: "Refresh",
    neutral_site: "Neutral site",
    knockout: "Knockout",
    expected_goals: "Expected Goals (xG)",
    training_rows: "Training rows",
    most_likely_scores: "Most likely scores (top 5)",
    score_matrix: "Score probability matrix",
    leagues_count: "leagues",
    matches_count: "matches",
    teams_count: "teams",
    online: "Online",
    offline: "Offline",
    // tab subtitles
    match_subtitle: "Dixon-Coles + Elo correction. Home/Draw/Away probabilities, expected goals, and the most likely final scorelines.",
    choose_teams: "Pick two teams, then click Predict",
    all_auto: "— All / Auto —",
    manual_predict_heading: "Predict a single match",
    upcoming_board_heading: "Upcoming matches · live predictions",
    more_options: "More options",
    featured_only_note: "Featured leagues only · fixtures with complete predictions",
    // /upcoming tab strings (R42 i18n pass)
    upcoming_title: "Upcoming fixtures + live predictions",
    upcoming_subtitle: "Pulls real fixtures from TheSportsDB (free, no API key) for the next few days. Each match gets a fresh Dixon-Coles + Elo prediction. Covers CSL, J1, MLS, Europe's top 5, Championship, Belgian, Eredivisie.",
    days_ahead_label: "Window",
    days_today_plus_3: "Today + 3 days",
    days_next_7: "Next 7 days",
    days_next_14: "Next 14 days",
    days_next_30: "Next 30 days",
    upcoming_loading: "Loading…",
    fixtures_count_suffix: "fixtures · source TheSportsDB",
    upcoming_loading_detail: "Fetching fixtures + running model per match… 10–30 seconds",
    upcoming_not_loaded: "Haven't fetched fixtures yet.",
    upcoming_load_now: "Fetch now",
    upcoming_empty_prefix: "TheSportsDB returned no fixtures in the next",
    upcoming_empty_suffix: "days.",
    upcoming_empty_hint: "Free tier usually returns 1-2 fixtures per league. Try widening the window to 14 or 30 days.",
    upcoming_try_30: "Retry with 30 days",
    alt_scores_label: "Alt",
    total_goals_label: "Goals",
    delta_h_ago: "h ago",
    upcoming_howto_heading: "How to use",
    upcoming_howto_1: "Fixtures come from TheSportsDB (free, no API key), so each league usually returns just 1–2 of the most recent matches",
    upcoming_howto_2: "Each prediction uses a pre-match-trained Dixon-Coles + Elo model. n= is training sample size",
    upcoming_howto_3: 'Status "Not Started" / "P" = upcoming; "Match Finished" = played (you can compare prediction vs. actual)',
    upcoming_howto_4: "Cached for 6 hours to avoid hammering TheSportsDB",
    metadata_only_warn: "⚠ Prediction unavailable",
    metadata_only_fallback_detail: "No historical data for this league",
    h2h_other_leagues: "Previous meetings (history from other competitions)",
    draw_short: "Draw",
    date_warn_title_prefix: "fd.org has this match on",
    date_confirmed_title: "fd.org also lists this match on this date",
    market_letter: "Mkt",
    model_word: "Model",
    market_implied: "Market implied",
    bet_rate_suffix: "bet rate",
    roi_red_zone_a: "Red zone = bankroll below starting",
    roi_red_zone_b: " level.",
    club_short: "club",
    national_short: "national",
    cont_pick_match_placeholder: "Please pick a match",
    inplay_current_minute: "Current minute",
    remaining_word: "Remaining",
    minutes_unit: "minutes",
    pre_match_was: "Pre-match:",
    home_win_short: "Home",
    away_win_short: "Away",
    draw_word: "Draw",
    train_samples: "Training samples",
    h2h_recent_prefix: "Last",
    h2h_recent_suffix: "head-to-head",
    perspective: "perspective",
    venue_home: "Home",
    venue_away: "Away",
    h2h_win_short: "W",
    h2h_draw_short: "D",
    h2h_loss_short: "L",
    matches_unit: "matches",
    days_unit: "days",
    upcoming_summary_a: "Predicted",
    upcoming_summary_b: "of",
    upcoming_summary_c: "future",
    no_prediction: "No prediction",
    upcoming_no_pred_a: "Over the next",
    upcoming_no_pred_b: "days, no covered-league fixtures to predict",
    view_all_prefix: "View all",
    view_all_with_minor: "matches (incl. minor leagues)",
    wc_n_sims_suffix: "simulations",
    coverage_history_matches: "matches in history",
    latest_label: "Latest",
    no_data: "No data",
    bt_train_set: "Training set",
    value_model_letter: "M",
    inplay_away_letter: "A",
    audit_of_which_prefix: "of which",
    audit_of_which_suffix: "had a date warning at prediction time",
    load_failed: "Load failed",
    avb_compared_prefix: "Compared",
    avb_compared_suffix: "leagues",
    avb_skipped_prefix: "skipped",
    avb_skipped_reason: "insufficient backtest data",
    diff_prefix: "off by",
    balls_unit: "goals",
    predicted_label: "Predicted",
    unmatched_warn_a: "still",
    unmatched_warn_b: "predictions don't match",
    unmatched_warn_c: "in the last 30 days",
    unmatched_warn_d: "do match",
    venues_flipped: "venues swapped",
    nearby_candidate_label: "Nearby candidate",
    suggested_alias: "Suggested alias",
    backfill_window: "Window",
    backfill_done_at: "Finished",
    overlap_no_conflict: "Cross-source overlaps with consistent scores",
    total_short: "total",
    collapse_action: "Collapse",
    expand_action: "Expand",
    submitting: "Submitting…",
    submit: "Submit",
    snapshot_time: "Snapshot time",
    status_ready: "Ready",
    status_sparse: "Sparse",
    status_empty: "Missing data",
    status_unknown: "Unknown",
    bt_subtitle_line1: "Walk-forward: train on prior matches only, predict the current one — measures real generalization.",
    bt_subtitle_line2: "~30 seconds per league. Accuracy = 1X2 hit rate. Brier ↓ better. Log loss ↓ better.",
    bt_fit_failed_pts_2: ")",
    h2h_win_legend: "W",
    h2h_loss_legend: "L",
    manual_entry_league_ph: "Premier League / premier_league / EPL",
    api_endpoints_listing: "POST /predict     Single-match prediction\nPOST /backtest    Walk-forward backtest\nPOST /update      Background incremental update\nGET  /doctor      Data-source diagnostics\nGET  /worldcup/forecast   World Cup simulation\nGET  /teams       Team list\nGET  /leagues     League list\nGET  /coverage    League coverage\nGET  /recent      Recent matches\nGET  /stats       Database stats\nGET  /health      Health check\nGET  /api-football/leagues  API-Football league ID lookup\nGET  /export/{matches|ratings|players|player_season_stats|update_state}  CSV export",
    footer_latest_data: "Latest data",
    match_score_prob_inline: "Model probability for this score",
    upcoming_click_to_expand_title: "Click to expand the full prediction",
    bt_fit_all_ok_label: "All ok",
    bt_fit_high_fail_label: "High failure rate",
    bt_fit_few_fail_label: "Some failures",
    value_dot_edge: " · edge",
    value_half_kelly_inline: "(half-Kelly is safer:",
    cont_knockout_opt: "Knockout",
    cont_neutral_opt: "Neutral",
    health_30_days_old: ">30 days stale",
    audit_date_mismatch_tooltip: "fd.org already flagged these matches' dates as possibly off — not the model's fault",
    audit_per_league_h: "Per league",
    avb_compare_h: "🆚 Audit vs backtest comparison (click to expand — first run takes 30–60s)",
    avb_running: "Running backtest… 5–15 seconds per league",
    avb_th_audit: "Audit (live)",
    avb_th_backtest: "Backtest (walk-forward)",
    avb_delta_legend_html: "Δ reading: hit rate <span class='font-mono'>&gt; +5pp</span> = backtest much better than audit → possible data drift; <span class='font-mono'>&lt; -5pp</span> = audit better than backtest (usually small-sample noise).",
    avb_empty_html: "No comparable leagues yet — audit needs ≥3 resolved per league to appear in <code>audit.by_league</code>. Wait a few more days for scheduled snapshots to fill in.",
    audit_best_calls_h: "✓ Most consistent hits (by W/D/L)",
    audit_pred_label: "Predicted",
    audit_score_matched: "⚡ Score matched",
    audit_none_yet: "(none yet)",
    audit_worst_misses_h: "✗ Biggest upsets / misses",
    audit_date_warn_tooltip: "fd.org flagged this match's date as possibly off at prediction time — not the model's fault",
    audit_none_yet_2: "(none yet)",
    audit_reason_date_mismatch: "Date mismatch (TheSportsDB had the wrong date — same teams have a real match within ±7 days)",
    audit_reason_no_db: "No league data on this date (backfill didn't cover it / quota exhausted)",
    audit_reason_name: "Team names don't match (alias needed)",
    audit_reason_no_close: "Data exists for the date but the teams don't match",
    audit_first_5_samples: "First 5 samples",
    audit_actual_at: "→ actually on",
    audit_exact_score_h: "⚡ Exact-score hits",
    audit_model_prob_for_score: "Model probability for this score",
    backfill_recent_h: "Most recent backfill",
    backfill_leagues_reached: "Leagues reached",
    backfill_new_matches: "New matches",
    backfill_errors: "Errors",
    backfill_duration: "Duration",
    backfill_league_details: "League details (",
    backfill_no_source_collapsed: "no_source skips collapsed)",
    source_consistency_h: "Source consistency",
    source_consistency_intro: "The three data sources (football-data.co.uk / fd.org / api-football) cover different leagues by default and don't overlap.",
    source_consistency_intro2: "When the same match appears in multiple sources, it's flagged below. Score conflicts are resolved by field priority (fd.org &gt; api-football &gt; fd.co.uk) to pick the \"authoritative\" version.",
    source_cross_coverage: "Cross-source coverage",
    source_score_conflicts: "Score conflicts",
    source_conflicts_h: "⚠️ Score conflicts (resolved by field priority)",
    source_th_chosen: "Chosen",
    source_status_conflict: "Score conflict",
    source_status_overlap: "Overlap",
    source_status_ok: "OK",
    manual_entry_h: "Manual result entry",
    manual_entry_intro_p1: "When fd.org / api-football / fd.co.uk all missed it (or got it wrong), add a result by hand.",
    manual_entry_intro_p2: "Automatically overrides other sources' data for the same match (by field priority).",
    manual_entry_stage_opt: "Stage (optional)",
    manual_entry_neutral_site: "Neutral venue",
    coverage_empty_league_short: "· empty league",
    cont_pick_match: "Pick a match",
    cont_home_label: "Home (treat as Team A if neutral)",
    cont_neutral: "Neutral",
    cont_predict: "Predict",
    cont_common_matchups: "Common matchups (click to fill)",
    cont_pick_first: "Pick a match and two teams, then click Predict.",
    cont_home_advantage: "Home advantage",
    cont_knockout_short: " · knockout",
    cont_advance_prob: "Advance probability (incl. ET + pens)",
    cont_default_filter: "— All / Auto —",
    inplay_current_score: "Current score",
    inplay_kickoff: "Kickoff 0'",
    inplay_halftime: "Half 45'",
    inplay_fulltime: "End 90'",
    inplay_pick_first: "Enter both teams + current score + minute, then click Recompute.",
    inplay_final_probs: "Final-result probabilities",
    inplay_expected_final: "Expected final score",
    inplay_state_adjust: "Game-state adjustment",
    inplay_state_legend: "Trailing side ×1.15, leading side ×0.92 (based on multi-year empirical evidence)",
    inplay_top_final: "Most likely final scores",
    roi_total_staked: "Total staked",
    roi_max_drawdown: "Max drawdown",
    roi_bankroll_curve: "Bankroll curve",
    str_sort_overall: "Overall (default)",
    str_sort_attack: "Attack",
    str_sort_defense: "Defense",
    str_sort_recent_gd: "Last-5 GD",
    str_sort_team: "Team name",
    str_fitting: "Fitting Dixon-Coles…",
    str_th_attack: "Atk",
    bt_avg_stale: "Mean staleness",
    bt_baseline_ref: "Baselines: random 1-of-3 = 33%; \"always pick home\" =",
    bt_fit_failed_pts: "Failed fit points (first",
    bt_how_to_read_html: "Accuracy ≥ 53% = near closing-odds level; value-bet candidates exist. 48–52% = slightly better than always-home; room to tune. &lt; 48% = too little data or high variance (common for Tier 3).",
    bt_market_close: "; closing odds ≈ 53–55%.",
    bt_max_stale: "Max staleness",
    bt_max_stale_hint: "matches (longest gap between refit and the prediction)",
    bt_max_stale_unit: "matches",
    bt_xg_note_html: "xG can be ingested via <code>python predict.py update --include-xg</code>; leagues with xG mix actual goals + xG during fit.",
    cont_draw_share_adapted: "draw rate adapts to",
    cont_elo_away: "away",
    cont_elo_diff: "Elo diff",
    cont_elo_home: "home",
    cont_model_cross_league: "Cross-league Dixon-Coles",
    cont_model_matches: "matches,",
    cont_model_merged_prefix: "merged",
    cont_model_merged_suffix: "top leagues from the region",
    cont_model_national_elo: "Pure Elo (national teams)",
    cont_model_national_elo_desc: "Based on eloratings.net national-team ratings, with an adaptive draw rate plus a simple ET/penalties advance model.",
    cont_model_source_h: "Model source",
    diag_ece_excellent: "< 0.04 excellent",
    diag_ece_ok: "0.04–0.08 ok",
    diag_ece_poor: "> 0.08 poor",
    health_configured: "Configured",
    health_freshness_fresh: "Fresh",
    health_freshness_old: "Historical",
    health_freshness_recent: "Recent",
    health_freshness_stale: "Stale",
    health_not_configured: "Not configured",
    inplay_caveat: "The model doesn't see red cards, injuries, tactical subs, or set-piece counts. It's a math baseline, not a 'psychic prediction'. For real-world scenarios — like the chance a 1-0 lead at 75' is blown — add a few percentage points of 'human factor' on top.",
    roi_final_bankroll: "Final bankroll",
    roi_final_roi: "Final ROI",
    str_th_defense: "Def",
    str_th_overall: "Overall",
    str_th_last5: "Last 5",
    str_th_gd: "GD",
    str_sort_overall_short: "Overall",
    str_compare_hint: "Click rows in the left table to pick 2 teams to compare.",
    str_compare_none: "None selected",
    str_compare_panel_h: "Attack/defense comparison",
    str_compare_legend: "Green = team 1, purple = team 2. Width normalized by absolute strength.",
    replay_scan_all: "Scan the league for surprises",
    replay_scanning: "Scanning… 30–90 seconds",
    replay_recent_30: "Last 30 matches (click to replay)",
    replay_pick_first: "Pick a match on the left to start the replay",
    replay_retraining: "Retraining Dixon-Coles to the day before…",
    replay_trained_prefix: "Trained on",
    replay_trained_suffix: "pre-match rows",
    replay_model_correct: "Model predicted correctly",
    replay_model_wrong: "Model predicted wrong",
    replay_pre_match_probs: "Pre-match model probability (bold = actual outcome)",
    replay_biggest_upsets_h: "Top 15 biggest upsets",
    replay_biggest_upsets_sub: "Model said \"nearly impossible\" — and yet it happened",
    replay_best_calls_h: "Top 15 best calls",
    replay_best_calls_sub: "High-confidence prediction, hit",
    roi_principal_label: "Principal",
    diag_pick_run: "Pick a league, then click Run diagnostics.",
    ui_lang_zh_label: "中",
    value_legend_h: "How to read these numbers",
    value_legend_html: "<li><strong>Edge</strong>: model probability − implied probability. Positive = model thinks the line is mispriced.</li><li><strong>EV</strong>: expected return per 1 unit staked. +5% = average +0.05 per unit over the long run.</li><li><strong>Kelly</strong>: theoretical optimal stake fraction. <strong>In practice, use 1/2 Kelly</strong> — the model probability itself has error.</li><li>This tool does not place bets and does not connect to any bookmaker. Numbers are guidance only. Variance dwarfs EV.</li>",
    roi_known_result_html: "This Dixon-Coles model produces a <strong>negative</strong> long-run ROI across every league and threshold combination in our dataset. The best configuration (EPL) lands at -3.6%. Closing odds are too efficient and the model's probabilities aren't sharp enough.",
    roi_why_lose_h: "Why is this likely to lose?",
    roi_why_lose_p1_html: "Bet365's closing odds absorb the entire market's price-discovery process — fuzzy private information ends up in the line. A vanilla Dixon-Coles model is well-calibrated (ECE &lt; 0.03), but the <strong>sharpness</strong> of its probabilities isn't enough — when the model says \"home 70%\", the market may already have it at 60–65%. The 5pp \"edge\" the model thinks it sees is really just a bet against \"market + vig\".",
    roi_why_lose_p2_html: "Profitable betting requires: <strong>information the market doesn't have</strong> (injuries, lineups, weather) + <strong>sharper probability models</strong> (xG, player-level, Bayesian shrinkage). Neither is in scope for this project.",
    match_most_likely_score: "Most likely score",
    match_xg_label: "Expected goals (xG)",
    match_elo_label: "Elo rating",
    match_top5_scores: "Most likely scores (top 5)",
    match_score_matrix: "Score probability matrix",
    match_matrix_corner: "H\\A",
    match_matrix_legend: "Values in %. Rows = home goals, columns = away goals.",
    h2h_draw_legend: "D",
    upcoming_fetch_first: "Loading upcoming fixtures… first load is slower (cross-checks fd.org dates)",
    upcoming_scope_hint: "Scope: top European leagues + East Asia + Saudi + Americas + continental cups",
    wc_computing: "Computing…",
    wc_from_cache: "Cache hit",
    wc_instant: "Just computed",
    wc_th_team: "Team",
    wc_th_final: "Final",
    wc_th_champion: "Champion",
    wc_chart_title: "Champion probability (top 10)",
    wc_run_hint: "Click Run to start the simulation. First run ~5s, then served from cache.",
    coverage_intro: "Per-league coverage tier, data state, and the last 30 matches.",
    coverage_th_data: "Data",
    coverage_pick_left: "Pick a league on the left",
    coverage_th_date: "Date",
    coverage_th_home: "Home",
    coverage_th_away: "Away",
    coverage_empty: "No data for this league yet. Likely a Tier 3 league — not covered by football-data.co.uk, needs API-Football.",
    bt_real_dist: "Actual outcome distribution",
    bt_how_to_read: "How to read this",
    diag_auto_option: "— Auto —",
    diag_n_hint: "n=50 usually completes within 30 seconds.",
    diag_walking: "Walk-forward + diagnostics…",
    diag_curve_title: "Calibration curve · predicted vs actual frequency",
    diag_curve_legend: "The dashed y=x line is perfect calibration. Above it = model conservative (reality more frequent); below = over-confident.",
    diag_ladder_title: "Confidence ladder · realized hit rate per top-probability band",
    diag_ladder_th_band: "Band",
    diag_ladder_th_hit: "Realized hit",
    diag_ladder_th_pred: "Mean predicted",
    diag_ladder_th_bias: "Bias",
    diag_ladder_note: "Red row = high-confidence band over-confident (bet cautiously). Green row = model is conservative; value-bet candidate.",
    value_odds_heading: "Bookmaker odds (decimal)",
    value_home_short: "Home",
    value_away_short: "Away",
    value_analyze: "Analyze value",
    value_pick_hint: "Fill in both teams + three odds, then click Analyze.",
    value_compare_heading: "Implied probability vs model probability",
    value_edge_short: "Edge",
    value_detected: "Value detected",
    value_none: "No clear value",
    value_kelly_advice: "Kelly suggestion",
    value_no_value_note: "On all three outcomes the bookmaker's implied probabilities match or beat the model. Usually means the market is right — try another match.",
    roi_min_edge_label: "Min edge (pp)",
    roi_min_edge_hint2: "5pp = only bet when model prob ≥ implied prob + 5 percentage points.",
    roi_min_ev_label: "Min EV",
    roi_min_ev_hint2: "5% = only bet when EV ≥ 5%.",
    roi_kelly_label: "Kelly factor",
    roi_kelly_hint2: "0.5 = half-Kelly (default, safer).",
    roi_model_label: "Model",
    roi_model_dc_default: "DC + Elo (default)",
    roi_model_market: "Market-Fused (recommended: safest)",
    roi_model_hint: "Market-Fused has the smallest realized loss on La Liga / Primeira.",
    roi_run_label: "Run simulation",
    roi_running_label: "Backtesting… 30–60 seconds",
    roi_pick_first: "Pick a league + thresholds, then click Run.",
    roi_walking_label: "Walk-forward + simulated betting…",
    roi_known_result_h: "Known result (spoiler)",
    roi_n_bets: "Bets placed",
    roi_win_rate: "Hit rate",
    inplay_recompute: "Recompute",
    strengths_compare: "Compare two teams",
    diag_run: "Run diagnostics",
    bt_league_auto: "— Auto (all merged) —",
    bt_min_train: "Min training matches",
    bt_min_train_hint: "Smaller = faster but earlier predictions are noisier. 100-200 is usually reasonable.",
    bt_refit_every: "Refit every N matches",
    bt_refit_every_hint: "refit_every=1 is most accurate but slow. 10-25 balances speed.",
    bt_run: "Run backtest",
    bt_running: "Computing… 30-60 seconds",
    bt_pick_league: "Pick a league, then click Run. Expect 30-60 seconds.",
    bt_walking: "Walk-forward training + predicting…",
    bt_sample_count: "Sample size",
    bt_score_metrics: "Score-level metrics",
    bt_scored_count: "Scored predictions",
    bt_score_acc_note: "Fraction of exact-score predictions correct (football typically 8-12%)",
    bt_goal_dist_note: "|predicted H − actual H| + |predicted A − actual A|",
    bt_by_league_hint: "Only leagues with ≥ 30 samples shown — smaller samples are unreliable. Sorted by N desc.",
    bt_fit_health: "Fit health",
    bt_fit_attempts: "Fit attempts",
    bt_fit_success: "Successful fits",
    bt_fit_failures: "Failures",
    bt_fit_fresh_pct: "Fresh-fit %",
    bt_fit_fresh_note: "Predictions made on the same iteration as a refit",
    // About tab (R43 hand-pass)
    about_para1: "Team-level football match predictor. Dixon-Coles bivariate Poisson + Elo prior correction. Data sources: football-data.co.uk (historical results), ClubElo (club Elo), eloratings.net (national Elo), optional API-Football/FBref.",
    about_why_text: "This is football's intrinsic variance — single matches are highly random. Closing odds' 3-way accuracy peaks around 53-55%. This model isn't aiming for certainty, it's aiming for <strong>value</strong>: when the model says 60% home win and the market implies 45%, that's the signal.",
    about_coverage_heading: "Data coverage tiers",
    about_coverage_t1: "<strong>T1</strong>: top European leagues — results + ClubElo + xG (optional). Accuracy 52-55%.",
    about_coverage_t2: "<strong>T2</strong>: Championship / League One. Results + ClubElo, no xG.",
    about_coverage_t3: "<strong>T3</strong>: Saudi/J1/K1/CSL/MLS/Brazil/Argentina/Liga MX/Liga Portugal 2 — free sources unreliable, requires API-Football key. Configured.",
    about_limits_heading: "What it can't do",
    about_limit_1: "Cannot predict individual player performance (needs paid player data)",
    about_limit_2: "Cannot perceive injuries, manager changes, or lineup shifts (ratings lag 4-6 matches)",
    about_limit_3: "Cannot consistently beat the market",
    about_doctor_heading: "Data source diagnostics",
    about_doctor_schema_check: "Check",
    // R44 — prose paragraphs (English mirrors)
    roi_tab_title: "ROI Audit — does value betting actually make money?",
    roi_intro_p1: "The <strong>Value Finder</strong> tab spits out EV +5%, +10% which sounds great in theory. This page does one thing: <strong>if we had actually bet on those rules over the past few years, would the bankroll go up or down?</strong>",
    roi_intro_p2: "How it works: walk-forward train Dixon-Coles → compute per-match EV → bet half-Kelly when edge + EV thresholds are met → settle with <strong>real Bet365 closing odds</strong> → see final bankroll.",
    value_tab_title: "Value Finder — EV and Kelly stake suggestions",
    value_intro_p: "Enter a match prediction and the bookmaker's odds. Computes expected value (EV) and Kelly stake. EV>0 means the model sees value at those odds.",
    value_warning_html: "<strong>⚠ Important:</strong> this model loses money long-run in historical backtests (proven by the ROI simulator). This tool just translates the model's view into betting language — it does NOT mean you'll make money.",
    continental_tab_title: "Continental — UCL/Europa/Libertadores/AFC CL/...",
    continental_intro_p1: "UCL / Europa / Conference / Copa Libertadores / AFC Champions League + national-team tournaments (Euro / Copa América / Asian Cup / AFCON).",
    continental_intro_p2: "Club competitions use <strong>cross-league fitting</strong> (Real Madrid vs Bayern → trained on combined La Liga + Bundesliga + other top European leagues). National tournaments use national-team Elo directly.",
    inplay_tab_title: "Live in-play — recompute final-result probabilities",
    inplay_intro_p: "Enter the current score + minute. Recomputes the probabilities of the final result.",
    inplay_math_explain: "Math: pre-match xG → remaining-time state correction (chasing team ×1.15, leading team ×0.92) → Poisson convolution of remaining goals → add current score.",
    strengths_tab_title: "Strengths — attack/defense parameters + last-5 form",
    strengths_intro_html: "Attack/defense parameters from the Dixon-Coles fit + last-5-match form. <strong>Attack</strong> higher = better at scoring; <strong>Defense</strong> higher = fewer goals conceded (inverted, so higher is always better).",
    diag_tab_title: "Diagnostics — three deeper views",
    diag_intro_html: "Accuracy alone isn't enough. Three deeper views: <strong>calibration curve</strong> (when the model says 70%, is it really 70%?), <strong>ECE</strong> (one-line calibration error), <strong>confidence ladder</strong> (hit rate by high/mid/low confidence).",
    replay_tab_title: "Replay — what did the model see at the time?",
    replay_intro_html: "Click any past match. The model rewinds to <strong>before</strong> that match (re-fits) and shows what it predicted vs what actually happened. The two leaderboards below show the model's <strong>biggest misses</strong> and <strong>best calls</strong> — a visual companion to the ROI audit.",
    health_caches_heading: "Disk caches",
    health_caches_hint: "Per-source on-disk cache directories. Empty rows mean the source isn't configured or hasn't cached anything yet.",
    health_th_cache: "Cache",
    health_th_files: "Files",
    health_th_size: "Total size",
    health_th_newest_file: "Newest file",
    health_th_path: "Path",
    audit_heading: "Realized accuracy (resolved /upcoming predictions)",
    audit_intro: "Reconciles past /upcoming predictions with actual scores. This is real-world accuracy — more honest than synthetic backtests.",
    audit_empty: "No resolved predictions yet. Fixtures predicted on the 'Upcoming' tab will be reconciled once they finish.",
    audit_resolved_count: "Resolved",
    audit_wdl_hit_rate: "W/D/L accuracy",
    audit_n_scored: "Recorded score predictions",
    audit_n_scored_note: "(recorded since round 18)",
    audit_score_hit_rate: "Exact-score accuracy",
    audit_score_hit_note: "Fraction of exact-score predictions correct (football typically 8-12%)",
    audit_goal_distance: "Mean goal distance",
    audit_per_league: "Per league",
    audit_per_league_note: "(same shape as /backtest's by_league — directly comparable)",
    audit_th_wdl: "W/D/L",
    audit_th_score: "Score",
    audit_th_goal_dist: "Goal distance",
    health_loading: "Loading…",
    health_total_matches: "Total matches",
    health_leagues_card: "Leagues",
    health_sources_card: "Sources",
    health_fresh_card: "≤7 days fresh",
    health_per_league: "Per-league status",
    health_per_source: "Per source",
    health_th_matches: "Matches",
    health_th_earliest: "Earliest",
    health_th_latest: "Latest",
    health_th_days_stale: "Stale (days)",
    health_th_primary_src: "Primary source",
    health_th_status: "Status",
    health_th_source: "Source",
    health_th_latest_match: "Latest match",
    health_th_service: "Service",
    health_register: "Sign up",
    health_sort_hint: "Sorted by freshness; oldest at the bottom",
    model_ensemble_label: "Ensemble (3-model avg)",
    model_dc_pblog_label: "Dixon-Coles (penaltyblog)",
    worldcup_title: "World Cup 2026 Monte Carlo",
    worldcup_subtitle: "48 teams, driven by national-team Elo. Simulates the full group + knockout phase.",
    worldcup_n_sims: "Simulations",
    worldcup_sims_fast: "1,000 (fast)",
    worldcup_sims_default: "5,000 (default)",
    worldcup_sims_precise: "20,000 (precise)",
    health_title: "Data Health Dashboard",
    health_subtitle: "Current status of each league + data source + API key + cache.",
    about_why_not_high_heading: "Why accuracy won't exceed 55%",
    leagues_filter_label: "League",
    leagues_tier_label: "Tier",
    strengths_sort_label: "Sort by",
    accuracy: "Accuracy",
    rps: "RPS",
    brier: "Brier",
    log_loss: "Log loss",
    lower_better: "Lower is better",
    // replay
    replay_subtitle: "Click any past match — the model rewinds, refits only on prior data, and shows what it would have predicted vs what actually happened.",
    replay_recent: "30 most recent matches (click to replay)",
    replay_choose: "Pick a match on the left to start the replay",
    replay_correct: "✓ Model called it correctly",
    replay_wrong: "✗ Model got it wrong",
    replay_scan: "Scan league for surprises",
    replay_scanning: "Scanning… 30-90s",
    replay_upsets: "15 biggest upsets",
    replay_best: "15 sharpest calls",
    // strengths
    strengths_subtitle: "Per-team attack & defense from a fresh Dixon-Coles fit, plus last-5 form. Higher is better on both dimensions (defense is sign-flipped for UX consistency).",
    strengths_sort: "Sort by",
    sort_overall: "Overall (default)",
    sort_attack: "Attack",
    sort_defense: "Defense",
    sort_recent_gd: "Recent GD",
    sort_elo: "ClubElo",
    sort_team: "Team name",
    home_advantage: "Home advantage",
    compare_two_teams: "Compare two",
    attack: "Att",
    defense: "Def",
    overall: "Overall",
    last5: "Last 5",
    net_gd: "GD",
  },
};

function app() {
  return {
    tab: "match",
    lang: "zh",  // 'zh' or 'en', persisted to localStorage
    activeTab: "px-3 py-1.5 rounded-md bg-slate-900 text-white",
    inactiveTab: "px-3 py-1.5 rounded-md text-slate-600 hover:bg-slate-100",

    $t(key) {
      // Returns the translated string for the current language, or the raw
      // key if it's not in the dict (helpful for finding missed strings).
      const table = I18N[this.lang] || I18N.zh;
      return table[key] ?? key;
    },

    toggleLang() {
      this.lang = this.lang === "zh" ? "en" : "zh";
      this._persistState({ lang: this.lang });
    },

    // Reference data
    leagues: [],
    teams: [],
    stats: {},
    health: { status: "loading" },
    coverage: { summary: {}, leagues: [] },
    coverageByKey: {},
    doctor: null,

    // Match form
    form: {
      league: "",
      home_team: "",
      away_team: "",
      neutral_site: false,
      knockout: false,
      model: "dixon_coles_elo",
    },
    result: null,
    loading: { predict: false, wc: false, recent: false, backtest: false, diag: false, value: false, roi: false, continental: false, inplay: false, strengths: false, replay: false, replayHistory: false, replaySurprises: false, upcoming: false, dataHealth: false },
    error: { predict: "", wc: "", recent: "", backtest: "", diag: "", value: "", roi: "", continental: "", inplay: "", strengths: "", replay: "", upcoming: "", dataHealth: "" },

    // Data health dashboard
    healthResult: null,

    // Audit vs Backtest comparison (lazy: only fetched when user expands the panel)
    auditVsBacktest: null,
    auditVsBacktestLoading: false,
    auditVsBacktestError: "",

    // Manual result entry (lives in the health tab)
    manualForm: {
      open: false,
      league: "",
      date: "",
      home_team: "",
      away_team: "",
      home_goals: 0,
      away_goals: 0,
      stage: "",
      neutral_site: false,
      busy: false,
      lastResult: "",
      lastError: "",
    },

    // World Cup
    wcForm: { n_sims: 5000 },
    wcResult: null,
    wcChart: null,

    // Leagues browser
    selectedLeague: "",
    recent: {},

    // Backtest
    backtestForm: { league: "", min_train_matches: 100, refit_every: 5 },
    backtestResult: null,

    // Diagnostics
    diagForm: { league: "premier_league", min_train_matches: 200, refit_every: 50 },
    diagResult: null,
    calibrationChart: null,

    // Value finder
    valueForm: { league: "", home_team: "", away_team: "", odds_home: null, odds_draw: null, odds_away: null },
    valueResult: null,

    // ROI simulator
    roiForm: { league: "premier_league", min_edge: 0.05, min_ev: 0.05, kelly_multiplier: 0.5, model: "dixon_coles_elo" },
    roiResult: null,
    roiChart: null,

    // In-play live predictor
    inplayForm: {
      league: "premier_league",
      home_team: "Arsenal",
      away_team: "Chelsea",
      current_home: 1,
      current_away: 0,
      minute_elapsed: 60,
    },
    inplayResult: null,

    // Team strengths
    strengthsForm: { league: "premier_league", sort_by: "overall" },
    strengthsResult: null,
    compareSet: [],

    // Match replay
    replayForm: { league: "premier_league" },
    replayHistory: [],
    replayResult: null,
    replaySurprises: null,

    // Upcoming fixtures
    upcomingForm: { days_ahead: 7 },
    upcomingResult: null,

    get sortedStrengths() {
      const teams = this.strengthsResult?.teams || [];
      const by = this.strengthsForm.sort_by;
      const arr = [...teams];
      if (by === "team") {
        arr.sort((a, b) => a.team.localeCompare(b.team));
      } else if (by === "club_elo") {
        arr.sort((a, b) => (b.club_elo ?? -Infinity) - (a.club_elo ?? -Infinity));
      } else {
        arr.sort((a, b) => (b[by] ?? 0) - (a[by] ?? 0));
      }
      return arr;
    },

    // Continental
    contForm: { league: "champions_league", home_team: "", away_team: "", neutral: true, knockout: true },
    contResult: null,
    continentalCompsList: [
      // UEFA club
      { key: "champions_league", name_zh: "欧冠", name_en: "UEFA Champions League", competition_type: "club" },
      { key: "europa_league", name_zh: "欧联", name_en: "UEFA Europa League", competition_type: "club" },
      { key: "conference_league", name_zh: "欧会杯", name_en: "UEFA Conference League", competition_type: "club" },
      { key: "uefa_super_cup", name_zh: "欧超杯", name_en: "UEFA Super Cup", competition_type: "club" },
      // Inter-continental
      { key: "fifa_club_world_cup", name_zh: "世俱杯", name_en: "FIFA Club World Cup", competition_type: "club" },
      // CONMEBOL club
      { key: "copa_libertadores", name_zh: "解放者杯", name_en: "Copa Libertadores", competition_type: "club" },
      { key: "copa_sudamericana", name_zh: "南美杯", name_en: "Copa Sudamericana", competition_type: "club" },
      { key: "recopa_sudamericana", name_zh: "南美超级杯", name_en: "Recopa Sudamericana", competition_type: "club" },
      // AFC club
      { key: "afc_champions_league", name_zh: "亚冠", name_en: "AFC Champions League Elite", competition_type: "club" },
      { key: "afc_champions_league_two", name_zh: "亚冠二级", name_en: "AFC Champions League Two", competition_type: "club" },
      // Other continents
      { key: "concacaf_champions_cup", name_zh: "中北美冠军杯", name_en: "CONCACAF Champions Cup", competition_type: "club" },
      { key: "caf_champions_league", name_zh: "非冠", name_en: "CAF Champions League", competition_type: "club" },
      { key: "ofc_champions_league", name_zh: "大洋冠", name_en: "OFC Champions League", competition_type: "club" },
      // National teams
      { key: "euro", name_zh: "欧洲杯", name_en: "UEFA Euro", competition_type: "national" },
      { key: "copa_america", name_zh: "美洲杯", name_en: "Copa América", competition_type: "national" },
      { key: "asian_cup", name_zh: "亚洲杯", name_en: "AFC Asian Cup", competition_type: "national" },
      { key: "africa_cup_of_nations", name_zh: "非洲杯", name_en: "Africa Cup of Nations", competition_type: "national" },
      { key: "uefa_nations_league", name_zh: "欧国联", name_en: "UEFA Nations League", competition_type: "national" },
      { key: "concacaf_nations_league", name_zh: "中北美国家联赛", name_en: "CONCACAF Nations League", competition_type: "national" },
    ],
    matchupPresets: {
      champions_league: [
        { label: "Real Madrid vs Bayern Munich", home: "Real Madrid", away: "Bayern Munich" },
        { label: "Manchester City vs Paris SG", home: "Man City", away: "Paris SG" },
        { label: "Liverpool vs Barcelona", home: "Liverpool", away: "Barcelona" },
        { label: "Inter vs Arsenal", home: "Inter", away: "Arsenal" },
      ],
      europa_league: [
        { label: "Roma vs Tottenham", home: "Roma", away: "Tottenham" },
        { label: "Sevilla vs Atalanta", home: "Sevilla", away: "Atalanta" },
      ],
      conference_league: [
        { label: "Fiorentina vs West Ham", home: "Fiorentina", away: "West Ham" },
      ],
      uefa_super_cup: [
        { label: "Real Madrid vs Atalanta (2024 actual)", home: "Real Madrid", away: "Atalanta" },
        { label: "Paris SG vs Tottenham (2025 actual)", home: "Paris SG", away: "Tottenham" },
      ],
      fifa_club_world_cup: [
        { label: "Real Madrid vs Flamengo", home: "Real Madrid", away: "Flamengo" },
        { label: "Manchester City vs Al Hilal", home: "Man City", away: "Al Hilal" },
        { label: "Chelsea vs Boca Juniors", home: "Chelsea", away: "Boca Juniors" },
      ],
      copa_libertadores: [
        { label: "Flamengo vs River Plate", home: "Flamengo", away: "River Plate" },
        { label: "Boca Juniors vs Palmeiras", home: "Boca Juniors", away: "Palmeiras" },
      ],
      copa_sudamericana: [
        { label: "Lanús vs Atlético Mineiro", home: "Lanús", away: "Atlético Mineiro" },
      ],
      recopa_sudamericana: [
        { label: "Botafogo vs Racing Club", home: "Botafogo", away: "Racing Club" },
      ],
      afc_champions_league: [
        { label: "Al Hilal vs Yokohama F. Marinos", home: "Al Hilal", away: "Yokohama F. Marinos" },
        { label: "Urawa Reds vs Al Ittihad", home: "Urawa Reds", away: "Al Ittihad" },
      ],
      afc_champions_league_two: [
        { label: "Sharjah vs Lion City", home: "Sharjah", away: "Lion City" },
      ],
      concacaf_champions_cup: [
        { label: "LAFC vs Cruz Azul", home: "LAFC", away: "Cruz Azul" },
        { label: "Pachuca vs Inter Miami", home: "Pachuca", away: "Inter Miami" },
      ],
      caf_champions_league: [
        { label: "Al Ahly vs Mamelodi Sundowns", home: "Al Ahly", away: "Mamelodi Sundowns" },
        { label: "Wydad vs ES Tunis", home: "Wydad", away: "ES Tunis" },
      ],
      ofc_champions_league: [
        { label: "Auckland City vs Hekari United", home: "Auckland City", away: "Hekari United" },
      ],
      euro: [
        { label: "Spain vs France", home: "Spain", away: "France" },
        { label: "England vs Germany", home: "England", away: "Germany" },
        { label: "Portugal vs Italy", home: "Portugal", away: "Italy" },
        { label: "Netherlands vs Croatia", home: "Netherlands", away: "Croatia" },
      ],
      copa_america: [
        { label: "Argentina vs Brazil", home: "Argentina", away: "Brazil" },
        { label: "Uruguay vs Colombia", home: "Uruguay", away: "Colombia" },
      ],
      asian_cup: [
        { label: "Japan vs Saudi Arabia", home: "Japan", away: "Saudi Arabia" },
        { label: "South Korea vs Iran", home: "South Korea", away: "Iran" },
        { label: "Australia vs Qatar", home: "Australia", away: "Qatar" },
      ],
      africa_cup_of_nations: [
        { label: "Morocco vs Senegal", home: "Morocco", away: "Senegal" },
        { label: "Nigeria vs Ivory Coast", home: "Nigeria", away: "Ivory Coast" },
      ],
      uefa_nations_league: [
        { label: "Spain vs France", home: "Spain", away: "France" },
        { label: "Germany vs Italy", home: "Germany", away: "Italy" },
        { label: "Netherlands vs Portugal", home: "Netherlands", away: "Portugal" },
      ],
      concacaf_nations_league: [
        { label: "United States vs Mexico", home: "United States", away: "Mexico" },
        { label: "Canada vs Panama", home: "Canada", away: "Panama" },
      ],
    },

    get continentalComps() { return this.continentalCompsList; },
    get currentContComp() {
      const meta = this.continentalCompsList.find(c => c.key === this.contForm.league);
      const fromRegistry = this.leagues.find(l => l.key === this.contForm.league);
      if (!meta || !fromRegistry) return null;
      return { ...meta, name: fromRegistry.name, note: fromRegistry.note };
    },
    get matchupsForCurrentComp() {
      return this.matchupPresets[this.contForm.league] || [];
    },

    async init() {
      this._restoreState();
      this.$watch("tab", (val) => {
        this._persistState({ tab: val });
        if (val === "strengths" && !this.strengthsResult && !this.loading.strengths) {
          this.loadStrengths();
        }
        if (val === "replay" && !this.replayHistory.length && !this.loading.replayHistory) {
          this.loadReplayHistory();
        }
        if (val === "upcoming" && !this.upcomingResult && !this.loading.upcoming) {
          this.loadUpcoming();
        }
        // The match tab now also shows a fixture board by default — load it
        // the first time the user lands there (or comes back to it). The
        // existing upcoming tab uses the same data, so the cache is shared.
        if (val === "match" && !this.upcomingResult && !this.loading.upcoming) {
          this.loadUpcoming();
        }
        if (val === "health" && !this.healthResult && !this.loading.dataHealth) {
          this.loadDataHealth();
        }
      });
      this.$watch("form", (val) => this._persistState({ form: val }), { deep: true });

      await Promise.allSettled([
        this.loadLeagues(),
        this.loadTeams(),
        this.loadStats(),
        this.loadHealth(),
        this.loadCoverage(),
        this.loadDoctor(),
      ]);

      // Trigger auto-load for the currently-active tab (the $watch above only
      // fires on subsequent CHANGES, not on initial value, so a page reload
      // that restored tab='upcoming' from localStorage would never load).
      if (this.tab === "upcoming" && !this.upcomingResult) this.loadUpcoming();
      // Match tab is the default landing — open with a live fixture board
      // pre-loaded, like a sports score site.
      if (this.tab === "match" && !this.upcomingResult) this.loadUpcoming();
      if (this.tab === "strengths" && !this.strengthsResult) this.loadStrengths();
      if (this.tab === "replay" && !this.replayHistory.length) this.loadReplayHistory();
      if (this.tab === "health" && !this.healthResult) this.loadDataHealth();
    },

    async loadDataHealth(forceRefresh = false) {
      if (!forceRefresh && this.healthResult) return;
      this.error.dataHealth = "";
      this.loading.dataHealth = true;
      try {
        const r = await fetch("/data-health");
        if (!r.ok) throw new Error(r.statusText);
        this.healthResult = await r.json();
      } catch (e) {
        this.error.dataHealth = String(e.message || e);
      } finally {
        this.loading.dataHealth = false;
      }
    },

    async loadAuditVsBacktest(forceFresh = false) {
      // Lazy: only triggered when the user expands the "compare with backtest"
      // section. Running 12 single-league backtests takes ~30-60s on first call;
      // the server caches for 24h so re-opens are instant.
      if (this.auditVsBacktestLoading) return;
      this.auditVsBacktestLoading = true;
      this.auditVsBacktestError = "";
      try {
        const url = `/audit/vs-backtest${forceFresh ? "?fresh=true" : ""}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        this.auditVsBacktest = await r.json();
      } catch (e) {
        this.auditVsBacktestError = String(e.message || e);
      } finally {
        this.auditVsBacktestLoading = false;
      }
    },

    async submitManualResult() {
      // Surface the most obvious client-side errors before round-tripping.
      const f = this.manualForm;
      f.lastResult = "";
      f.lastError = "";
      if (!f.league || !f.date || !f.home_team || !f.away_team) {
        f.lastError = "请填齐联赛/日期/主队/客队";
        return;
      }
      f.busy = true;
      try {
        const payload = {
          league: f.league.trim(),
          date: f.date,
          home_team: f.home_team.trim(),
          away_team: f.away_team.trim(),
          home_goals: Number(f.home_goals) || 0,
          away_goals: Number(f.away_goals) || 0,
          neutral_site: !!f.neutral_site,
          stage: f.stage?.trim() || null,
        };
        const r = await fetch("/manual-result", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) {
          f.lastError = body.detail || `HTTP ${r.status}`;
          return;
        }
        f.lastResult = `已录入：${body.home_team} ${body.score} ${body.away_team} (${body.league_key})`;
        // Refresh the health snapshot so the user sees the new row reflected.
        this.loadDataHealth(true);
      } catch (e) {
        f.lastError = String(e.message || e);
      } finally {
        f.busy = false;
      }
    },

    _persistState(patch) {
      try {
        const prev = JSON.parse(localStorage.getItem("fp:state") || "{}");
        localStorage.setItem("fp:state", JSON.stringify({ ...prev, ...patch }));
      } catch (_) { /* private mode etc. — silently skip */ }
    },

    _restoreState() {
      try {
        const s = JSON.parse(localStorage.getItem("fp:state") || "{}");
        if (s.tab) this.tab = s.tab;
        if (s.lang === "en" || s.lang === "zh") this.lang = s.lang;
        if (s.form) Object.assign(this.form, s.form);
      } catch (_) { /* ignore */ }
    },

    async loadLeagues() {
      try {
        const r = await fetch("/leagues");
        if (!r.ok) throw new Error(r.statusText);
        const data = await r.json();
        this.leagues = data.leagues.sort(this._leagueSort);
      } catch (e) {
        console.error("loadLeagues", e);
      }
    },

    // Explicit priority for the dropdowns. User-requested order:
    // EPL → La Liga → Bundesliga → Ligue 1 → Serie A → Primeira → CSL,
    // then the remaining top-flights, then everything else.
    _leagueOrder: [
      "premier_league",         // 英超
      "la_liga",                // 西甲
      "bundesliga",             // 德甲
      "ligue_1",                // 法甲
      "serie_a",                // 意甲
      "primeira",               // 葡超
      "chinese_super_league",   // 中超
      "eredivisie",             // 荷甲
      "belgian_pro",            // 比甲
      "saudi_pro",              // 沙特
      "j1",                     // 日 J1
      "k1",                     // 韩 K1
      "mls",                    // 美职联
      "brasileirao",            // 巴甲
      "primera_argentina",      // 阿甲
      "liga_mx",                // 墨超
      // Continental cups
      "champions_league",       // 欧冠
      "europa_league",
      "conference_league",
      "copa_libertadores",
      "copa_sudamericana",
      "afc_champions_league",
      "euro",
      "copa_america",
      "asian_cup",
      "africa_cup_of_nations",
      // Lower divisions last
      "championship",
      "league_one",
      "segunda",
      "serie_b",
      "bundesliga_2",
      "ligue_2",
      "liga_portugal_2",
    ],

    _leagueSort(a, b) {
      const order = (window.__leagueOrder ||= [
        "premier_league","la_liga","bundesliga","ligue_1","serie_a","primeira",
        "chinese_super_league","eredivisie","belgian_pro","saudi_pro",
        "j1","k1","mls","brasileirao","primera_argentina","liga_mx",
        // Continental club cups (Europe → SAM → Asia → CONCACAF → CAF → OFC)
        "champions_league","europa_league","conference_league","uefa_super_cup",
        "fifa_club_world_cup",
        "copa_libertadores","copa_sudamericana","recopa_sudamericana",
        "afc_champions_league","afc_champions_league_two",
        "concacaf_champions_cup","caf_champions_league","ofc_champions_league",
        // National-team comps
        "euro","copa_america","asian_cup","africa_cup_of_nations",
        "uefa_nations_league","concacaf_nations_league",
        // Lower divisions
        "championship","league_one","segunda","serie_b","bundesliga_2",
        "ligue_2","liga_portugal_2",
      ]);
      const ai = order.indexOf(a.key);
      const bi = order.indexOf(b.key);
      // Both in priority list → use the order
      if (ai !== -1 && bi !== -1) return ai - bi;
      // Only one in list → that one wins
      if (ai !== -1) return -1;
      if (bi !== -1) return 1;
      // Neither in list → fall back to tier + name
      return (a.tier - b.tier) || a.name.localeCompare(b.name);
    },

    async loadTeams() {
      try {
        const q = this.form.league ? `?league=${encodeURIComponent(this.form.league)}` : "";
        const r = await fetch("/teams" + q);
        if (!r.ok) throw new Error(r.statusText);
        const data = await r.json();
        this.teams = data.teams || [];
      } catch (e) {
        console.error("loadTeams", e);
        this.teams = [];
      }
    },

    async loadStats() {
      try {
        const r = await fetch("/stats");
        if (!r.ok) return;
        this.stats = await r.json();
      } catch (_) { /* ignore */ }
    },

    async loadCoverage() {
      try {
        const r = await fetch("/coverage");
        if (!r.ok) return;
        this.coverage = await r.json();
        this.coverageByKey = Object.fromEntries(
          (this.coverage.leagues || []).map((row) => [row.key, row]),
        );
      } catch (e) {
        console.error("loadCoverage", e);
      }
    },

    async loadHealth() {
      try {
        const r = await fetch("/health");
        if (!r.ok) {
          this.health = { status: "down" };
          return;
        }
        this.health = await r.json();
      } catch (_) {
        this.health = { status: "down" };
      }
    },

    async loadDoctor() {
      try {
        const r = await fetch("/doctor");
        if (!r.ok) return;
        this.doctor = await r.json();
      } catch (e) {
        console.error("loadDoctor", e);
      }
    },

    coverageFor(key) {
      return this.coverageByKey[key] || null;
    },

    statusLabel(status) {
      const key = status === "ready" ? "status_ready"
        : status === "sparse" ? "status_sparse"
        : status === "empty" ? "status_empty"
        : "status_unknown";
      return this.$t(key);
    },

    // Image URLs that failed to load (TSDB sometimes returns paths that 404).
    // Tracked as a plain Set; ``markBadgeBroken(url)`` adds, ``isBadgeBroken(url)``
    // queries. Alpine's reactivity needs us to reassign the Set (not mutate
    // in place) so the template re-renders the fallback monogram.
    brokenBadges: new Set(),
    markBadgeBroken(url) {
      if (!url || this.brokenBadges.has(url)) return;
      this.brokenBadges = new Set([...this.brokenBadges, url]);
    },
    isBadgeBroken(url) {
      return !url || this.brokenBadges.has(url);
    },

    /**
     * Leagues featured on the homepage matchup board. Curated to match the
     * scope the user actually cares about: European top divisions + East Asia
     * + Saudi Arabia + North/South America + continental club competitions.
     *
     * Lower European divisions (Segunda, Serie B, Bundesliga 2, Ligue 2,
     * Liga Portugal 2, League One, Championship) are excluded — they bury
     * the marquee fixtures with noise. They're still visible in the full
     * "upcoming" tab.
     *
     * Iran / UAE aren't currently in the registry (no fd.org or API-Football
     * coverage on our free tier); they'd need to be added there first.
     */
    featuredLeagues: new Set([
      // Europe — top flights only
      "premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1",
      "primeira", "eredivisie", "belgian_pro",
      // East Asia
      "chinese_super_league", "j1", "k1",
      // West Asia
      "saudi_pro",
      // Americas
      "mls", "liga_mx", "brasileirao", "primera_argentina",
      // Continental club competitions (these matter for cross-region fans)
      "champions_league", "europa_league",
      "copa_libertadores", "copa_sudamericana",
      "afc_champions_league",
    ]),

    /**
     * Filter the /upcoming list to fixtures that (a) have a usable prediction
     * and (b) belong to a featured league. Returns the same shape each fixture
     * has in the API response, just pruned.
     */
    featuredFixtures() {
      const all = this.upcomingResult?.fixtures || [];
      return all.filter(fx => {
        if (!this.featuredLeagues.has(fx.league_key)) return false;
        const pred = fx.prediction;
        if (!pred || pred.error) return false;
        if (!pred.probabilities) return false;
        return true;
      });
    },

    /**
     * Deterministic color for a team name. Same name → same color, every render.
     * Used by the matchup hero card to give each team a visual identity even
     * without club logos. Returns a Tailwind-friendly inline-style HSL string.
     */
    teamColor(name) {
      if (!name) return "hsl(220, 10%, 60%)";
      let hash = 0;
      for (let i = 0; i < name.length; i++) {
        hash = (hash << 5) - hash + name.charCodeAt(i);
        hash |= 0;
      }
      // Spread teams across the full hue wheel; pin saturation+lightness so
      // we get clean, distinguishable backgrounds (not too washed-out, not too neon).
      const hue = Math.abs(hash) % 360;
      return `hsl(${hue}, 55%, 48%)`;
    },

    /**
     * Two-letter (or one Chinese character) monogram for a team. Used in the
     * matchup hero. Tries to use initials of the first 1-2 significant words.
     */
    teamMonogram(name) {
      if (!name) return "?";
      // Chinese / single-token CJK names → first character
      if (/[一-鿿]/.test(name)) return name.slice(0, 1);
      // Latin: take first letter of first two words, skipping articles
      const words = name.split(/\s+/).filter(w => !/^(fc|cf|sc|ac|de|la|el)$/i.test(w));
      if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
      return name.slice(0, 2).toUpperCase();
    },

    statusClass(status) {
      return {
        ready: "bg-emerald-100 text-emerald-700",
        sparse: "bg-amber-100 text-amber-700",
        empty: "bg-rose-50 text-rose-600",
      }[status] || "bg-slate-100 text-slate-500";
    },

    async predict() {
      this.error.predict = "";
      this.result = null;
      if (!this.form.home_team || !this.form.away_team) {
        this.error.predict = "请填写主队和客队";
        return;
      }
      if (this.form.home_team === this.form.away_team) {
        this.error.predict = "两队不能相同";
        return;
      }
      this.loading.predict = true;
      try {
        const body = {
          home_team: this.form.home_team,
          away_team: this.form.away_team,
          league: this.form.league || null,
          neutral_site: this.form.neutral_site,
          knockout: this.form.knockout,
          model: this.form.model || "dixon_coles_elo",
        };
        const r = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.result = await r.json();
      } catch (e) {
        this.error.predict = String(e.message || e);
      } finally {
        this.loading.predict = false;
      }
    },

    async runWorldCup() {
      this.error.wc = "";
      this.loading.wc = true;
      try {
        const r = await fetch(`/worldcup/forecast?n_sims=${this.wcForm.n_sims}&top=24`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.wcResult = await r.json();
        this.$nextTick(() => this._drawWorldCupChart());
      } catch (e) {
        this.error.wc = String(e.message || e);
        this.wcResult = null;
      } finally {
        this.loading.wc = false;
      }
    },

    _drawWorldCupChart() {
      const canvas = document.getElementById("wc-chart");
      if (!canvas || !this.wcResult) return;
      const top = this.wcResult.table.slice(0, 10);
      if (this.wcChart) {
        this.wcChart.destroy();
      }
      this.wcChart = new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
          labels: top.map((r) => r.team),
          datasets: [
            {
              label: "夺冠概率",
              data: top.map((r) => +(r.p_champion * 100).toFixed(2)),
              backgroundColor: "rgba(15, 118, 110, 0.7)",
              borderColor: "rgba(15, 118, 110, 1)",
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { callback: (v) => v + "%" },
            },
          },
        },
      });
    },

    async runBacktest() {
      this.error.backtest = "";
      this.backtestResult = null;
      this.loading.backtest = true;
      try {
        const body = {
          league: this.backtestForm.league || null,
          min_train_matches: this.backtestForm.min_train_matches,
          refit_every: this.backtestForm.refit_every,
          summary_only: true,
        };
        const r = await fetch("/backtest", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.backtestResult = await r.json();
      } catch (e) {
        this.error.backtest = String(e.message || e);
      } finally {
        this.loading.backtest = false;
      }
    },

    async runDiagnostics() {
      this.error.diag = "";
      this.diagResult = null;
      this.loading.diag = true;
      try {
        const r = await fetch("/diagnostics", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            league: this.diagForm.league || null,
            min_train_matches: this.diagForm.min_train_matches,
            refit_every: this.diagForm.refit_every,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.diagResult = await r.json();
        this.$nextTick(() => this._drawCalibrationChart());
      } catch (e) {
        this.error.diag = String(e.message || e);
      } finally {
        this.loading.diag = false;
      }
    },

    _drawCalibrationChart() {
      const canvas = document.getElementById("calibration-chart");
      if (!canvas || !this.diagResult) return;
      if (this.calibrationChart) this.calibrationChart.destroy();

      // Build one scatter dataset per outcome, plus a y=x reference line.
      const datasets = [];
      const colors = {
        home_win: { bg: "rgba(16,185,129,0.7)", border: "rgba(16,185,129,1)" },
        draw: { bg: "rgba(148,163,184,0.7)", border: "rgba(148,163,184,1)" },
        away_win: { bg: "rgba(99,102,241,0.7)", border: "rgba(99,102,241,1)" },
      };
      const labels = { home_win: "主胜", draw: "平", away_win: "客胜" };
      for (const outcome of ["home_win", "draw", "away_win"]) {
        const points = this.diagResult.diagnostics.calibration_curve[outcome]
          .filter((b) => b.n > 0)
          .map((b) => ({ x: b.mean_predicted, y: b.observed_frequency, n: b.n }));
        datasets.push({
          label: labels[outcome],
          data: points,
          backgroundColor: colors[outcome].bg,
          borderColor: colors[outcome].border,
          pointRadius: (ctx) => Math.min(12, 3 + Math.sqrt(ctx.raw.n) * 0.5),
          showLine: false,
        });
      }
      // y=x reference
      datasets.push({
        label: "完美校准 (y=x)",
        data: [
          { x: 0, y: 0 },
          { x: 1, y: 1 },
        ],
        borderColor: "rgba(15,23,42,0.4)",
        borderDash: [4, 4],
        borderWidth: 1.5,
        pointRadius: 0,
        showLine: true,
        type: "line",
        fill: false,
      });

      this.calibrationChart = new Chart(canvas.getContext("2d"), {
        type: "scatter",
        data: { datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom", labels: { boxWidth: 12 } },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const r = ctx.raw;
                  if (r.n === undefined) return `${ctx.dataset.label}`;
                  return `${ctx.dataset.label}: 预测 ${(r.x * 100).toFixed(1)}% · 实际 ${(r.y * 100).toFixed(1)}% (n=${r.n})`;
                },
              },
            },
          },
          scales: {
            x: {
              title: { display: true, text: "模型预测概率" },
              min: 0,
              max: 1,
              ticks: { callback: (v) => (v * 100).toFixed(0) + "%" },
            },
            y: {
              title: { display: true, text: "实际发生率" },
              min: 0,
              max: 1,
              ticks: { callback: (v) => (v * 100).toFixed(0) + "%" },
            },
          },
        },
      });
    },

    async findValue() {
      this.error.value = "";
      this.valueResult = null;
      if (!this.valueForm.home_team || !this.valueForm.away_team) {
        this.error.value = "请填写两支球队";
        return;
      }
      const { odds_home, odds_draw, odds_away } = this.valueForm;
      if (!odds_home || !odds_draw || !odds_away ||
          odds_home <= 1 || odds_draw <= 1 || odds_away <= 1) {
        this.error.value = "三个赔率都要 > 1.0（十进制盘口）";
        return;
      }
      this.loading.value = true;
      try {
        // Step 1: get model probabilities via /predict.
        const r = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            home_team: this.valueForm.home_team,
            away_team: this.valueForm.away_team,
            league: this.valueForm.league || null,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        const prediction = await r.json();
        const probs = prediction.probabilities;

        // Step 2: compute EV + Kelly client-side. Both are textbook formulas
        // and doing the math here keeps the API surface small.
        // EV per 1 unit stake = p * (odds - 1) - (1 - p)
        // Kelly = (p * (odds - 1) - (1 - p)) / (odds - 1) = (p * odds - 1) / (odds - 1)
        const odds = {
          home_win: odds_home,
          draw: odds_draw,
          away_win: odds_away,
        };
        const labels = {
          home_win: this.valueForm.home_team + " 胜",
          draw: "平局",
          away_win: this.valueForm.away_team + " 胜",
        };

        const overround = 1 / odds_home + 1 / odds_draw + 1 / odds_away;
        const rows = ["home_win", "draw", "away_win"].map((outcome) => {
          const p = probs[outcome];
          const o = odds[outcome];
          const impliedRaw = 1 / o;
          const implied = impliedRaw / overround;  // normalised to remove vig
          const ev = p * (o - 1) - (1 - p);
          const kelly = Math.max(0, (p * o - 1) / (o - 1));
          return {
            outcome,
            outcome_label: labels[outcome],
            model_prob: p,
            implied_prob: implied,
            edge: p - implied,
            ev,
            kelly,
          };
        });

        const bestBet = rows
          .filter((r) => r.ev > 0.01 && r.edge > 0.02)
          .sort((a, b) => b.ev - a.ev)[0] || null;

        this.valueResult = { rows, best_bet: bestBet, overround };
      } catch (e) {
        this.error.value = String(e.message || e);
      } finally {
        this.loading.value = false;
      }
    },

    async loadStrengths() {
      this.error.strengths = "";
      this.strengthsResult = null;
      this.compareSet = [];
      if (!this.strengthsForm.league) return;
      this.loading.strengths = true;
      try {
        const q = encodeURIComponent(this.strengthsForm.league);
        const r = await fetch(`/teams/strengths?league=${q}`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.strengthsResult = await r.json();
      } catch (e) {
        this.error.strengths = String(e.message || e);
      } finally {
        this.loading.strengths = false;
      }
    },

    toggleCompare(team) {
      const idx = this.compareSet.indexOf(team);
      if (idx >= 0) {
        this.compareSet.splice(idx, 1);
      } else {
        if (this.compareSet.length >= 2) this.compareSet.shift();
        this.compareSet.push(team);
      }
    },

    strengthsByTeam(team) {
      return (this.strengthsResult?.teams || []).find((t) => t.team === team);
    },

    barWidth(team, dim) {
      // Normalize the absolute strength on [0, ~1.5] to 0-50%.
      // Bars sum to 100% across two teams; each gets a fair share.
      const a = this.strengthsByTeam(this.compareSet[0])?.[dim] ?? 0;
      const b = this.strengthsByTeam(this.compareSet[1])?.[dim] ?? 0;
      const ratio = (Math.abs(a) + Math.abs(b)) || 1;
      const target = team === this.compareSet[0] ? a : b;
      return Math.max(5, Math.min(95, (Math.abs(target) / ratio) * 100));
    },

    async loadReplayHistory() {
      this.error.replay = "";
      this.replayHistory = [];
      if (!this.replayForm.league) return;
      this.loading.replayHistory = true;
      try {
        const q = encodeURIComponent(this.replayForm.league);
        const r = await fetch(`/match-history?league=${q}&limit=30`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        const data = await r.json();
        this.replayHistory = data.matches || [];
      } catch (e) {
        this.error.replay = String(e.message || e);
      } finally {
        this.loading.replayHistory = false;
      }
    },

    async runReplay(match) {
      this.error.replay = "";
      this.replayResult = null;
      this.loading.replay = true;
      try {
        const r = await fetch("/match-replay", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            league: this.replayForm.league,
            match_date: match.date,
            home_team: match.home_team,
            away_team: match.away_team,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.replayResult = await r.json();
      } catch (e) {
        this.error.replay = String(e.message || e);
      } finally {
        this.loading.replay = false;
      }
    },

    async loadSurprises() {
      this.error.replay = "";
      this.loading.replaySurprises = true;
      try {
        const q = encodeURIComponent(this.replayForm.league);
        const r = await fetch(`/match-replay/surprises?league=${q}&refit_every=25`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.replaySurprises = await r.json();
      } catch (e) {
        this.error.replay = String(e.message || e);
      } finally {
        this.loading.replaySurprises = false;
      }
    },

    async loadUpcoming(forceRefresh = false) {
      this.error.upcoming = "";
      // Skip only when we already have a fresh result AND the user didn't ask
      // for an explicit refresh AND the days_ahead matches what we fetched.
      if (!forceRefresh && this.upcomingResult &&
          this.upcomingResult.days_ahead === this.upcomingForm.days_ahead) {
        return;
      }
      this.loading.upcoming = true;
      try {
        const r = await fetch(`/upcoming?days_ahead=${this.upcomingForm.days_ahead}`);
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.upcomingResult = await r.json();
      } catch (e) {
        this.error.upcoming = String(e.message || e);
      } finally {
        this.loading.upcoming = false;
      }
    },

    async predictInPlay() {
      this.error.inplay = "";
      this.inplayResult = null;
      if (!this.inplayForm.home_team || !this.inplayForm.away_team) {
        this.error.inplay = "请填写两支球队";
        return;
      }
      this.loading.inplay = true;
      try {
        const r = await fetch("/predict/in-play", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            home_team: this.inplayForm.home_team,
            away_team: this.inplayForm.away_team,
            league: this.inplayForm.league || null,
            current_home: this.inplayForm.current_home,
            current_away: this.inplayForm.current_away,
            minute_elapsed: this.inplayForm.minute_elapsed,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.inplayResult = await r.json();
      } catch (e) {
        this.error.inplay = String(e.message || e);
      } finally {
        this.loading.inplay = false;
      }
    },

    async predictContinental() {
      this.error.continental = "";
      this.contResult = null;
      if (!this.contForm.home_team || !this.contForm.away_team) {
        this.error.continental = "请填写主队和客队";
        return;
      }
      this.loading.continental = true;
      try {
        const r = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            home_team: this.contForm.home_team,
            away_team: this.contForm.away_team,
            league: this.contForm.league,
            neutral_site: this.contForm.neutral,
            knockout: this.contForm.knockout,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.contResult = await r.json();
      } catch (e) {
        this.error.continental = String(e.message || e);
      } finally {
        this.loading.continental = false;
      }
    },

    async runROI() {
      this.error.roi = "";
      this.roiResult = null;
      this.loading.roi = true;
      try {
        const r = await fetch("/roi-simulation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            league: this.roiForm.league,
            min_edge: this.roiForm.min_edge,
            min_ev: this.roiForm.min_ev,
            kelly_multiplier: this.roiForm.kelly_multiplier,
            model: this.roiForm.model || "dixon_coles_elo",
            include_bets: false,
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          throw new Error(err.detail || r.statusText);
        }
        this.roiResult = await r.json();
        this.$nextTick(() => this._drawROIChart());
      } catch (e) {
        this.error.roi = String(e.message || e);
      } finally {
        this.loading.roi = false;
      }
    },

    _drawROIChart() {
      const canvas = document.getElementById("roi-chart");
      if (!canvas || !this.roiResult) return;
      if (this.roiChart) this.roiChart.destroy();
      const curve = this.roiResult.bankroll_curve;
      const start = this.roiResult.summary.starting_bankroll;
      this.roiChart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
          labels: curve.map((p) => p.date),
          datasets: [
            {
              label: "资金",
              data: curve.map((p) => p.bankroll),
              borderColor: "rgba(15,23,42,0.9)",
              backgroundColor: "rgba(15,23,42,0.05)",
              fill: true,
              tension: 0.05,
              pointRadius: 0,
              borderWidth: 1.5,
            },
            {
              label: "起始本金",
              data: curve.map(() => start),
              borderColor: "rgba(148,163,184,0.7)",
              borderDash: [4, 4],
              borderWidth: 1.5,
              pointRadius: 0,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom", labels: { boxWidth: 12 } },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  if (ctx.datasetIndex === 1) return `起始本金: ${start.toFixed(0)}`;
                  return `资金: ${ctx.parsed.y.toFixed(2)} (第 ${ctx.dataIndex + 1} 注)`;
                },
              },
            },
          },
          scales: {
            x: {
              ticks: {
                maxTicksLimit: 8,
                callback: function (val, idx) {
                  // Show only year-month for readability.
                  const label = this.getLabelForValue(val);
                  return label && label.length >= 7 ? label.slice(0, 7) : label;
                },
              },
            },
            y: {
              beginAtZero: false,
            },
          },
        },
      });
    },

    async selectLeague(key) {
      this.selectedLeague = key;
      this.recent = {};
      this.error.recent = "";
      this.loading.recent = true;
      try {
        const r = await fetch(`/recent?league=${encodeURIComponent(key)}&limit=30`);
        if (!r.ok) throw new Error(r.statusText);
        this.recent = await r.json();
      } catch (e) {
        this.error.recent = String(e.message || e);
      } finally {
        this.loading.recent = false;
      }
    },
  };
}
