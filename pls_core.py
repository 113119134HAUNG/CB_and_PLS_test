# 1) 跑模型（只用 plspm；回傳 model + scores）
model1, scores1 = run_plspm_python(cog, Xpls, path1, lv_blocks1, lv_modes1)

# 2) 對齊 LV 順序（不創造新值，只做欄位排序/改名）
scores1 = scores1[order1] if set(order1).issubset(scores1.columns) else scores1.iloc[:, :len(order1)]
scores1.columns = order1

# 3) sign-fix（只乘 -1，不產生新數值）
anchors1 = choose_anchors_by_max_abs_loading(Xpls, scores1, lv_blocks1)  # 若你已有就用你原本那支
sign_map1 = get_sign_map_by_anchors(Xpls, scores1, anchors1)
scores1 = sign_fix_scores_by_anchors(scores1, Xpls, anchors1)

# 4) cross-loadings（可用 corr；這是檢視表，不是 outer loading 來源）
PLS1_cross = corr_items_vs_scores(Xpls[item_cols], scores1[order1]).round(3)

# 5) Outer：只從 model 取（strict=True）
outer1 = get_outer_results(model1, Xpls[item_cols], scores1[order1], lv_blocks1, lv_modes1, strict=True)
outer1 = apply_sign_to_outer(outer1, sign_map1)
PLS1_outer = outer1.round(3)

# 6) CR/AVE：用 outer loadings（model-based）
PLS1_quality = quality_paper_table(outer1, lv_modes1, order=order1).round(3)

# 7) Paths：只從 model 取（strict=True）
pe1 = get_path_results(model1, path1, strict=True)
pe1 = apply_sign_to_paths(pe1, sign_map1)

key1 = pe1[["from", "to"]].copy()
est1 = pe1["estimate"].astype(float).values
