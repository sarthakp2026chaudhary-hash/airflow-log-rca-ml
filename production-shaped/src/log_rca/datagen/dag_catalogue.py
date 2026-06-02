"""The 20 bank-flavoured DAGs the simulator picks from."""

from __future__ import annotations

DAGS: dict[str, list[str]] = {
    "etl_customer_daily":         ["extract_crm", "validate", "transform", "load_bq", "notify"],
    "risk_pnl_hourly":            ["fetch_positions", "fetch_marks", "compute_pnl", "persist"],
    "kyc_refresh":                ["pull_kyc", "screen_sanctions", "score", "alert_ops"],
    "mortgage_scoring":           ["extract_apps", "feature_eng", "score_model", "write_decisions"],
    "fraud_detection_streaming":  ["consume_kafka", "enrich", "score", "publish_alerts"],
    "regulatory_basel_report":    ["aggregate_positions", "compute_rwa", "generate_xbrl", "submit"],
    "card_txn_settlement":        ["pull_txns", "match", "settle", "reconcile"],
    "branch_metrics_daily":       ["extract_branches", "aggregate", "publish_dashboard"],
    "loan_originations_etl":      ["extract", "validate", "enrich_credit_bureau", "load_bq"],
    "aml_monitor_daily":          ["fetch_txns", "rule_engine", "score_ml", "create_cases"],
    "treasury_cash_position":     ["pull_balances", "fx_convert", "aggregate", "report"],
    "wealth_portfolio_rebalance": ["fetch_holdings", "rebalance", "execute_orders"],
    "credit_card_rewards":        ["pull_spend", "compute_rewards", "post_to_accounts"],
    "marketing_campaign_attrib":  ["pull_clicks", "join_conversions", "attribute", "publish"],
    "deposit_interest_accrual":   ["pull_balances", "compute_accrual", "post_gl"],
    "swift_message_parser":       ["consume", "parse", "validate", "route"],
    "collateral_valuation":       ["fetch_collateral", "mark_to_market", "haircut", "persist"],
    "stress_test_scenarios":      ["load_scenarios", "run_models", "aggregate", "report"],
    "customer_360_refresh":       ["pull_sources", "resolve_entities", "build_profile"],
    "operational_loss_etl":       ["pull_incidents", "classify", "aggregate", "report"],
}
