from tcrca.eval.metrics import (
    EvalRecord,
    aggregate_binary_alerts,
    avg_top_k_accuracy,
    dataset_top_k_accuracy,
    edge_f1,
    edge_jaccard,
    mean_reciprocal_rank,
    rank_accuracy_curve,
    root_hit_at_k,
    root_rank,
)


def test_root_hits_and_mrr():
    ranked = ["c", "a", "b", "d"]
    assert root_hit_at_k(ranked, "a", 1) == 0
    assert root_hit_at_k(ranked, "a", 2) == 1
    assert root_rank(ranked, "a") == 2
    assert mean_reciprocal_rank([2, None, 1]) == (0.5 + 0 + 1.0) / 3


def test_rank_accuracy_curve():
    ranked_per = [["a", "b"], ["x", "y", "z"]]
    true_ids = ["b", "x"]
    out = rank_accuracy_curve(ranked_per, true_ids, ks=(1, 2))
    assert out["hit@1"] == 0.5
    assert out["hit@2"] == 1.0


def test_edge_metrics():
    pred = {("a", "b"), ("b", "c")}
    gt = {("a", "b"), ("b", "d")}
    assert edge_jaccard(pred, gt) == 1 / 3
    p, r, f1 = edge_f1(pred, gt)
    assert 0 < f1 < 1


def test_paper_a_at_k_and_avg_at_k():
    recs = [
        EvalRecord("a", 1, 1, 1, 1, 1.0),
        EvalRecord("b", 0, 1, 1, 2, 0.5),
    ]
    assert dataset_top_k_accuracy(recs, 1) == 0.5
    assert dataset_top_k_accuracy(recs, 2) == 1.0
    assert abs(avg_top_k_accuracy(recs, 2) - 0.75) < 1e-9


def test_alert_confusion():
    conf = aggregate_binary_alerts(
        [True, True, False, False],
        [True, False, False, True],
    )
    assert conf["TP"] == 1.0
    assert conf["FN"] == 1.0
    assert conf["FP"] == 1.0
    assert conf["TN"] == 1.0
