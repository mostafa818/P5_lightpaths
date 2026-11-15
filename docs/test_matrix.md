# Test matrix (plan Step 8.1)

The plan asks for at least 5 normal, 3 edge and 2 impossible tests. Every requirement below is covered by the tests
named next to it (`file::test function`, all in `src/tests/`). `src/tests/test_matrix.py` fails if a test named here
no longer exists or if a section drops below the minimum, so this table cannot rot silently.
Tests that need the git-ignored raw data skip cleanly in a fresh clone; every requirement below also names at least one test that runs without it (checked, see the last section).
Run everything with `pytest` (see the README); `pip install coverage` and `coverage run --include="*/src/*,*/run_demo.py,*/make_data.py" -m pytest`
measures line coverage.

## Normal (minimum 5)
- UCS finds the known shortest path: `test_search.py::test_ucs_and_astar_find_the_hand_computed_shortest_path`
- A* and UCS cost equal the networkx optimum, BFS has the fewest hops (150 random graphs, no raw data needed): `test_search.py::test_random_graphs_ucs_astar_and_bfs_match_networkx_without_raw_data`; and on all 132 Abilene pairs (needs the raw topology): `test_search.py::test_abilene_ucs_and_astar_match_networkx_on_all_pairs`
- Requests sharing a link get different wavelengths: `test_csp.py::test_requests_sharing_a_link_get_different_wavelengths`
- Link-disjoint requests may share a wavelength: `test_csp.py::test_link_disjoint_requests_may_share_a_wavelength`
- BT, FC and AC-3 agree and return valid solutions: `test_csp.py::test_variants_agree_with_brute_force_on_small_instances`, `test_csp.py::test_real_abilene_instances_all_variants_agree_and_are_valid`, `test_integration.py::test_methods_agree_and_unknown_method_rejected`
- Full pipeline output is valid for every method: on a toy network `test_integration.py::test_evidence_request_must_be_dropped_on_toy`, `test_integration.py::test_evidence_more_routes_fix_a_failure` (no raw data needed); on Abilene `test_integration.py::test_abilene_full_pipeline_outputs_are_valid`
- CSP extras: pre-check never changes the answer, AC-3 is sound and reaches a fixed point: `test_csp.py::test_precheck_never_changes_the_answer`, `test_csp.py::test_ac3_reaches_a_fixed_point_and_is_never_unsound`
- Data: hourly means equal a manual average of 12 five-minute values: `test_data.py::test_to_hourly_is_mean_of_12` (synthetic, no raw data needed), `test_data.py::test_hourly_matches_manual_average` (on the real file)
- NN forecast: shape, non-negative, same seed same result: `test_nn.py::test_forecast_shape_nonnegative`, `test_nn.py::test_same_seed_same_result`

## Edge (minimum 3)
- K = 1: `test_csp.py::test_single_request_and_empty_input`, `test_integration.py::test_edge_k_equals_one`, `test_routes.py::test_k_equals_one_returns_the_shortest`
- W = 1 with disjoint requests: `test_csp.py::test_link_disjoint_requests_may_share_a_wavelength`, `test_integration.py::test_edge_w1_disjoint_requests_share_wavelength`
- Reach limit exactly equal to a route length: `test_routes.py::test_reach_limit_is_inclusive`, `test_integration.py::test_edge_reach_exactly_equal_to_route_length`
- Tied demands: `test_csp.py::test_tied_demands_are_solved_deterministically`, `test_integration.py::test_edge_tied_demands_are_deterministic`
- Solver cap equal to the work needed still solves: `test_csp.py::test_cap_equal_to_the_work_needed_still_solves`
- No future data reaches the forecast or the planner: `test_nn.py::test_no_feature_uses_hour_t_or_later`, `test_integration.py::test_abilene_no_future_data_reaches_the_planner`

## Impossible (minimum 2)
- (a) wavelengths exhausted on a link, clear message: `test_csp.py::test_wavelengths_exhausted_on_a_link`, `test_integration.py::test_impossible_case_a_strict_and_normal`
- (b) no route within reach for a request, clear message: `test_csp.py::test_no_route_within_reach_for_a_request`, `test_integration.py::test_impossible_case_b_no_route_within_reach`
- Solver timeout is reported, not a hang or a crash: `test_csp.py::test_max_assignments_returns_timeout_with_counters`, `test_integration.py::test_solver_timeout_is_a_failure_not_a_crash`

## Integration evidence (plan 5.7)
- More routes fix a failure (toy network; none exists on Abilene, see the README): `test_integration.py::test_evidence_more_routes_fix_a_failure`
- A request must be dropped: `test_integration.py::test_evidence_request_must_be_dropped_on_abilene`, `test_integration.py::test_evidence_request_must_be_dropped_on_toy`

## Per-stage test steps of the plan (Member B)
Where a step's tests need the git-ignored raw files, the line says so (they skip in a fresh clone without them);
the other lines have at least one test that runs without raw data.
- 1.9 node mapping (needs raw data): `test_node_mapping.py::test_a_paths_follow_distance_shortest_paths`, `test_node_mapping.py::test_x01_slot21_matches_sndlib_sample`
- 1.13 data: shape, no NaN or negatives, hourly value = mean of 12: `test_data.py::test_processed_shape_and_values`, `test_data.py::test_hourly_matches_manual_average`, `test_data.py::test_to_hourly_is_mean_of_12`
- 2.6 search: hand-drawn graph, networkx on all pairs: `test_search.py::test_ucs_and_astar_find_the_hand_computed_shortest_path`, `test_search.py::test_abilene_ucs_and_astar_match_networkx_on_all_pairs`
- 2.8 routes: all simple paths of the toy graph in order, `src == dst` raises, fewer than k, unreachable, networkx on Abilene: `test_routes.py::test_all_seven_simple_paths_in_order`, `test_routes.py::test_src_equals_dst_raises`, `test_routes.py::test_fewer_than_k_when_fewer_exist`, `test_routes.py::test_unreachable_node_gives_empty_list`, `test_routes.py::test_abilene_routes_match_networkx_on_all_pairs`
- 2.9 germany50 (needs raw data): `test_germany50.py::test_germany50_size_and_connectivity`, `test_germany50.py::test_germany50_own_haversine_roughly_matches_declared_dist`
- 3.8 CSP: variants agree with brute force, same solution, valid: `test_csp.py::test_variants_agree_with_brute_force_on_small_instances`, `test_csp.py::test_variants_find_the_same_solution_and_prune_in_order`
- 4.8 NN: shape, non-negative, same seed same result, no feature uses hour t or later: `test_nn.py::test_forecast_shape_nonnegative`, `test_nn.py::test_same_seed_same_result`, `test_nn.py::test_no_feature_uses_hour_t_or_later`, `test_nn.py::test_forecast_ignores_future`
- 5.8 pipeline on a toy network and on Abilene, `is_valid` on every output: `test_integration.py::test_metrics_on_toy`, `test_integration.py::test_abilene_full_pipeline_outputs_are_valid`

## Checks done for plan Step 8.2 and coverage (Member A, run on the Stage 8 branch)
- **Fresh clone, no raw data** (new directory, new venv, `pip install -r requirements.txt`, `pip install -e .`): 176 passed, 59 skipped, 0 failed. The skips are the tests that need the git-ignored raw files; each says so.
  The first fresh-clone run found one failure (`test_experiments.py::test_exp2_methods_agree_on_a_real_instance` did not skip without the raw topology); fixed in this branch. Every requirement above names at least one test that passes in this setting (verified by running each row's named tests in the clone; the first check found two rows of this very matrix and one row of Member B's draft that depended on raw data only, now fixed).
- **Fresh clone with the raw data copied in:** 235 passed, 0 skipped. `python make_data.py` regenerates `data/processed/` and the traffic plot byte for byte; `python -m nn.train` regenerates the models and `nn_eval.json` byte for byte; `python -m eval.exp3_nn --reuse` regenerates the Experiment 3 summary byte for byte; `python run_demo.py --W 4 --trace --plots` runs and writes all four figures.
- **Coverage** (`coverage run --include="*/src/*,*/run_demo.py,*/make_data.py" -m pytest`): Member A's modules (`data`, `nn`, `integration`, `eval/plots.py`, `eval/nn_effect.py`, `eval/exp3_nn.py`, `common.py`, `run_demo.py`) 99%; the only uncovered lines (5) are inside the four `if __name__ == "__main__"` guards (`make_data.py` is a script and is tested by running it in a scratch folder instead). Whole repository: about 93%.
- Not covered by any test, by design: the full runs of Experiments 1 and 2 (hours; the experiment tests run small slices and the raw results are committed and were reproduced independently by both members).
