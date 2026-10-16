FiGRA manuscript result data
Release paper-results-20261016

This package contains the numerical results displayed in Tables 1–2 and Figures 4–8 of the FiGRA manuscript. It is a repository data release, not a journal supplementary-material file.

Table_1_summary.csv and Table_2_summary.csv contain unrounded means. Their values round to every displayed manuscript cell. Separate seed-value files contain only the underlying values for these tables. The frozen baseline is evaluated once; it is not treated as three independent runs.

Figure_4_gains.csv contains all displayed action/sequence gains and sample standard deviations. Figure_4_seed_values.csv supplies the three values behind each error bar. The baseline is fixed, so the standard deviation of the paired gain equals the standard deviation of FiGRA across training seeds.

Figure_6_coordinate_jitter.csv contains every plotted clean and coordinate-jitter point for all six methods. Other perturbation families are excluded. Each line point is the archived aggregate over training and perturbation runs; this package does not claim to contain full raw predictions.

Figure_7_budget.csv contains all five displayed budgets, four methods, two datasets and two metrics. Figure_7_seed_values.csv provides the three values behind each mean and sample standard deviation. The methods retain the original protocols described in the manuscript.

Figure_5_case_index.csv identifies exactly the six displayed MuPoTS examples. Figure_5_displayed_predictions.csv contains the two displayed model reconstructions in millimetres, after the pelvis centering and target bone-length mapping used in the figure. No rotation fitting or smoothing is introduced. Original images and reference annotations remain available from the original MuPoTS provider; this package does not redistribute those assets. These post-hoc qualitative examples are illustrative, not a representative accuracy sample.

Figure_8_case_index.csv identifies exactly the four displayed ordinary-video snapshots. Figure_8_displayed_predictions.csv contains the two pelvis-relative model reconstructions in millimetres. There is no 3D ground truth for these clips. Original footage links remain in DATASETS.md in the parent FiGRA repository; the videos themselves are not redistributed here.

The 17-joint coordinate order is pelvis, right hip, right knee, right ankle, left hip, left knee, left ankle, spine, thorax, neck, head, left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist. Frame and track indices are zero-based.

PCK is evaluated at 150 mm. AUC is over thresholds from 0 to 150 mm. Both metrics are percentages and larger is better. HumanEva uses valid-joint pooled aggregation; MuPoTS averages the 20 sequence metrics. Training seed identifiers are labels for existing runs, not new runs. Standard deviations use ddof=1.

Figures 1–3 explain the method and have no additional experimental results. No abandoned experiment, unused metric, extra backbone comparison, training log, checkpoint-selection history, or unreported occlusion experiment is included. This release provides reported-result data; it is not a complete training or benchmark-reproduction pipeline. The manuscript's original-dataset references and the repository DATASETS.md provide upstream access information.

No new model training or inference was run for this export. RESULTS_INDEX.json records archive hashes; MANIFEST_SHA256.json records all distributed file hashes. verify_results.py checks these hashes and recomputes the exported means and standard deviations using only the Python standard library.
