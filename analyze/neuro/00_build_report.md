Read through the results in the `runs-neuro` directory and make an interactive html report `analyze/neuro/report.html`.

The report should show the main result (the metric as a function of iterations) for each run, labeling the run with the model / thinking effort it uses. You might need to read the copilot cli logs to find those (if you do, write a `metadata.json` file into each run's folder with the relevant info for the report for next time).

The `fmri-may27-run1` did not trim the story ends (all other runs trimmed 30 TRs off each end). Present the may27 run first separately from the rest and mention this difference.

Make it so that the user can interactively see what methods were tried, and give a writeup based on the overall_results.csv descriptions on what methods were tried, and which worked / didn't.

Flag any iterations that used any training on data, as this is explicitly disallowed.

Style everything including the code as a white-background, clean design.

At the bottom (after scrolling down), show a summary plot that has all the datasets in one plot. The points for each run should be connected as a line.

Check the rendering at the end and make sure that things are non-overlapping and readable, even when things are clicked. If not, make adjustments to the layout and spacing.
