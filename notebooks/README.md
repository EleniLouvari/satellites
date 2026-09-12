# Notebooks

Select a kernel where the repository has been installed in editable mode. The
notebooks import `satellites` directly and do not modify `sys.path`.

| Folder | Contents |
| --- | --- |
| `projects/volvi/` | Download, parcel statistics (openEO or Planet), feature enrichment, EDA, and classification |
| `experiments/` | ML inspection and multi-seed sensitivity experiments |
| `maintenance/` | openEO job inspection and repository quality reports |

The numbered Volvi notebooks stay together so the full workflow is easy to
follow. The two `1_` notebooks are alternative imagery-source workflows.
Existing external input and output paths and saved notebook outputs are retained.
Review the path/configuration cell before running a notebook against your data.

New code uses explicit imports. `satellites.shared.notebook.configure_notebook`
provides opt-in display/reproducibility settings. Only notebooks requesting
`tensorflow=True` need TensorFlow for that setup.
