# Data and implementation credits

Code is an independent MIT implementation by itsmygithubacct. No Cactus code,
Needle model weights, or teacher-model outputs are included.

Training sources selected by the project owner:

| Source | Author | License | Use |
| --- | --- | --- | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Hugging Face and contributors | ODC-By 1.0 | 70% of base-pretraining tokens |
| [SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH) | Pleias and AI Alliance | CC-BY-4.0 | 30% of base-pretraining tokens; English query and answer fields |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | NVIDIA Corporation; Hayley Ross, Ameya Sunil Mahabaleshwarka, Yoshi Suhara | CC-BY-4.0 | Supported training SFT clarification/inability examples converted to empty call lists |

Dataset revisions, file hashes, filtering, exclusions and transformation counts
are recorded in acquisition/corpus manifests. When2Call's test sets are not used
for training. Its converted no-call examples do not preserve its natural-language
benchmark. SYNTH reasoning traces are not included in the selected prose.

Kilix templates migrated from kilix-needle-tuning are MIT; their original
provenance is in `domains/kilix_panes/MIGRATION.json`. Later independently authored
templates have separate provenance. Model weights produced by this project use
MIT with these data credits. No trained weights are bundled with the code.
