# Different types of syntactic agreement recruit the same units within large language models

[![arXiv](https://img.shields.io/badge/arXiv-2512.03676-b31b1b.svg)](https://arxiv.org/abs/2512.03676)

Using a functional localization approach inspired by cognitive neuroscience, we identify the LLM units most responsive to 67 English syntactic phenomena in seven open-weight models. These units are consistently recruited across sentences containing the phenomena and causally support the models’ syntactic performance. Critically, different types of syntactic agreement (e.g., subject-verb, anaphor, determiner-noun) recruit overlapping sets of units, suggesting that agreement constitutes a meaningful functional category for LLMs. This pattern holds in English, Russian, and Chinese; and further, in a cross-lingual analysis of 57 diverse languages, structurally more similar languages share more units for subject-verb agreement.

## Setup

1. Create virtual environment: `python -m venv .venv`
2. Activate environment: `. .venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`

## Repository Structure

* `/`: Experiment scripts and plotting scripts
  * Experiment script example: `python ablation.py --ablation-type zero --model-name google/gemma-3-4b-pt --dataset blimp --percentage 1.0 --savedir english/ablation` runs the zero-ablation experiment with the Gemma model, BLiMP benchmark, and 1% unit localization. The result (txt file) will be saved in the `english/ablation` directory.
  * Plotting script example: `python ablation_plot.py --dataset BLiMP --directory english/ablation --display error-bars` plots average ablation effects over *all* ablation result files in the `english/ablation` directory. Change `--display` to `model-markers` to get a figure with model markers instead of error bars.

* `english/`: Data and figures for the English experiments
  * `cross-validation/`: Main cross-validation experiment (Sec. 4.1, Fig. 1)
  * `ablation/`: Main zero-ablation experiment (Sec. 4.1, Fig. 2) + figure showing individual model scores (Appendix G, Fig. 20)
  * `cross-overlap/`: Within-category and cross-category overlaps between phenomena (Sec. 4.2, Figs. 3-5)
  * `0.5%/`: Experiments targeting top-0.5% of units (Appendix A, Figs. 9-11)
  * `5%/`: Experiments targeting top-5% of units (Appendix B, Figs. 12-14)
  * `finegrained/`: Experiments targeting MLP and attention submodules (Appendix C, Figs. 15-16)
  * `5-fold/`: Five-fold cross-validation (Appendix D, Fig. 17)
  * `generalization/`: Comparison of units localized on BLiMP versus other benchmarks (Appendix E, Fig. 18)
  * `mean-ablation/`: The mean ablation experiment (Appendix F, Fig. 19)
  * `scatterplot.png`/`pdf`: Correlation between cross-validation consistency and ablation effect (Appendix H, Fig. 21)
  * `cv-blimp-gemma/`: Cross-validation result for Gemma on BLiMP (upper subplot of Fig. 22)
  
* `multilingual/`: Data and figures for the multilingual experiments (Sec. 4.3)
  * `rublimp/`: Experiments with the RuBLiMP benchmark (left subplot of Fig. 6, upper subplot of Fig. 7, middle subplot of Fig. 22)
  * `sling/`: Experiments with the SLING benchmark (right subplot of Fig. 6, lower subplot of Fig. 7, lower subplot of Fig. 22)
  * `multiblimp/`: Experiment with the MultiBLiMP benchmark (Fig. 8)
  
* `benchmarks/`: Scripts for converting minimal pair benchmarks into the appropriate format for unit localization
  * `processed/`: Converted datasets
  * `raw/`: Original datasets were placed here
 
* `t-test/`: Utility for running one-sample and two-sample t-tests

* `cache/`: Storage for the localized units' masks

* `models/`: Modeling files for the LLMs considered in the paper (modified from respective files in the [transformers repo](https://github.com/huggingface/transformers/tree/main/src/transformers/models) to support ablation)

## Citation

```
@misc{syntax-units,
      title={Different types of syntactic agreement recruit the same units within large language models},
      author={Daria Kryvosheieva and Andrea de Varda and Evelina Fedorenko and Greta Tuckute},
      year={2025},
      eprint={2512.03676},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.03676},
}
```
