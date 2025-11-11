# Interface-Final Research Roadmap

This document captures the forward-looking plan for evolving the Interface-Final system into a publishable artifact for IEEE VIS, CHI, or a comparable tier-one venue. Each pillar blends LLM assistance with HCI rigor so that upcoming engineering tasks map directly to research contributions.

## 1. Vision & Core Questions
1. **How can large-language models accelerate experimental reproducibility for immunofluorescence workflows?**
2. **What interaction patterns keep expert users in control while LLMs generate study definitions, annotations, or figure-ready insights?**
3. **How do lightweight vision LLMs and statistics-aware UIs improve collaborative reasoning about NET formation or similar cellular phenomena?**

## 2. System Pillars (User Ideas + Extensions)

### 2.1 LLM-Guided Study Definition
- Integrate the Tokens API (or a fine-tuned local model) to collect study metadata. The agent guides users through input directory selection, subject grouping, replica counts, and threshold priors via a conversational wizard embedded in the left panel.
- Persist each decision as structured provenance (who/when/why) so re-running the study or handing it off to collaborators stays transparent.
- Research hook: log decision trajectories and compare time-to-configuration against the current manual workflow in a within-subjects study.

### 2.2 LLM Micro-Interpretations of Results
- After charts refresh, auto-generate short textual reads (e.g., “Group KO exceeds WT by 23% ±5% on NET area”). Tie each statement to the chart interaction history so reviewers can trace evidence.
- Support both guided (LLM asks for group aliases or biological context) and unguided modes where scientists supply their own vocabulary. Maintain an approval queue before anything is exported.
- Research hook: evaluate whether LLM summaries improve sensemaking accuracy/recall without inflating confirmation bias.

### 2.3 Lightweight vLLM Visual Reasoning
- Deploy a trimmed vLLM model (e.g., LLaVA-Med-lite) to scan immunofluorescence previews, flag anomalies (low SNR, segmentation drift), and recommend which figures best illustrate a manuscript’s claims.
- Present the suggestions directly inside PreviewPane with confidence tags and quick actions (“Use as overlay in Fig. 3”). Keep the human final say to remain CHI-compliant.
- Research hook: quantitative study on whether vLLM triage reduces per-study figure selection time and increases reviewer-rated clarity.

### 2.4 Publication Graphic Assistant (New)
- Combine metrics + vLLM cues to recommend visualization templates (bar+scatter, ridge plots, violin). Let users approve a layout and auto-generate the corresponding Plotly spec / Illustrator export.
- Highlight how the assistant enforces accessibility defaults (color-blind palettes, text contrast) to strengthen the HCI story.

### 2.5 Narrative Builder & Provenance
- Chain the LLM outputs (study setup, interpretations, figure picks) into a structured “analysis narrative.” Each sentence links to raw data, thresholds, and preview IDs.
- Enables replication packages and satisfies VIS/CHI expectations around audit trails.

## 3. Additional Eye-Opening Directions
1. **Active Critique Loop:** LLM monitors user adjustments (threshold drags, ratio edits), predicts likely frustrations, and offers just-in-time help (“You lowered Ch2 in KO but not WT; do you want to sync thresholds?”).
2. **Uncertainty Surfacing:** Blend bootstrap stats with LLM explanations so the assistant proactively states when data are under-powered.
3. **Multi-user Collaboration:** Introduce shared sessions where LLM acts as mediator, summarizing disagreements between pathologists and computational scientists and suggesting consensus actions.
4. **Bias & Ethics Monitor:** Track when study definitions omit metadata (sex, batch) and prompt users to justify exclusions—useful for CHI discussions on responsible automation.

## 4. Evaluation Blueprint
| Phase | Study Type | Participants | Metrics |
| --- | --- | --- | --- |
| Pilot | Think-aloud sessions on LLM-guided study builder | 4–6 lab members | Task time, NASA-TLX, qualitative friction themes |
| Main Study A | Controlled comparison of manual vs LLM-assisted configuration | 12 scientists | Accuracy of group/subject mapping, config completeness |
| Main Study B | Insight generation with vs without LLM explanations | 12 scientists | Correct inference rate, time to first correct interpretation |
| Supplementary | Figure selection w/ vLLM vs baseline | 8 scientists | Selection time, reviewer clarity ratings |

Include log analysis (token usage, overrides) and statistical tests (rm-ANOVA or mixed models) to match IEEE/CHI rigor.

## 5. Publication Fit & Timeline
- **Short term (workshops / BioVis):** Demonstrate LLM-guided configuration + interpretive summaries with qualitative evidence.
- **Full paper (VIS/CHI):** All four pillars plus mixed-method evaluations and reproducible artifacts (open-source prompts, anonymized previews, ablation on vLLM accuracy).
- Target 3–4 months for MVP of pillars 2.1–2.3, 2 additional months for narrative builder and evaluations, then paper writing sprint.

---
Use this roadmap to prioritize upcoming sprints; each shipped feature should trace back to a research question and planned evaluation.
