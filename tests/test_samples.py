"""Test samples for Binoculars AI text detection.

Contains realistic grant-writing samples in four categories:
1. HUMAN_SAMPLES — authentic academic/grant writing (expected: score < 0.85)
2. AI_SAMPLES — LLM-generated grant text (expected: score > 0.90)
3. MIXED_SAMPLES — human text with AI-edited portions
4. EDGE_CASES — short text, references, tables, non-English
"""

# ---------------------------------------------------------------------------
# Category 1: Human-Written Grant Sections
# These mimic real NIH grant writing — idiosyncratic phrasing, imperfect
# flow, domain-specific jargon, personal voice.
# ---------------------------------------------------------------------------

HUMAN_SAMPLES = [
    {
        "id": "human_specific_aims",
        "label": "human",
        "section": "Specific Aims",
        "text": (
            "Despite three decades of research, the mechanisms by which chronic "
            "stress accelerates atherosclerotic plaque formation remain poorly "
            "understood. Our lab has spent the past 8 years characterizing the "
            "neuroimmune axis in ApoE-/- mice and we keep running into the same "
            "puzzle: sympathetic nerve fibers innervating perivascular adipose "
            "tissue release norepinephrine at concentrations 10-fold higher than "
            "circulating levels, yet the downstream macrophage polarization "
            "profile doesn't match what you'd predict from catecholamine "
            "signaling alone. Something else is going on. We suspect co-released "
            "neuropeptide Y (NPY) acts synergistically with NE to skew "
            "macrophages toward an M1 phenotype via Y1R-mediated NF-kB "
            "activation. Preliminary data from our R21 (Fig. 2A) shows that "
            "Y1R blockade with BIBP3226 reduced plaque area by 38% in stressed "
            "mice (p=0.003, n=12/group), which was honestly a bigger effect "
            "than we expected."
        ),
    },
    {
        "id": "human_significance",
        "label": "human",
        "section": "Significance",
        "text": (
            "Pancreatic ductal adenocarcinoma remains one of the most lethal "
            "cancers, with a five-year survival rate stubbornly stuck around "
            "12%. The field has poured enormous effort into immunotherapy "
            "approaches, but PDAC tumors are notoriously immunologically cold. "
            "Part of the problem — and this is what makes our angle different — "
            "is that most studies focus on T cell exclusion from the tumor "
            "parenchyma and ignore what's happening in the perineural niche. "
            "We stumbled onto this in 2019 when a graduate student noticed that "
            "Schwann cells surrounding invaded nerves were expressing PD-L1 at "
            "levels comparable to tumor cells themselves. That finding sat in "
            "our notebooks for two years because we weren't sure it wasn't an "
            "artifact of our staining protocol. It wasn't. Three independent "
            "replications later, we believe perineural Schwann cell PD-L1 "
            "constitutes a previously unrecognized immune checkpoint niche."
        ),
    },
    {
        "id": "human_innovation",
        "label": "human",
        "section": "Innovation",
        "text": (
            "The proposed work introduces a dual-reporter system we call "
            "TRACE-seq (Temporal Recording of Allelic Chromatin Events by "
            "sequencing). Unlike existing lineage tracing tools like CellTagging "
            "or LARRY, TRACE-seq doesn't just tell you where a cell ended up — "
            "it records the sequence of chromatin accessibility changes the cell "
            "went through to get there. The trick is a pair of competing "
            "recombinases (Cre and Flp) whose activity is gated by different "
            "enhancer elements. As enhancers open and close during "
            "differentiation, they leave a combinatorial barcode in the genomic "
            "DNA. We've been optimizing the construct design for 14 months now "
            "and the error rate is down to roughly 2 misrecombinations per 1000 "
            "events (Supplementary Fig. S4). No one else has this kind of "
            "temporal resolution at single-cell level. The closest comparable "
            "tool is scDAM-ID, which measures chromatin contacts but cannot "
            "reconstruct the order in which they occurred."
        ),
    },
    {
        "id": "human_approach",
        "label": "human",
        "section": "Approach",
        "text": (
            "Aim 2 will test whether pharmacological inhibition of DYRK1A "
            "rescues the synaptic deficit in our Ts65Dn mouse model. We "
            "initially planned to use the DYRK1A inhibitor harmine, but pilot "
            "studies revealed unacceptable off-target effects on MAO-A at the "
            "doses needed for CNS penetrance (tremors, weight loss >15% in "
            "3/10 mice). Instead, we switched to the more selective compound "
            "ALGERNON (IC50 = 76 nM for DYRK1A vs >10 µM for MAO-A), which "
            "we obtain through an MTA with Dr. Bhatt's lab at Kyoto University. "
            "Mice will receive daily i.p. injections of ALGERNON (10 mg/kg) or "
            "vehicle for 21 days starting at P60. We chose P60 because our "
            "electrophysiology data (Fig. 4C) indicates that the LTP deficit "
            "in CA1 is fully established by this age but hippocampal plasticity "
            "is still responsive to intervention. Honestly if we'd started this "
            "aim two years ago we would have picked P30, but the newer data "
            "makes P60 the right call."
        ),
    },
    {
        "id": "human_preliminary_data",
        "label": "human",
        "section": "Preliminary Data",
        "text": (
            "In a pilot cohort (N=23 treatment-naive patients with MDD, 15F/8M, "
            "mean age 34.2±7.1), we collected fMRI during a monetary incentive "
            "delay task before and after 8 weeks of SSRI treatment. Contrary to "
            "our original hypothesis, ventral striatum activation during reward "
            "anticipation did NOT predict treatment response (r=0.08, p=0.72). "
            "However — and this was the surprise — the coupling between ventral "
            "striatum and dorsomedial prefrontal cortex during reward RECEIPT "
            "was strongly predictive (AUC=0.84, 95% CI [0.71, 0.97]). This "
            "finding pushed us to rethink the model entirely. The traditional "
            "view is that anhedonia reflects blunted reward anticipation, but "
            "our data suggest it may be more about impaired integration of "
            "reward outcomes into future behavior — a computation that requires "
            "intact striatal-prefrontal communication. This is consistent with "
            "recent work from Pizzagalli's group, though they framed it "
            "differently."
        ),
    },
]


# ---------------------------------------------------------------------------
# Category 2: AI-Generated Grant Sections
# Written to exhibit typical LLM patterns: smooth flow, generic phrasing,
# balanced structure, hedging, no personal voice or surprises.
# ---------------------------------------------------------------------------

AI_SAMPLES = [
    {
        "id": "ai_specific_aims",
        "label": "ai_generated",
        "section": "Specific Aims",
        "text": (
            "The long-term goal of this research program is to elucidate the "
            "molecular mechanisms underlying neuroinflammation in Alzheimer's "
            "disease and to identify novel therapeutic targets for intervention. "
            "Alzheimer's disease is a devastating neurodegenerative disorder "
            "that affects approximately 6.7 million Americans and represents "
            "a significant public health burden. Despite substantial advances "
            "in our understanding of amyloid-beta and tau pathology, the role "
            "of neuroinflammation in disease progression remains incompletely "
            "understood. Emerging evidence suggests that microglial activation "
            "and the subsequent release of pro-inflammatory cytokines, "
            "including interleukin-1 beta, tumor necrosis factor alpha, and "
            "interleukin-6, contribute to synaptic dysfunction and neuronal "
            "loss. The central hypothesis of this proposal is that targeting "
            "the NLRP3 inflammasome pathway in microglia will attenuate "
            "neuroinflammatory responses and preserve cognitive function in "
            "mouse models of Alzheimer's disease. This hypothesis is supported "
            "by our preliminary data demonstrating elevated NLRP3 expression "
            "in activated microglia surrounding amyloid plaques."
        ),
    },
    {
        "id": "ai_significance",
        "label": "ai_generated",
        "section": "Significance",
        "text": (
            "Cardiovascular disease remains the leading cause of morbidity and "
            "mortality worldwide, accounting for approximately 17.9 million "
            "deaths annually according to the World Health Organization. Heart "
            "failure, a common endpoint of various cardiovascular conditions, "
            "affects over 64 million people globally and is associated with "
            "significant healthcare costs and reduced quality of life. Current "
            "therapeutic approaches, including pharmacological interventions "
            "such as ACE inhibitors, beta-blockers, and mineralocorticoid "
            "receptor antagonists, have improved survival outcomes but have "
            "not adequately addressed the fundamental problem of cardiomyocyte "
            "loss and fibrotic remodeling. Therefore, there is an urgent and "
            "unmet need for innovative strategies that promote cardiac "
            "regeneration and restore functional myocardium. The proposed "
            "research addresses this critical gap by investigating the "
            "therapeutic potential of engineered extracellular vesicles derived "
            "from induced pluripotent stem cell-derived cardiomyocytes as a "
            "novel cell-free approach to cardiac repair."
        ),
    },
    {
        "id": "ai_innovation",
        "label": "ai_generated",
        "section": "Innovation",
        "text": (
            "The proposed research is innovative in several important respects. "
            "First, it represents the first systematic investigation of the "
            "intersection between epitranscriptomic modifications and immune "
            "cell metabolism in the context of solid tumor immunology. While "
            "previous studies have examined m6A modifications in tumor cells "
            "and others have characterized metabolic reprogramming in "
            "tumor-infiltrating lymphocytes, no prior work has integrated "
            "these two rapidly evolving fields to understand how RNA "
            "modifications regulate the metabolic fitness of anti-tumor "
            "immune responses. Second, our approach leverages cutting-edge "
            "single-cell multi-omics technologies, including paired "
            "scRNA-seq and scATAC-seq with simultaneous measurement of "
            "m6A modifications using the recently developed DART-seq protocol. "
            "This multi-modal approach will provide unprecedented resolution "
            "of the epitranscriptomic landscape at the single-cell level. "
            "Third, the computational framework we have developed integrates "
            "machine learning algorithms with mechanistic pathway modeling "
            "to identify causal relationships between RNA modifications and "
            "metabolic phenotypes."
        ),
    },
    {
        "id": "ai_approach",
        "label": "ai_generated",
        "section": "Approach",
        "text": (
            "To achieve the objectives outlined in Specific Aim 1, we will "
            "employ a comprehensive and rigorous experimental approach. We will "
            "first establish primary neuronal cultures from embryonic day 18 "
            "rat hippocampal tissue using well-established protocols previously "
            "optimized in our laboratory. Neurons will be cultured on "
            "poly-D-lysine-coated coverslips in Neurobasal medium supplemented "
            "with B27 and GlutaMAX for 14 days in vitro to ensure mature "
            "synaptic development. To investigate the role of BDNF-TrkB "
            "signaling in activity-dependent synaptic plasticity, we will "
            "utilize a combination of pharmacological and genetic approaches. "
            "Specifically, we will treat cultures with recombinant human BDNF "
            "at concentrations of 10, 50, and 100 ng/mL for 30 minutes, "
            "followed by electrophysiological recordings of miniature "
            "excitatory postsynaptic currents using whole-cell patch-clamp "
            "techniques. Statistical analyses will be performed using "
            "one-way ANOVA with Tukey's post-hoc test for multiple "
            "comparisons, with significance set at p < 0.05."
        ),
    },
    {
        "id": "ai_preliminary_data",
        "label": "ai_generated",
        "section": "Preliminary Data",
        "text": (
            "Our preliminary studies provide strong support for the proposed "
            "research and demonstrate the feasibility of our experimental "
            "approach. In an initial series of experiments, we performed "
            "comprehensive transcriptomic profiling of tumor-associated "
            "macrophages isolated from a cohort of 45 patients with "
            "non-small cell lung cancer using single-cell RNA sequencing. "
            "Our analysis revealed the existence of a previously "
            "uncharacterized macrophage subpopulation, which we designated "
            "as TAM-3, characterized by high expression of SPP1, TREM2, "
            "and MARCO. Importantly, the abundance of TAM-3 macrophages "
            "was significantly correlated with poor overall survival "
            "(hazard ratio = 2.34, 95% CI [1.56, 3.51], p < 0.001) and "
            "resistance to immune checkpoint inhibitor therapy. Furthermore, "
            "in vitro co-culture experiments demonstrated that TAM-3 "
            "macrophages suppressed CD8+ T cell proliferation by 67% "
            "compared to classical M1 macrophages (p < 0.0001, Student's "
            "t-test), suggesting a potent immunosuppressive function."
        ),
    },
]


# ---------------------------------------------------------------------------
# Category 3: Mixed Samples (Human Base + AI-Edited Portions)
# These simulate a common real-world scenario: a researcher drafts text,
# then uses an LLM to "polish" certain paragraphs.
# ---------------------------------------------------------------------------

MIXED_SAMPLES = [
    {
        "id": "mixed_human_start_ai_end",
        "label": "mixed",
        "section": "Significance (mixed)",
        "description": "First half human-written, second half AI-polished",
        "text": (
            # Human-written opening (informal, personal)
            "We got into this project almost by accident. A postdoc in the lab "
            "was running a proteomics experiment on retinal ganglion cells and "
            "noticed that the expression of complement component C1q was wildly "
            "elevated — like 15-fold over controls — in samples from diabetic "
            "donors. At first we assumed it was contamination or a batch effect. "
            "It took us almost a year to convince ourselves it was real, and "
            "another six months to figure out that the C1q was coming from the "
            "ganglion cells themselves, not from infiltrating immune cells. "
            # AI-polished continuation (smooth, formal, structured)
            "This finding has significant implications for our understanding of "
            "diabetic retinopathy pathogenesis. The complement system has "
            "traditionally been viewed as a component of innate immunity that "
            "operates primarily through serum proteins produced by hepatocytes. "
            "However, accumulating evidence from multiple laboratories has "
            "demonstrated that local complement production by resident tissue "
            "cells plays a critical role in maintaining homeostasis and "
            "responding to cellular stress. Our discovery that retinal ganglion "
            "cells upregulate C1q in the diabetic milieu suggests a novel "
            "autocrine or paracrine mechanism by which neurons may inadvertently "
            "tag their own synapses for elimination, thereby contributing to "
            "the synaptic loss that precedes overt neurodegeneration."
        ),
    },
    {
        "id": "mixed_ai_rewrite_of_human",
        "label": "mixed",
        "section": "Innovation (mixed)",
        "description": "Human concept, AI-rewritten for 'clarity'",
        "text": (
            "Our approach introduces several key innovations that distinguish "
            "it from existing methodologies in the field. The development of "
            "our novel microfluidic platform, which we have termed ChemoTrap, "
            "enables the simultaneous assessment of chemotactic responses "
            "across 96 independent gradient conditions using as few as 50,000 "
            "cells per experiment. This represents a significant advance over "
            "conventional Boyden chamber assays, which typically require "
            "500,000 cells per condition and can only test one gradient at a "
            "time. Furthermore, ChemoTrap incorporates real-time live-cell "
            "imaging capabilities that allow for the quantitative analysis of "
            "cell migration speed, directionality, and persistence at single-cell "
            "resolution. We have validated this platform using human neutrophils "
            "responding to fMLP gradients and demonstrated excellent "
            "reproducibility with a coefficient of variation of less than 8% "
            "across biological replicates."
        ),
    },
    {
        "id": "mixed_human_with_ai_sentences",
        "label": "mixed",
        "section": "Approach (mixed)",
        "description": "Mostly human with individual AI-generated sentences inserted",
        "text": (
            "For Aim 3, we'll use our existing colony of Nf1-flox mice "
            "crossed with GFAP-Cre to delete neurofibromin specifically in "
            "astrocytes. These mice develop optic pathway gliomas by 3 months "
            "of age with about 85% penetrance — we've been working with this "
            "model since 2017 and have a good handle on the variability. "
            "The proposed experiments will utilize state-of-the-art in vivo "
            "two-photon calcium imaging to characterize neuronal activity "
            "patterns in the visual cortex of tumor-bearing animals with "
            "unprecedented spatial and temporal resolution. "
            "We'll implant cranial windows over V1 at P45, before tumors are "
            "detectable on MRI, and start imaging weekly. The hard part is "
            "keeping the windows clear — we lose about 20% to bone regrowth "
            "or inflammation, which is why we always start with 15 mice per "
            "group rather than the 10 that the power analysis suggests."
        ),
    },
]


# ---------------------------------------------------------------------------
# Category 4: Edge Cases
# Designed to test boundaries of the detection system.
# ---------------------------------------------------------------------------

EDGE_CASES = [
    {
        "id": "edge_too_short",
        "label": "uncertain",
        "section": "Short Fragment",
        "description": "Below 50-token minimum — should be flagged as unreliable",
        "text": "The role of BRCA1 in DNA repair is well established.",
    },
    {
        "id": "edge_references",
        "label": "skip",
        "section": "References",
        "description": "Bibliography — formulaic regardless of authorship, should be excluded",
        "text": (
            "1. Smith JA, Johnson BC, Williams KL. Microglial activation in "
            "Alzheimer's disease: a systematic review and meta-analysis. "
            "J Neuroinflammation. 2023;20(1):145-162. doi:10.1186/s12974-023-02831-x. "
            "2. Chen Y, Liu M, Zhang R, et al. NLRP3 inflammasome activation "
            "promotes tau pathology in transgenic mice. Nat Neurosci. "
            "2022;25(3):388-401. doi:10.1038/s41593-022-01034-w. "
            "3. Park SH, Kim DW, Lee JH. Complement-mediated synaptic pruning "
            "in neurodegenerative disease. Annu Rev Neurosci. 2024;47:231-258. "
            "4. Rodriguez-Perez AI, Borrajo A, Diaz-Ruiz C, Garrido-Gil P, "
            "Labandeira-Garcia JL. Crosstalk between insulin-like growth factor-1 "
            "and angiotensin-II in dopaminergic neurons and glial cells: role in "
            "neuroinflammation and aging. Oncotarget. 2016;7(21):30049-30067."
        ),
    },
    {
        "id": "edge_technical_jargon",
        "label": "human",
        "section": "Methods (Dense Jargon)",
        "description": "Extremely dense technical writing — tests whether jargon skews scores",
        "text": (
            "Hippocampal slices (400 µm) were prepared from P21-P28 Sprague-Dawley "
            "rats using a Leica VT1200S vibratome in ice-cold NMDG-HEPES cutting "
            "solution (in mM: 92 NMDG, 2.5 KCl, 1.25 NaH2PO4, 30 NaHCO3, 20 "
            "HEPES, 25 glucose, 2 thiourea, 5 Na-ascorbate, 3 Na-pyruvate, 0.5 "
            "CaCl2, 10 MgSO4, pH 7.3-7.4 with HCl, 300-310 mOsm). Slices "
            "recovered in the same solution at 34°C for 12 min then transferred "
            "to room temp ACSF (in mM: 119 NaCl, 2.5 KCl, 1.25 NaH2PO4, 24 "
            "NaHCO3, 12.5 glucose, 2 CaCl2, 2 MgSO4). fEPSPs were recorded "
            "from stratum radiatum of CA1 using borosilicate glass electrodes "
            "(1-3 MΩ) filled with ACSF. Schaffer collaterals were stimulated "
            "at 0.05 Hz with a bipolar tungsten electrode. LTP was induced by "
            "TBS (5 bursts of 4 pulses at 100 Hz, 200 ms inter-burst interval, "
            "repeated 4x at 10 s intervals). We typically get 150-180% "
            "potentiation with this protocol."
        ),
    },
    {
        "id": "edge_non_english",
        "label": "uncertain",
        "section": "Spanish Abstract",
        "description": "Non-English text — Binoculars is English-centric, expect degraded accuracy",
        "text": (
            "El objetivo de este estudio es investigar los mecanismos moleculares "
            "que subyacen a la resistencia a la quimioterapia en el cáncer de "
            "mama triple negativo. Nuestros datos preliminares sugieren que la "
            "activación aberrante de la vía de señalización Notch contribuye "
            "a la supervivencia de las células madre tumorales después del "
            "tratamiento con doxorrubicina. Proponemos utilizar modelos de "
            "xenoinjerto derivados de pacientes para evaluar la eficacia de "
            "inhibidores gamma-secretasa en combinación con regímenes "
            "quimioterapéuticos estándar."
        ),
    },
]


# ---------------------------------------------------------------------------
# Combined list for iteration
# ---------------------------------------------------------------------------

ALL_SAMPLES = HUMAN_SAMPLES + AI_SAMPLES + MIXED_SAMPLES + EDGE_CASES


def get_samples_by_label(label: str) -> list[dict]:
    """Return all samples with a given expected label."""
    return [s for s in ALL_SAMPLES if s["label"] == label]


def get_sample_by_id(sample_id: str) -> dict | None:
    """Return a single sample by ID."""
    for s in ALL_SAMPLES:
        if s["id"] == sample_id:
            return s
    return None


# Quick summary when run directly
if __name__ == "__main__":
    print(f"Total test samples: {len(ALL_SAMPLES)}")
    from collections import Counter
    counts = Counter(s["label"] for s in ALL_SAMPLES)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print()
    for sample in ALL_SAMPLES:
        words = len(sample["text"].split())
        print(f"  [{sample['label']:>12}] {sample['id']:<35} ({words} words)")
