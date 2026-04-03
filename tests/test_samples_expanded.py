"""Expanded test samples for Binoculars threshold calibration.

20 additional grant-writing samples (10 human, 10 AI-generated) designed
to stress-test the Binoculars detector on academic/grant text.

Human samples exhibit: personal voice, hedging, self-correction, specific
lab details, imperfect transitions, informal asides.

AI samples exhibit: smooth polished flow, generic academic phrasing, perfect
paragraph structure, balanced hedging, no personal voice.
"""

# ---------------------------------------------------------------------------
# Human-Written Grant Sections (10 samples)
# Real-sounding grant prose with personal voice, rough edges, specifics.
# ---------------------------------------------------------------------------

HUMAN_SAMPLES_EXPANDED = [
    {
        "id": "human_exp_background_1",
        "label": "human",
        "section": "Background",
        "text": (
            "The idea that gut microbiota influence brain function sounded like "
            "hand-waving to most of us in the field until maybe 2015 or so. Our "
            "lab was firmly in the skeptics camp. But then we ran a pilot where "
            "we colonized germ-free C57BL/6J mice with stool from patients with "
            "treatment-resistant depression, and honestly the behavioral results "
            "knocked us sideways. The colonized mice showed a 40% reduction in "
            "sucrose preference (p=0.008, n=8/group) and spent significantly more "
            "time immobile in the forced swim test. We still aren't totally sure "
            "what's driving it — could be short-chain fatty acids, could be "
            "tryptophan metabolism, could be vagal signaling. Probably all three. "
            "The point is, we went from skeptics to writing this R01 in about "
            "eighteen months, which tells you how compelling the data were."
        ),
    },
    {
        "id": "human_exp_background_2",
        "label": "human",
        "section": "Background",
        "text": (
            "Chimeric antigen receptor T cell therapy has transformed outcomes in "
            "B-cell malignancies, but solid tumors remain a different beast entirely. "
            "We've been banging our heads against the immunosuppressive tumor "
            "microenvironment in pancreatic cancer for six years now. The main "
            "problem — and this took us embarrassingly long to figure out — isn't "
            "T cell trafficking. Our CAR-T cells actually get into the tumor just "
            "fine, at least in the KPC mouse model. They just die within 48 hours. "
            "Flow cytometry from day 2 explants shows massive upregulation of "
            "TIM-3 and LAG-3, which, okay, that's expected. But we also found "
            "elevated reactive oxygen species at levels that would kill any "
            "lymphocyte. The stellate cells are basically creating a free radical "
            "death zone. So now our approach is to armor the T cells with catalase "
            "overexpression before infusion."
        ),
    },
    {
        "id": "human_exp_rigor_1",
        "label": "human",
        "section": "Rigor and Reproducibility",
        "text": (
            "We learned the hard way about biological sex as a variable in this "
            "model. Our first two cohorts were all male mice because, frankly, "
            "that's what the postdoc had available at the time. When we finally "
            "ran females for the R21 resubmission, the effect size on hepatic "
            "steatosis was about half of what we'd reported. Reviewer 2 was not "
            "pleased, and rightly so. Going forward, all experiments in this "
            "proposal use equal numbers of male and female Ldlr-/- mice on a "
            "C57BL/6J background. We power each experiment for sex as a covariate "
            "in a two-way ANOVA, which means larger group sizes (n=15/sex/group "
            "instead of 10), and yes, that's reflected in the budget. Sample "
            "randomization uses a computer-generated block design and the "
            "histology scoring is done blinded by a pathologist in Dr. Wendt's "
            "lab who doesn't know our hypotheses."
        ),
    },
    {
        "id": "human_exp_rigor_2",
        "label": "human",
        "section": "Rigor and Reproducibility",
        "text": (
            "Antibody validation has been a real headache for this project. We "
            "went through four different anti-phospho-STAT3 (Y705) antibodies "
            "before finding one that gave consistent results in our IHC protocol "
            "on FFPE sections. The Cell Signaling 9145 clone works, but only "
            "if you do antigen retrieval with citrate buffer at pH 6.0 for "
            "exactly 20 minutes — 15 minutes gives patchy staining, 25 minutes "
            "destroys the morphology. We've documented all of this in our "
            "lab protocols on protocols.io (dx.doi.org/10.17504/protocols.io.xyz). "
            "For the RNA-seq experiments, we run each sample on a Bioanalyzer "
            "and reject anything with RIN below 8.0. Surprisingly, the mouse "
            "liver samples have been fine, but the human biopsies from our "
            "collaborator at Cleveland Clinic average around 7.2, so we may need "
            "to switch to a degradation-tolerant library prep like QuantSeq."
        ),
    },
    {
        "id": "human_exp_environment_1",
        "label": "human",
        "section": "Environment",
        "text": (
            "The PI's lab is in the Biomedical Research Building, which was "
            "renovated in 2019 and has dedicated BSL-2 space with two Class II "
            "biosafety cabinets, a Keyence BZ-X810 fluorescence microscope that "
            "we share with Dr. Patel's group, and a tissue culture room that, "
            "I'll be honest, is a bit cramped for the number of people using it. "
            "We've requested additional space in the new wing but that won't be "
            "ready until 2027. For the flow cytometry work, we use the university "
            "core facility, which has a BD FACSAria III sorter and a Cytek Aurora "
            "spectral analyzer. Wait times for the sorter can be 2-3 weeks during "
            "peak periods so we typically batch our sorting experiments. The "
            "mouse colony is maintained in a specific-pathogen-free barrier "
            "facility run by the Department of Comparative Medicine, with "
            "per diem costs of $0.85/cage/day."
        ),
    },
    {
        "id": "human_exp_environment_2",
        "label": "human",
        "section": "Environment",
        "text": (
            "Our computational work happens on a shared HPC cluster called "
            "Carbonate, which has 8 GPU nodes with NVIDIA A100s. Getting "
            "allocations is competitive — we currently have 50,000 GPU-hours "
            "per year through a departmental allocation, which was barely enough "
            "for last year's single-cell analysis. For this project we've "
            "requested a supplemental allocation of 100,000 hours. If that "
            "doesn't come through, we have a backup plan to use AWS Batch, "
            "though that eats into the supplies budget. The PI also has a local "
            "workstation with 128 GB RAM and an RTX 4090 for prototyping "
            "analyses before scaling to the cluster. For data storage we use "
            "the Research Data Archive which gives us 20 TB at no cost, plus "
            "we have an existing AWS S3 bucket for sharing large files with "
            "our collaborators at Karolinska. Honestly the data management "
            "aspect of this project is the thing that keeps me up at night."
        ),
    },
    {
        "id": "human_exp_budget_1",
        "label": "human",
        "section": "Budget Justification",
        "text": (
            "The postdoc salary (Dr. Kim, 100% effort) is the biggest line "
            "item and I want to explain why. This project requires someone who "
            "can do both the wet lab work (stereotaxic injections, "
            "electrophysiology, immunohistochemistry) and the computational "
            "analysis (spike sorting with Kilosort3, dimensionality reduction, "
            "Bayesian decoding models). That's a rare skill set and we're "
            "lucky to have Dr. Kim, who trained in Dr. Bhatt's systems neuro "
            "lab before joining us. The NIH salary cap applies here. For "
            "supplies: the Neuropixels 2.0 probes are $1,850 each and we "
            "typically get 3-4 good recording sessions before signal degrades. "
            "With 60 planned recording sessions across the aims, we need "
            "approximately 18 probes ($33,300). The AAV vectors for "
            "optogenetic constructs come from Addgene at $350/prep but we "
            "need high-titer custom preps from the Penn Vector Core ($2,800 "
            "per virus, 4 constructs)."
        ),
    },
    {
        "id": "human_exp_budget_2",
        "label": "human",
        "section": "Budget Justification",
        "text": (
            "Travel: we're requesting $3,500/year for the PI or postdoc to "
            "attend the Society for Neuroscience annual meeting and one "
            "specialized workshop. Last year the Computational Neuroscience "
            "workshop in Lisbon was incredibly useful — our postdoc came "
            "back with a completely new approach to fitting the drift-diffusion "
            "model that saved us probably two months of dead-end work. "
            "Publication costs: we budget $4,000/year based on our experience "
            "that open-access fees at journals like eLife and PNAS run between "
            "$2,500 and $5,000 per article. Mouse costs: 200 breeding pairs "
            "at $0.85/cage/day comes to roughly $62,000 over the project "
            "period. This seems like a lot but the Cre-dependent intersectional "
            "strategy requires maintaining four separate colonies. We looked "
            "into cryopreserving some lines to save money but the rederivation "
            "timeline doesn't work with the experimental schedule."
        ),
    },
    {
        "id": "human_exp_biosketch_1",
        "label": "human",
        "section": "Biosketch Narrative",
        "text": (
            "I started my independent lab in 2016 with an R00 transition award "
            "focused on circuit mechanisms of compulsive behavior. That first "
            "grant taught me a lot about what I didn't know — my initial aims "
            "were way too ambitious and I spent the first two years mostly "
            "troubleshooting fiber photometry in the dorsal striatum. But the "
            "struggle paid off. We published our first real paper in Neuron "
            "in 2019 showing that striatopallidal D2-MSN activity during "
            "reward omission was predictive of compulsive-like behavior in "
            "our progressive ratio task. Since then, we've built up to an "
            "R01-funded lab with two postdocs, three graduate students, and a "
            "research technician. I serve on the BRAIN Initiative review panel "
            "and I'm an associate editor at Neuropsychopharmacology. The "
            "current proposal builds directly on our expertise in striatal "
            "circuit dissection and extends it into a disease model we've "
            "been developing for two years."
        ),
    },
    {
        "id": "human_exp_biosketch_2",
        "label": "human",
        "section": "Biosketch Narrative",
        "text": (
            "My training is somewhat unusual for this field. I did my PhD in "
            "chemical engineering at MIT working on microfluidic devices, then "
            "pivoted to immunology during my postdoc with Dr. Bhatt at Stanford. "
            "People sometimes ask why the switch, and the honest answer is that "
            "I watched my father go through cancer treatment and realized I "
            "wanted to work on something with more direct clinical impact. The "
            "engineering background turns out to be incredibly useful though — "
            "our lab's main contribution has been developing high-throughput "
            "screening platforms for T cell function that nobody else has. "
            "We published the CytoScan platform in Nature Methods in 2021 "
            "and it's now being used by about 15 labs worldwide. I was promoted "
            "to Associate Professor with tenure in 2023. I also co-direct our "
            "department's immunoengineering graduate training program, which "
            "currently has 12 students from four departments."
        ),
    },
]


# ---------------------------------------------------------------------------
# AI-Generated Grant Sections (10 samples)
# Smooth, polished, generic academic writing with no personal voice.
# ---------------------------------------------------------------------------

AI_SAMPLES_EXPANDED = [
    {
        "id": "ai_exp_background_1",
        "label": "ai_generated",
        "section": "Background",
        "text": (
            "The gut-brain axis has emerged as a critical area of investigation "
            "in the neurosciences, with accumulating evidence demonstrating that "
            "the intestinal microbiome exerts profound influences on central "
            "nervous system function and behavior. Recent advances in "
            "metagenomic sequencing and gnotobiotic animal models have revealed "
            "that microbial metabolites, particularly short-chain fatty acids "
            "and tryptophan derivatives, can modulate neuronal activity through "
            "multiple signaling pathways, including vagal afferent stimulation, "
            "immune-mediated mechanisms, and direct neuroendocrine signaling. "
            "It is now well established that dysbiosis of the gut microbiome is "
            "associated with a range of neuropsychiatric conditions, including "
            "major depressive disorder, anxiety, and autism spectrum disorder. "
            "However, the precise molecular mechanisms by which specific "
            "microbial communities influence neural circuit function remain "
            "incompletely understood, representing a significant gap in our "
            "current knowledge that the proposed research seeks to address."
        ),
    },
    {
        "id": "ai_exp_background_2",
        "label": "ai_generated",
        "section": "Background",
        "text": (
            "Chimeric antigen receptor T cell therapy has revolutionized the "
            "treatment of hematological malignancies, with remarkable clinical "
            "responses observed in patients with relapsed or refractory B-cell "
            "lymphomas and acute lymphoblastic leukemia. Despite these "
            "significant achievements, the application of CAR-T cell therapy "
            "to solid tumors has been hampered by several formidable challenges, "
            "including the immunosuppressive tumor microenvironment, inadequate "
            "T cell infiltration, and the heterogeneous expression of target "
            "antigens. The tumor microenvironment in solid malignancies is "
            "characterized by elevated levels of immunosuppressive factors, "
            "including transforming growth factor beta, interleukin-10, and "
            "reactive oxygen species, which collectively impair T cell effector "
            "function and promote T cell exhaustion. Addressing these challenges "
            "requires the development of novel engineering strategies that "
            "enhance CAR-T cell persistence and function within the hostile "
            "tumor microenvironment."
        ),
    },
    {
        "id": "ai_exp_rigor_1",
        "label": "ai_generated",
        "section": "Rigor and Reproducibility",
        "text": (
            "The proposed research incorporates multiple strategies to ensure "
            "scientific rigor and reproducibility in accordance with NIH "
            "guidelines. All animal experiments will include both male and "
            "female subjects to account for potential sex-based differences in "
            "experimental outcomes. Sample sizes have been determined through "
            "rigorous power analyses based on preliminary data, with a target "
            "statistical power of 0.80 and an alpha level of 0.05. "
            "Randomization of experimental groups will be performed using "
            "computer-generated random number sequences, and investigators "
            "conducting behavioral assessments and histological analyses will "
            "be blinded to treatment conditions. All key findings will be "
            "validated through independent replication by at least two "
            "laboratory members. Data will be recorded in electronic laboratory "
            "notebooks with automated version control and timestamping. "
            "Authentication of cell lines will be performed using STR profiling, "
            "and all antibodies will be validated using appropriate positive "
            "and negative controls as recommended by the International Working "
            "Group for Antibody Validation."
        ),
    },
    {
        "id": "ai_exp_rigor_2",
        "label": "ai_generated",
        "section": "Rigor and Reproducibility",
        "text": (
            "To ensure the highest standards of scientific rigor, the proposed "
            "studies will employ a comprehensive quality control framework "
            "throughout all phases of the research. Biological samples will be "
            "processed using standardized operating procedures that have been "
            "validated and documented in our laboratory protocols. RNA integrity "
            "will be assessed using the Agilent Bioanalyzer system, with a "
            "minimum RNA Integrity Number of 8.0 required for downstream "
            "sequencing applications. For immunohistochemical analyses, antibody "
            "specificity will be confirmed through the use of knockout tissue "
            "controls and peptide competition assays. All quantitative analyses "
            "will be performed by investigators blinded to experimental "
            "conditions, and results will be independently verified by a second "
            "analyst. Statistical methods will be selected a priori based on "
            "the distribution characteristics of the data, with parametric "
            "tests used when assumptions of normality are met and non-parametric "
            "alternatives employed otherwise."
        ),
    },
    {
        "id": "ai_exp_environment_1",
        "label": "ai_generated",
        "section": "Environment",
        "text": (
            "The research environment at the host institution provides "
            "outstanding resources and infrastructure to support the successful "
            "completion of the proposed research. The principal investigator's "
            "laboratory is located in the state-of-the-art Biomedical Research "
            "Center, which was constructed in 2018 and features modern "
            "laboratory spaces equipped with advanced instrumentation. The "
            "facility includes dedicated tissue culture rooms with class II "
            "biological safety cabinets, a molecular biology suite with "
            "real-time PCR systems and gel documentation equipment, and a "
            "shared imaging core facility housing confocal and super-resolution "
            "microscopy platforms. The institution's animal facility is "
            "AAALAC-accredited and provides comprehensive veterinary support "
            "and specialized housing for transgenic mouse colonies. The "
            "computational resources available include a high-performance "
            "computing cluster with GPU nodes suitable for machine learning "
            "applications and bioinformatic analyses."
        ),
    },
    {
        "id": "ai_exp_environment_2",
        "label": "ai_generated",
        "section": "Environment",
        "text": (
            "The institutional environment is highly conducive to the proposed "
            "research and provides a collaborative and intellectually "
            "stimulating setting for interdisciplinary investigation. The "
            "Department of Biomedical Engineering, in which the principal "
            "investigator holds a primary appointment, has experienced "
            "significant growth in recent years, with the recruitment of "
            "several new faculty members with complementary expertise in "
            "biomaterials, tissue engineering, and computational biology. "
            "The department maintains shared core facilities for materials "
            "characterization, including scanning electron microscopy, "
            "X-ray diffraction, and mechanical testing equipment. Additionally, "
            "the principal investigator benefits from close proximity to the "
            "School of Medicine, which facilitates productive collaborations "
            "with clinical investigators and provides access to patient "
            "samples through established tissue banking programs. The "
            "institution has demonstrated a strong commitment to supporting "
            "early-career investigators through startup packages, mentoring "
            "programs, and protected research time."
        ),
    },
    {
        "id": "ai_exp_budget_1",
        "label": "ai_generated",
        "section": "Budget Justification",
        "text": (
            "Personnel costs represent the largest component of the proposed "
            "budget and are essential for the successful execution of the "
            "research plan. The principal investigator will devote 30% effort "
            "to this project, providing scientific direction, experimental "
            "design, data interpretation, and manuscript preparation. A "
            "postdoctoral research associate will be supported at 100% effort "
            "to conduct the primary experimental work described in Specific "
            "Aims 1 and 2, including cell culture, molecular biology assays, "
            "and animal model experiments. A graduate research assistant at "
            "50% effort will support the computational analyses described in "
            "Specific Aim 3, including bioinformatic pipeline development and "
            "statistical modeling. Supply costs include reagents for molecular "
            "biology experiments, cell culture media and supplements, antibodies "
            "for immunoassays, and sequencing library preparation kits. Animal "
            "costs are calculated based on the number of experimental animals "
            "required as determined by our power analyses."
        ),
    },
    {
        "id": "ai_exp_budget_2",
        "label": "ai_generated",
        "section": "Budget Justification",
        "text": (
            "Travel funds are requested to support the dissemination of research "
            "findings at national and international scientific conferences. The "
            "principal investigator and postdoctoral associate will attend "
            "two conferences annually, including the Annual Meeting of the "
            "American Association for Cancer Research and the International "
            "Conference on Tumor Microenvironment. These meetings provide "
            "valuable opportunities for presenting research findings, receiving "
            "feedback from the scientific community, and establishing "
            "collaborative relationships with investigators at other "
            "institutions. Publication costs are budgeted to support open-access "
            "publication of research findings in peer-reviewed journals, in "
            "accordance with NIH public access policies. Equipment funds are "
            "requested for the purchase of a benchtop flow cytometer that will "
            "significantly enhance the efficiency of routine immunophenotyping "
            "experiments and reduce dependence on the shared core facility, "
            "thereby accelerating the pace of the proposed research."
        ),
    },
    {
        "id": "ai_exp_biosketch_1",
        "label": "ai_generated",
        "section": "Biosketch Narrative",
        "text": (
            "The principal investigator has established a robust and productive "
            "research program focused on understanding the molecular mechanisms "
            "of neurodegeneration and developing novel therapeutic strategies "
            "for Alzheimer's disease. Since establishing an independent "
            "laboratory in 2017, the principal investigator has secured "
            "extramural funding totaling over 3.5 million dollars, including "
            "an R01, an R21, and an Alzheimer's Association Research Grant. "
            "The laboratory has published 28 peer-reviewed manuscripts in "
            "high-impact journals, including Nature Neuroscience, Cell Reports, "
            "and the Journal of Neuroscience. The principal investigator serves "
            "on multiple NIH study sections and editorial boards, reflecting "
            "recognition as an established expert in the field. The research "
            "team currently comprises two postdoctoral fellows, three graduate "
            "students, and two undergraduate researchers, providing a "
            "collaborative and mentored training environment."
        ),
    },
    {
        "id": "ai_exp_biosketch_2",
        "label": "ai_generated",
        "section": "Biosketch Narrative",
        "text": (
            "The principal investigator brings a unique and highly relevant "
            "combination of expertise in biomedical engineering and immunology "
            "that is ideally suited for the proposed research. Following "
            "doctoral training in biomedical engineering at a leading research "
            "university, the principal investigator completed postdoctoral "
            "training in tumor immunology, where foundational expertise in "
            "the design and characterization of immunomodulatory biomaterials "
            "was developed. This interdisciplinary training has enabled the "
            "establishment of a research program at the intersection of "
            "materials science and cancer immunology. The principal investigator "
            "has been recognized with several prestigious awards, including "
            "a National Science Foundation CAREER Award and a Department of "
            "Defense Breakthrough Award. These accomplishments demonstrate "
            "the productivity, innovation, and scientific leadership necessary "
            "to ensure the successful completion of the proposed research "
            "program."
        ),
    },
]


# ---------------------------------------------------------------------------
# Combined lists
# ---------------------------------------------------------------------------

ALL_EXPANDED_SAMPLES = HUMAN_SAMPLES_EXPANDED + AI_SAMPLES_EXPANDED


def get_expanded_samples_by_label(label: str) -> list[dict]:
    """Return expanded samples with a given label."""
    return [s for s in ALL_EXPANDED_SAMPLES if s["label"] == label]


if __name__ == "__main__":
    print(f"Expanded test samples: {len(ALL_EXPANDED_SAMPLES)}")
    from collections import Counter
    counts = Counter(s["label"] for s in ALL_EXPANDED_SAMPLES)
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print()
    for sample in ALL_EXPANDED_SAMPLES:
        words = len(sample["text"].split())
        print(f"  [{sample['label']:>12}] {sample['id']:<35} ({words} words)")
