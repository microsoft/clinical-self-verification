# choice: don't include "and" in any of the labels
# choice: don't include "control" but do include "placebo"
ANNOTS = {
    "10070173": [
        "budesonide Turbuhaler",
        "budesonide aqua",
    ],
    "10390665": ["perphenazine", "granisetron"],
    "10475150": [
        "somatostatin",
    ],
    "10578479": ["soy", "normal diet"],
    "10589810": [
        "plastic curettes",
        "conventional steel curettes",
    ],
    "10763172": [
        "Dexpanthenol",
        "xylometazoline",
        "verum",
        "placebo",
    ],
    "10764172": ["morphine"],
    "10912743": ["vitamin E", "aspirin"],
    "10934569": [
        "intensive treatment",
        "parent training",
        # "intensive early intervention",
    ],
    "10940525": [
        "Selamectin",
        "fenthion",
        "pyrethrins",
        "ivermectin",
    ],
    "11099086": ["ciprofloxacin", "ceftriaxone", "placebo", "im placebo"],
    "11229858": [
        "antipyretics",
        "naproxen",
        "metamizol",
    ],
    "11099086": ["ciprofloxacin", "ceftriaxone"],
    "11229858": ["antipyretic drugs", "antipyretics", "naproxen", "metamizol"],
    "1131298": [
        "garlic juice",
        "onion juice",
        "garlic essential oil",
        "onion essential oil",
    ],
    "11317090": [
        "aquatic exercise classes",
    ],
    "11381289": [
        "ramipril",  # "ACE inhibitor",
        # "angiotensin-converting enzyme ( ACE ) inhibitor",
        "vitamin E",
    ],
    "11401641": ["latanoprost", "timolol", "timolol + latanoprost"],
    "11454878": [
        "pegylated liposomal doxorubicin",
        "topotecan",
        # "pegylated liposomal doxorubicin ( PLD",
        # "PLD",
    ],
    "11495215": [
        "Oral contraceptive",
        # "OC",
        "cigarette smoking",
        "nicotine",
        # "nicotine deprivation",
    ],
    "11642083": [
        "Neoton",
        "thrombolytic therapy",
        # "thrombolytic therapy ( TLT",
        # "TLT",
        # "streptokinase",
        "streptokinase preparations",
    ],
    "11737955": [
        # "Antioxidant supplementation",
        # "antipyrine hydroxylates",
        "dl-alpha-tocopheryl acetate",
        "ascorbic acid",
        "beta-carotene",
        "placebo",
        # "Antipyrine",
    ],
    "11750293": ["standard outpatient clinic treatment"],
    "11829043": [
        "progressive muscle relaxation training",
        # "progressive muscle relaxation training ( PMRT",
        # "PMRT",
    ],
    "11891832": ["methotrexate", "placebo"],
    "12139812": [
        "Limbal epithelial autograft transplantation",
        "pterygium excision",
        # "excision of pterygium",
        # "limbal epithelial autograft transplantation surgery",
        # "simple pterygium excision",
    ],
    "12477021": [
        "Reichert AT550",
        "Reichert Xpert Plus",
        "Goldmann applanation tonometer",
        "Perkins tonometer",
    ],
    "12576806": [
        "stenting",
        # "endopyelotomy",
        # "Internal stenting",
        # "stent",
        # "7/14Fr internal endopyelotomy stent placement",
    ],
    "12586799": [
        "concurrent chemoradiotherapy",
        "radiotherapy",
    ],
    "12595499": ["placebo", "Ramipril"],
    "12709693": [
        "remifentanil",
        "propofol",
        "fentanyl",
        "midazolam",
    ],
    "1286547": ["nitrendipine", "placebo"],
    "12960652": ["Epigastric impedance"],
    "1459268": [
        "gonadotropin-releasing hormone agonist",
        # "gonadotropin-releasing hormone agonist ( GnRH-a",
        # "GnRH-a",
    ],
    "14679127": ["topotecan", "paclitaxel"],
    "14739125": [
        "torcetrapib",
        # "cholesteryl ester transfer protein ( CETP ) inhibitor torcetrapib ( CP-529,414",
        "placebo",
    ],
    "14763035": [
        "Valsartan",
        # "selective angiotensin II receptor blocker ( ARB",
        # "ARB valsartan",
        "placebo",
        # "ARB",
    ],
    "15014018": [
        # "interferon",
        "IFN-alpha 2b",
        "observation",
        # "IFN-alpha 2b ( HDI",
        # "Obs",
        # "HDI",
    ],
    "15133359": [
        # "guiding catheter alone",
        # "electrophysiology catheter advanced",
        # "electrophysiology catheter aided ",
        "EPA",
        "GCA",
    ],
    "15193668": [
        # "Combined descriptive and explanatory information",
        "descriptive and explanatory information",
        # "descriptive ( AUT-D",
        # "descriptive and explanatory information ( AUT-D + E )",
        "descriptive information",
        # "combination of descriptive and explanatory information",
    ],
    "15358868": ["age"],  # this is really hard, not explicitly in text
    "15848261": [
        # "conventional physical therapy alone",
        # "with a specialised balance retraining program",
        "conventional therapy",
        "therapy and retraining",
        # "conventional therapy",
        # "therapy combined with standing balance training by biofeedback ( BPM Monitor )",
    ],
    "15854186": ["placebo", "cophenylcaine spray"],
    "15858959": [
        "Saccharomyces boulardii",
        # "S. boulardii",
        # "placebo ( placebo",
        "placebo",
    ],
    "16055807": [
        "Comprehensive cognitive-behavioural therapy",
        "atypical neuroleptic amisulpride",
    ],
    "16167234": [
        "branch chain amino acid enriched formula",
        # "branch chain amino acid ( BCAA ) enriched formula",
        # "BCAA enriched",
        "routine amino acid",
        # "BCAA",
        # "BCAA-enriched",
        # "BCAA enriched formula",
    ],
    "16295154": [
        # "Intervention",
        "nursing intervention",
        # "post-diagnosis nursing intervention",
        # "usual care",
        # "contact with a pediatric nurse practitioner ( PNP ) for counseling , instruction , and assistance with implementation of the recommended treatment plan",
        # "consultation session to receive the results of diagnostic tests and a written copy of the recommended treatment plan",
    ],
    "16427787": ["melatonin", "placebo"],
    "16505427": ["erythropoietin", "epoetin alfa"],
    "1669598": [
        "local immunotherapy",
        # "powder extract of house dust mite ( HDM",
        "parenteral immunotherapy",
        "placebo",
        "lactose",
        "disodium cromoglycate",
        # "DSCG",
    ],
    "17293018": [
        # "whey protein concentrate",
        # "immunized against Clostridium difficile ( C. difficile ) and its toxins , toxin A and toxin B",
        # "anti-C. difficile whey protein concentrate ( anti-CD WPC",
        "anti-CD WPC",
        # "regular whey protein concentrate",
    ],
    "17321989": ["itraconazole", "fluconazole"],
    "17616069": [
        "trivalent ferrum preparation",
        # "sacharose ferric oxide ( Venofer",
        # "Venofer",
        # "trivalent saccharose ferric oxide",
    ],
    "1780092": [
        "nimesulide",
        # "non-steroid anti-inflammatory drug . Nimesulide",
        "flurbiprofen",
    ],
    "17897478": [
        "Salmeterol",
        "fluticasone propionate",
        "montelukast",
        # "corticosteroids",
        "Seretide",
        # "FP",
        # "montelukast ( FP/M",
        # "SFC",
        # "FP/M",
        # "FP/M . FP/M",
    ],
    "18077611": ["Zostavax frozen", "Zostavax refrigerated"],
    "18189160": [
        "FEC",
        "epirubicin",
        # "fluorouracil/epirubicin/cyclophosphamide ( FEC",
        "pegfilgrastim",
        # "FEC ( 90 )",
    ],
    "18229990": [
        "joint attention",
        "symbolic play",
        # "joint attention ( JA",
        # "symbolic play ( SP",
        # "JA",
        # "SP",
    ],
    "18337284": ["amoxicillin", "penicillin V"],
    "18503531": [
        "multi-component social skills intervention",
        # "Junior Detective Training Program",
        # "small group sessions , parent training sessions",
        # "teacher handouts",
    ],
    "18544974": [
        "galacto-oligosaccharides",  # and Lactobacillus GG","Lactobacillus GG",
        # "Lactobacillus GG ( LGG ) and galacto-oligosaccharides ( GOS",
        # "LGG",
        # "GOS",
        # "LGG + GOS",
    ],
    "18773733": ["Vibrotactile stimuli" "vibrotactile stimulus", "vibrotaction"],
    "18783504": [
        "bupropion sustained-release",
        # "bupropion sustained-release ( SR",
        # "bupropion SR",
        "placebo",
    ],
    "18845996": [
        "doxorubicin",
        "pegylated liposomal doxorubicin",
        # "doxorubicin ( DOX",
        # "pegylated liposomal formulation ( PLD",
        # "DOX",
        # "PLD",
    ],
    "18958161": [
        "own face",
        "faces of others",
        # "images of the subjects ' own face and to faces of others",
        # "their own face and a gender-matched other face",
        # "images",
        # "self- and other-processing",
    ],
    "18975051": ["iopamidol 300", "iopamidol 370"],
    "19096921": [
        "medication",
        "parent training",
        "risperidone",
        # "structured parent training",
    ],
    "19108790": [
        "propofol",
        "droperidol",
        "metoclopramide",
        # "propofol",
        # "placebo",
        # "droperidol",
        "placebo",
        # "sevoflurane",
        # "oxygen",
    ],
    "19159999": [
        "nifedipine sustained release",
        "Ginkgo biloba extract",
        # "nifedipine sustained release ( nifedipine SR",
        # "nifedipine SR",
    ],
    "19301724": [
        "continuous femoral nerve block",
        "continuous epidural infusion",
        # "femoral nerve block ( FNB",
        # "CEI",
        "levobupivacaine",
        "morphine",
        # "CFNB",
    ],
    "19376304": [
        "warfarin",
        "dabigatran",
        #   . Vitamin K antagonists",
        # "Dabigatran etexilate",
        # "dabigatran",
    ],
    "1963383": [
        "liu wei di huang decoction",
        "jin gui shen qi decoction",
        # "Kidney-tonifying decoction ( Liu Wei Di Huang",
        # "Chinese herb",
        # "Kidney-tonifying decoction",
        # "Chinese Kidney-tonifying decoction",
    ],
    "1968178": [
        "self-inflation of the cuff",
        "wearing the inflated cuff",
        # "Inflating the cuff",
        # "inflated cuff",
        # "cuff inflation",
        # "inflate their cuff",
    ],
    "19708562": [
        "transcranial electrostimulation",
        "traditional treatment",
        # "transcranial electrotherapy",
        # "traditional therapy plus electrostimulation",
    ],
    "19727232": [
        "auditory integrative training",
        "no training",
        # "auditory integrative training ( AIT",
    ],
    "19802506": [
        "weight training",
        # "weight training exercises",
        "risedronate",
        "calcium",
        "vitamin D",
        # "strength/weight training ( ST ) exercises",
        # "exercise plus medication",
        # "medication",
        # "calcium",
        # "risedronate",
        # "exercise",
        # "ST exercises",
        # "exercising",
        # "exercised",
        # "Strength/weight training exercises",
    ],
    "20147856": [
        "gemcitabine",
        "S-1",
        # "of gemcitabine",
        # "with S-1",
        # "oral S-1",
        # "with gemcitabine",
        # "of gemcitabine and S-1",
    ],
    "20189026": [
        "Remote ischaemic conditioning",
        # "Remote ischaemic preconditioning",
        # "remote conditioning",
    ],
    "20226474": [
        # "Cryoprobe biopsy",
        # "cryoprobe",
        # "cryobiopsies",
        "cryobiopsy",
        "forceps biopsy",
        # "forceps biopsy and cryobiopsy",
    ],
    "20509069": [
        "Virtual reality hypnosis",
        "Virtual reality without hypnosis",
        # "virtual reality hypnosis ( VRH ) -hypnotic induction",
        # "VRH",
        # "VR without hypnosis",
        "standard care",
    ],
    "20560622": [
        # "Maillard reaction products",
        # "Maillard reaction products ( MRP",
        "white diet",
        "brown diet",
        # "MRP",
    ],
    "20618920": ["hygienic-dietary recommendations", "antidepressant"],
    "2066441": [
        "tetracycline-immobilized collagen film",
        "root planing",
    ],
    "20739054": [
        "Eltrombopag",
        "placebo",
    ],
    "20804366": [
        "acupuncture",
        "sham acupuncture",
        # "tongue acupuncture",
        # "sham tongue acupuncture",
    ],
    "2088233": ["doxycycline", "placebo"],
    "20943715": [
        # "Visual and kinesthetic locomotor imagery training integrated with auditory step rhythm",
        "visual and kinesthetic locomotor imagery training",
        "auditory step rhythm",
    ],
    "21029638": ["paliperidone", "risperidone"],
    "21106663": ["alcohol intervention"],
    "21170734": [
        "anesthesia",
        "l⁻¹ clove solution",
    ],
    "21211686": [
        "pioglitazone",
        "glimepiride",
        # "Pioglitazone-treated",
        # "pioglitazone-induced",
    ],
    "21393467": ["perturbations to treadmill walking"],
    "21491990": [
        "chlorhexidine",
        # "chlorhexidine ( CHX",
        "prophylaxis",
        "placebo",
    ],
    "21669557": [
        # "chimney woodstove",
        "stove intervention",
        "open fire",
    ],
    "21718094": [
        "MDI therapy",
        # "starting insulin pump",
        # "daily insulin",
        # "comparing MDI",
        "subcutaneous insulin therapy",
        # "of pump therapy",
        # "in insulin",
        # "of insulin pump therapy",
    ],
    "21866656": [
        "laparoscopic surgery",
        "quyu jiedu recipe",
        # "Quyu Jiedu Recipe ( QYJDR",
        "progesterone",
        # "QYJDR",
    ],
    "2188766": ["flosequinan", "placebo"],
    "21896676": [
        "physiotherapy service",
        # "physiotherapy",
    ],
    "21914768": [
        "Testicular cancer + TSE information",
        "Testicular cancer information",
        "Testicular self-examination",
    ],
    "21924760": [
        "Time in therapeutic range",
        # "mean TTR",
        # "CONCLUSIONS",
        # "Maintaining a high",
    ],
    "21975269": [
        "hormonal contraceptives",
    ],
    "21982657": [
        "skeletal myoblast implantation",
        # "myoblasts",
        "placebo",
        # "myoblast-treated",
        # "of myoblasts",
        # "by myoblast",
    ],
    "22154769": [
        "zoster vaccine",
        "placebo",
    ],
    "2223363": ["propofol"],
    "2224060": ["bestatin"],
    "22271197": ["psychosocial interventions"],
    "22341427": ["cardiac transplantation"],
    "22416755": [
        "pretense",
    ],
    "22420121": [
        "referential communication training",
    ],
    "22435114": [
        "Group cognitive behavior therapy",
        "treatment-as-usual",
    ],
    "22538329": ["spotty calcification"],
    "22646975": [
        "relaxation training",
    ],
}
