from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
PREDICTION_DIR = BASE_DIR / "predictions"

UD_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh",
    "English-Penn": "en",
    "English-EWT": "en",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Russian": "ru",
    "Tamil": "ta",
    "Uyghur": "ug",
    "Wolof": "wo"
    }

# language code reference: https://github.com/stanfordnlp/stanza/blob/dev/stanza/models/common/constant.py
STNZ_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh-hans",
    "English-Penn": "en",
    "English-EWT": "en",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Russian": "ru",
    "Tamil": "ta",
    "Uyghur": "ug",
    "Wolof": "wo"
    }

DIR_ABBR_LOOKUP = {
    "Ancient_Greek": "grc",
    "Chinese": "zh-hans",
    "English-Penn": "en-penn",
    "English-EWT": "en-ewt",
    "Finnish": "fi",
    "French": "fr",
    "Hebrew": "he",
    "Russian": "ru",
    "Tamil": "ta",
    "Uyghur": "ug",
    "Wolof": "wo",
    "combined": "combined"
    }


TREEBANK_LOOKUP = {
    "Ancient_Greek": "Perseus",
    "Chinese": "Penn",
    "English-Penn": "Penn",
    "English-EWT": "EWT",
    "Finnish": "TDT",
    "French": "GSD",
    "Hebrew": "HTB",
    "Russian": "GSD",
    "Tamil": "TTB",
    "Uyghur": "UDT",
    "Wolof": "WTB"
    }

def remove_treebank(lang):
    if lang in ["English-Penn", "English-EWT"]:
        lang = lang.split("-")[0]
    return lang

def check_dir(output_dir):
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created: {output_dir}")

# def get_vocab_dir():
#     folder = BASE_DIR / "artifacts" 
#     check_dir(folder)
#     return folder

@dataclass
class DataPaths:
    lang: str
    split: str

    @property
    def ud_lang(self):
        return remove_treebank(self.lang)

    @property
    def ud_abbr(self):
        return UD_ABBR_LOOKUP[self.lang]

    @property
    def dir_abbr(self):
        return DIR_ABBR_LOOKUP[self.lang]
    
    @property
    def treebank(self):
        return TREEBANK_LOOKUP[self.lang]
    
    def raw(self) -> Path:
        folder = DATA_DIR / "raw" / f"UD_{self.ud_lang}-{self.treebank}"
        path = f"{self.ud_abbr}_{self.treebank.lower()}-ud-{self.split}.conllu"
        check_dir(folder)
        return folder / path

    def upstream(self) -> Path:
        folder = DATA_DIR / "upstream" / f"UD_{self.ud_lang}-{self.treebank}"
        path = f"{self.ud_abbr}_{self.treebank.lower()}-ud-{self.split}.conllu"
        check_dir(folder)
        return folder / path

    def projectivized(self, head=None, path=None) -> Path:
        if head is None and path is None:
            greatgrandparent = "common"
            parent = f"UD_{self.ud_lang}-{self.treebank}"
        else:
            greatgrandparent = "label_experiments"
            parent = f"lang={self.dir_abbr},head={head},path={path}"
        folder = DATA_DIR / greatgrandparent / "projectivized" / parent
        path = f"{self.ud_abbr}__{self.split}.conllu"
        check_dir(folder)

    def constituentized(self, pos, head=None, path=None) -> Path:
        if head is None and path is None:
            greatgrandparent = "common"
            parent = f"lang={self.dir_abbr},pos={pos.lower()}"
        else:
            greatgrandparent = "label_experiments"
            path = f"lang={self.dir_abbr},pos={pos.lower()},head={head},path={path}"
        folder = DATA_DIR / greatgrandparent / "constituentized" / parent
        path = f"{self.ud_abbr}__{self.split}.conllu"
        check_dir(folder)
        return folder / path

    def records(self, head, path) -> Path:
        folder = DATA_DIR / "label_experiments" / "projectivized" / f"lang={self.dir_abbr},head={head},path={path}"
        path = f"sentence_added_to_{self.split}.conllu"
        return folder / path

@dataclass
class UpstreamPredictionPaths:
    lang: str
    pos: str
    split: str
    # epochs: int

    @property
    def ud_lang(self):
        return remove_treebank(self.lang)
    
    @property
    def ud_abbr(self):
        return UD_ABBR_LOOKUP[self.lang]

    @property
    def dir_abbr(self):
        return DIR_ABBR_LOOKUP[self.lang]
    
    @property
    def treebank(self):
        return TREEBANK_LOOKUP[self.lang]
    
    def upstream_output(self, epochs) -> Path:
        folder = PREDICTION_DIR / "raw"
        path = f"lang={self.dir_abbr},split={self.split},pos={self.pos.lower()},epochs={epochs}.mrg"
        check_dir(folder)
        return folder / path
    
    def linearized(self, is_target) -> Path:
        folder = DATA_DIR / "downstream" / "linearized" / f"lang={self.dir_abbr},pos={self.pos.lower()}"
        path = f"{self.split}.tgt.txt" if is_target else f"{self.split}.src.txt"
        return folder / path

    def language_file(self) -> Path:
        folder = DATA_DIR / "downstream" / "linearized" / f"lang={self.dir_abbr},pos={self.pos.lower()}"
        path = f"{self.split}.lang.txt"
        check_dir(folder)
        return folder / path

    def edit_actions(self) -> Path:
        folder = DATA_DIR / "downstream" / "edit_actions" / f"lang={self.dir_abbr},pos={self.pos.lower()}"
        path = f"{self.split}.json"
        check_dir(folder)
        return folder / path
        
    def structured_tokens(self, overlap=0) -> Path:
        filename = f"lang={self.dir_abbr},pos={self.pos.lower()},overlap={overlap}" if overlap > 0 else f"lang={self.dir_abbr},pos={self.pos.lower()}"
        folder = DATA_DIR / "downstream" / "structured_tokens" / filename
        path = f"{self.split}.json"
        check_dir(folder)
        return folder / path

@dataclass
class FinalPredictionPaths:
    lang: str
    pos: str
    epochs: int = None
    gate: bool = None
    head: bool = None
    path: bool = None

    @property
    def ud_lang(self):
        return remove_treebank(self.lang)
    
    @property
    def ud_abbr(self):
        return UD_ABBR_LOOKUP[self.lang]

    @property
    def dir_abbr(self):
        return DIR_ABBR_LOOKUP[self.lang]
    
    @property
    def treebank(self):
        return TREEBANK_LOOKUP[self.lang]

    @property
    def gate_str(self):
        if self.gate is not None:
            return "yes" if self.gate else "no"
        return None

    def get_path_name(self, extension, deprojz=False):
        base = f"lang={self.dir_abbr},pos={self.pos.lower()},"
        deprojz = ",deprojz=yes" if deprojz else ""
        if self.gate_str is not None:
            path = base + f"gate={self.gate}{deprojz}.{extension}"
        elif self.epochs is not None:
            if self.head is not None or self.path is not None:
                head = "yes" if self.head else "no"
                path = "yes" if self.path else "no"
                path = base + f"head={head},path={path},epochs={self.epochs}{deprojz}.{extension}"
            else:
                path = base + f"epochs={self.epochs}{deprojz}.{extension}"
        else:
            raise ValueError("Either gate or epochs must not be None.")
        return path

    def check_subfolder(self, subfolder):
        if subfolder not in ["neural", "rule_based", "label_experiments"]:
            raise ValueError("Must be 'neural', 'rule_based' or 'label_experiments'.")

    def constituentized(self) -> Path:
        if self.head is not None or self.path is not None:
            subfolder = "label_experiments"
            head = "yes" if self.head else "no"
            path = "yes" if self.path else "no"
            path = f"lang={self.dir_abbr},split=test,pos={self.pos.lower()},head={head},path={path},epochs={self.epochs}.mrg"
        else:
            subfolder = "raw"
            path = f"lang={self.dir_abbr},split=test,pos={self.pos.lower()},epochs={self.epochs}.mrg"
        folder = PREDICTION_DIR / subfolder
        # path = f"lang={self.dir_abbr},split=test,pos={self.pos.lower()},epochs={self.epochs}.mrg"
        check_dir(folder)
        return folder / path
    
    def structured_tokens(self) -> Path:
        folder = PREDICTION_DIR / "neural"
        path = f"lang={self.dir_abbr},pos={self.pos.lower()},gate={self.gate}.json"
        return folder / path

    def linearized(self) -> Path:
        folder = PREDICTION_DIR / "neural"
        path = self.get_path_name("txt")
        return folder / path

    def delinearized(self) -> Path:
        folder = PREDICTION_DIR / "neural"
        path = self.get_path_name("mrg")
        check_dir(folder)
        return folder / path

    def conllu(self, subfolder: str) -> Path:
        # subfolder = "neural" if is_neural else "rule_based"
        self.check_subfolder(subfolder)
        folder = PREDICTION_DIR / subfolder
        path = self.get_path_name("conllu")
        return folder / path

    def deprojectivized(self, subfolder: str):
        # subfolder = "neural" if is_neural else "rule_based"
        self.check_subfolder(subfolder)
        folder = PREDICTION_DIR / subfolder
        path = self.get_path_name("conllu", True)
        return folder / path






        

    
    

    

   