# web/views/chat.py
import os, json, re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from functools import lru_cache

from django.http import JsonResponse, HttpRequest
from django.shortcuts import render
from django.conf import settings
from django.views.decorators.csrf import ensure_csrf_cookie

import torch
from sentence_transformers import SentenceTransformer, util

# ============= تنظیمات پایه =============
CHAT_TEMPLATE = "chatbot/customer/page-bot-chat.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR: Path = Path(settings.BASE_DIR)
CHATBOT_DIR: Path = BASE_DIR / "chatbot"

QUESTIONS_FILE = CHATBOT_DIR / "questions_.json"                  # بانک سؤالات
DIFF_QUESTIONS_FILE = CHATBOT_DIR / "differential_questions.json" # سؤالات تمایز

SENTENCE_MODEL_NAME = settings.CHATBOT.get(
    "SENTENCE_MODEL", "paraphrase-multilingual-mpnet-base-v2"
)

BATCH_ITEMS_PER_FAMILY = 5
ST_BATCH_SIZE = int(os.getenv("ST_BATCH_SIZE", "8"))
BATCH_MAX_GROUPS = 12

# برچسب‌ها
DEFAULT_LABELS: Dict[str, str] = {
    "depression": "اختلالات خلقی مرتبط (افسردگی)",
    "bipolar": "اختلالات خلقی مرتبط (دوقطبی/مانیا)",
    "anxiety": "اختلالات اضطرابی",
    "ocd_related": "وسواس فکری‌عملی و اختلالات مرتبط",
    "trauma_stressor": "اختلالات مرتبط با تروما و استرسور",
    "psychosis": "طیف اسکیزوفرنی و اختلالات روان‌پریشی",
    "eating": "اختلالات خوردن",
    "sleep_wake": "اختلالات خواب و بیداری",
    "neurodev": "اختلالات عصبی‌رشدی",
    "dissociative": "اختلالات گسستی",
    "somatic": "سوماتیک/اضطراب بیماری",
    "substance": "اختلالات مصرف مواد/الکل/تنباکو",
    "sexual_function": "اختلالات عملکرد جنسی",
    "paraphilic": "پارافیلیک",
    "gender_identity": "دیفوریا/ناهماهنگی جنسیتی",  # ← اضافه شد
    "diff": "سؤالات تمایز",

    # سازگاری با لیبل‌های قدیمی عددی
    "0": "آپنهٔ انسدادی خواب (OSA)",
    "1": "عصبی/رشدی/زبان/خلقی (پایه/وسواس و…)",
    "2": "اضطراب/فوبیا/سوگ و مرتبط",
    "3": "شخصیت/نامشخص و دیگر",
    "4": "مصرف مواد/الکل/تنباکو",
    "5": "عملکرد جنسی/پارافیلیک",
    "6": "ADHD/یادگیری/هماهنگی",
    "7": "اختلالات خلقی مرتبط (افسردگی)",
    "8": "اختلالات خواب/ریتم/PMDD/DMDD",
    "9": "کودک/وابستگی/دفع/روان‌پریشی ناشی از ماده/جسمی",
    "10": "سایر",
    "22": "اختلالات خلقی مرتبط (دوقطبی/مانیا)"
}

# عبارات اضطراری
EMERGENCY_KEYWORDS = [
    "خودکشی","می‌خوام خودکشی","میخوام خودکشی","به خودم آسیب","کشتن خود",
    "می‌خوام خودمو تموم کنم","تمومش کنم","میرم خودمو بکشم","به دیگران آسیب","کشتن کسی","قتل"
]

# ============= ابزار لود =============
def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_question_bank(path: Path):
    obj = _load_json(path)
    bank: List[Dict[str, Any]] = obj.get("question_bank", [])
    by_symptom = {it.get("symptom"): it for it in bank if it.get("symptom")}
    by_id      = {it.get("id"): it for it in bank if it.get("id")}
    return bank, by_symptom, by_id

def _load_labels() -> Dict[str, str]:
    cand = CHATBOT_DIR / "disorder_labels.json"
    labels = DEFAULT_LABELS.copy()
    if cand.exists():
        try:
            custom = _load_json(cand)
            if isinstance(custom, dict):
                labels.update({str(k): str(v) for k, v in custom.items()})
        except Exception:
            pass
    return labels

def _load_diff_bank(path: Path) -> List[Dict[str, Any]]:
    try:
        arr = _load_json(path)
        if isinstance(arr, dict):
            arr = arr.get("diff_questions", [])
        ok = []
        for c in (arr or []):
            if isinstance(c, dict) and c.get("cluster") and isinstance(c.get("questions"), list):
                ok.append(c)
        return ok
    except Exception:
        return []

def _norm_label(s: str) -> str:
    s = re.sub(r"[\(\（][^)）]*[\)\）]", "", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()

def normalize_yes_no(v: Any) -> str:
    s = ("" if v is None else str(v)).strip().lower()
    return "yes" if s in ["بله","اره","آره","yes","y","true","۱","1","✔","✓","on"] else "no"

def check_emergency(text: str) -> bool:
    t = (text or "").replace("‌", " ").lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)

def token_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", (text or "").strip()) if w])

# Session sets <-> JSON
def _st_to_session(st: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(st)
    for k in ("asked_ids", "asked_norms"):
        if isinstance(out.get(k), set):
            out[k] = list(out[k])
    return out

def _st_from_session(obj: Dict[str, Any]) -> Dict[str, Any]:
    st = dict(obj)
    for k in ("asked_ids", "asked_norms"):
        if isinstance(st.get(k), list):
            st[k] = set(st[k])
    return st

# ============= داده‌های جهانی =============
try:
    _BANK, _BY_SYM, _BY_ID = _load_question_bank(QUESTIONS_FILE)
except FileNotFoundError:
    _BANK, _BY_SYM, _BY_ID = [], {}, {}

_LABELS = _load_labels()
_BANK_TITLES = [it.get("symptom", "") for it in _BANK]
_BANK_TITLES_NORM = [_norm_label(t) for t in _BANK_TITLES]
_DIFF_BANK = _load_diff_bank(DIFF_QUESTIONS_FILE)

# ============= مدل امبدینگ =============
@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    m = SentenceTransformer(SENTENCE_MODEL_NAME)
    try: m = m.to(DEVICE)
    except Exception: pass
    return m

@lru_cache(maxsize=1)
def get_bank_emb():
    if not _BANK_TITLES:
        return None
    m = get_model()
    emb = m.encode(
        _BANK_TITLES,
        convert_to_tensor=True,
        batch_size=ST_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    try: emb = emb.to(DEVICE)
    except Exception: pass
    return emb

# ============= امتیازدهی =============
def score_answer(meta_q: Dict[str, Any], value: Any) -> int:
    t = meta_q.get("response_type")
    if t == "yesno":
        return 1 if normalize_yes_no(value) == "yes" else 0
    if t == "likert_0_3":
        try: n = int(value)
        except Exception: n = 0
        return max(0, min(3, n))
    if t in ("open","text"):
        return 1 if str(value or "").strip() else 0
    return 0

def max_score_for(meta_q: Dict[str, Any]) -> int:
    t = meta_q.get("response_type")
    if t == "likert_0_3": return 3
    if t == "yesno": return 1
    if t in ("open","text"): return 1
    return 0

def severity_label(percent: float) -> str:
    if percent >= 66: return "زیاد"
    if percent >= 33: return "متوسط"
    return "کم"

# ============= کلیدواژه‌ها =============
def _has_any(text: str, vocab: Set[str]) -> bool:
    t = (text or "").replace("‌"," ").lower()
    return any(v.lower() in t for v in vocab)

# اضطراب/پانیک
KW_GAD_CORE: Set[str] = {"نگرانی","دلشوره","استرس","بی‌قراری","تنش","کنترل‌ناپذیر"}
KW_PANIC: Set[str] = {
    "حمله پانیک","حملهٔ پانیک","حمله وحشت","حمله اضطراب","پانیک",
    "تپش قلب","قلبم تند می‌زنه","قلبم تند میزنه",
    "تنگی نفس","نفس کم میارم","احساس خفگی","خفگی","نمی‌تونم نفس بکشم","نمیتونم نفس بکشم",
    "سرگیجه","سبکی سر","تعریق","لرزش","مورمور","بی‌حسی","گزگز",
    "ترس از مردن","می‌میرم الان","ترس از دیوونه شدن","کنترل از دست میره",
    "حمله ناگهانی","ناگهانی میاد","یهویی میاد"
}

# وسواس
KW_OCD: Set[str] = {"وسواس","افکار مزاحم","ناخواسته","اجبار","شستن","چک کردن","مرتب کردن","شمردن"}
KW_OCD_STRONG: Set[str] = {
    "کثیفه","کثیف","آلودگی","آلوده","نمی‌تونم به چیزی دست بزنم","می‌شورم","چند بار","مرتب می‌شورم","چک می‌کنم","ضدعفونی"
}

# مانیا/افسردگی
KW_SLEEP: Set[str] = {"بی‌خوابی","بی خواب","کم‌خوابی","پرخوابی","خواب","بیدار","صبح زود","کابوس","ریتم"}
KW_DEPRESSIVE: Set[str] = {"افسرد","غم","غمگین","ناامید","بی‌انگیزه","بی‌علاقه","لذت نمی‌برم","پرخوابی","پوچی","خستگی","حالم بده"}
KW_IRRITABILITY: Set[str] = {"عصبی","عصبانی","زودرنج","تحریک‌پذیر","تحریک پذیری"}
KW_MANIC: Set[str] = {
    "پرانرژی","انرژیم بالاست","کاهش نیاز به خواب","پرحرف",
    "ولخرجی","ریسکی","میل جنسی زیاد","خوشحال غیرعادی","تحریک‌پذیر",
    "مانیا","هیپومانیا","خلق بالا","افکار تندتند","نوسان خلق","بی‌قرار","تمرکز ندارم","حواس‌پرتی"
}

# جنسیت/دیفوریا و پارافیلیک
KW_GENDER_DYSPHORIA: Set[str] = {
    "با جنسیت خودم راحت نیستم","ناراحتی از جنسیت","دوست ندارم جنسیت خودم",
    "می‌خوام مرد باشم","می‌خوام زن باشم",
    "اسم خودمو صدا نزنن","ضمیر","می‌خوام با ضمیر دیگه صدام کنن",
    "دوست دارم لباس جنس مقابل بپوشم","نقش اجتماعی جنس دیگر","ویژگی‌های جنسی اذیتم می‌کنه",
    "دوست ندارم بدن/اندام جنسی فعلی"
}
KW_SEXUAL_AROUSAL_WORDS: Set[str] = {
    "تحریک","برانگیختگی","شهوت","لذت جنسی","فانتزی جنسی","برایم تحریک‌کننده است","ارگاسم"
}

# سایر دسته‌ها
KW_AVOIDANT_PD: Set[str] = {"اجتناب","طرد","نقد","کفایت","بی‌عرضگی","خجالت","کمرویی","تنهایی","فاصله","روابط صمیمی"}
KW_TRAUMA: Set[str] = {"تروما","حادثه","آزار","تصادف","جنگ","فاجعه","مرگ ناگهانی","تجاوز"}
KW_PTSD_SYMPTOMS: Set[str] = {"فلش‌بک","کابوس","اجتناب","گوش به زنگ","بی‌حسی هیجانی"}

KW_BINGE_EATING: Set[str] = {"پرخوری","مقدار زیاد غذا","کنترل از دست رفته","شرم","گناه"}
KW_COMPENSATORY_BEHAVIORS: Set[str] = {"استفراغ","ملین","ورزش زیاد","روزه","جبران"}
KW_EATING_TRIGGER: Set[str] = {"بی‌اشتهایی","لاغری","چاقی","وزن","رژیم","اندام","بدن","غذا"}

KW_SUBSTANCE: Set[str] = {"مواد","الکل","سیگار","قلیان","تریاک","شیشه","حشیش","ترک","دارو","اعتیاد"}
KW_MEDICAL: Set[str]   = {"بیماری جسمی","تیروئید","قلب","صرع","پارکینسون","دیابت"}

KW_SEXUAL_GENERAL: Set[str] = {"رابطه جنسی","سکس","میل جنسی","انزال","ارگاسم","درد هنگام رابطه"}
KW_SEXUAL_ED: Set[str] = {"نعوظ","نعوذ","سفت نمیشه","نعوظ سخت","قادر به نعوظ نیستم"}

KW_CHILDHOOD_ONSET: Set[str] = {"از کودکی","کودکی","قبل از ۱۲","قبل از12","قبل از دوازده"}
KW_ADHD: Set[str] = {"adhd","بیش‌فعالی","بیش فعالی","نقص توجه"}

KW_SHIFT: Set[str] = {"شیفت","شیفت کاری","نوبت‌کاری","شیفت شب"}
KW_PHASE: Set[str] = {"خیلی دیر می‌خوابم","دیر می‌خوابم","دیر بیدار می‌شم","تا دیروقت بیدارم"}

KW_BDD: Set[str] = {"بدریخت","بدشکلی","ظاهر","دماغ","پوست","آینه","عکس","پوشاندن","مقایسه"}
KW_HEALTH_ANX: Set[str] = {"بیماری جدی","سرطان","ام اس","ms","آزمایش می‌دم","چک می‌کنم بدن"}
KW_BPD: Set[str] = {"ترس از رها شدن","رابطه‌هام بالا پایین","قهر","مرزی","بی‌ثباتی هویت"}
KW_DISS: Set[str] = {"مسخ شخصیت","مسخ واقعیت","غیرواقعی","گسست","هویت","یادم نمیاد","فراموشی"}
KW_GRIEF: Set[str] = {"سوگ","عزا","عزاداری","فقدان","از دست دادم","مرگ","فوت"}
KW_PERIPARTUM: Set[str] = {"بارداری","حامله","زایمان","پس از زایمان","پیرامون‌زایمان","نوزاد","شیردهی"}
KW_EXCESSIVE_SLEEPINESS: Set[str] = {"خواب‌آلودگی","حملات خواب","کاتاپلکسی","چرت‌های ناگهانی"}

def is_mania_like(text: str) -> bool:
    return _has_any(text, KW_MANIC)

def is_grief_dominant(text: str) -> bool:
    t = (text or "")
    return _has_any(t, KW_GRIEF) and not is_mania_like(t)

def has_adhd_signal(text: str) -> bool:
    return (
        _has_any(text, {"adhd","بیش‌فعالی","بیش فعالی","نقص توجه"}) or
        (_has_any(text, {"تمرکز","حواس","بی‌قراری"}) and _has_any(text, KW_CHILDHOOD_ONSET))
    )

# ============= امبدینگ/رنکینگ =============
@torch.no_grad()
def rank_disorders_from_text(user_text: str, top_k: int = 5, min_sim: float = 0.45) -> List[Tuple[str, float, int]]:
    m = get_model()
    bank_emb = get_bank_emb()
    if bank_emb is None or not _BANK_TITLES:
        return []
    q = m.encode([user_text], convert_to_tensor=True, normalize_embeddings=True).to(DEVICE)
    sims = util.cos_sim(q, bank_emb)[0]

    best_by_did: Dict[str, Tuple[float, int]] = {}
    for i, it in enumerate(_BANK):
        did = str(it.get("disorder_id",""))
        if not did or not it.get("symptom"):
            continue
        s = float(sims[i])
        cur = best_by_did.get(did)
        if (cur is None) or (s > cur[0]):
            best_by_did[did] = (s, i)

    rows: List[Tuple[str, float, int]] = []
    for did,(sim_sem, idx) in best_by_did.items():
        if sim_sem < min_sim:
            continue
        rows.append((did, float(sim_sem), idx))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]

def pick_representative_items(rows: List[Tuple[str,float,int]]) -> List[Dict[str,Any]]:
    return [_BANK[idx] for _,_,idx in rows]

def _find_item_by_id(item_id: str) -> Optional[Dict[str, Any]]:
    if not item_id: return None
    return next((it for it in _BANK if it.get("id")==item_id), None)

def _find_representative_item_for_did(did: str,
                                      prefer_ids: Optional[List[str]] = None,
                                      prefer_symptom_subs: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    prefer_ids = prefer_ids or []
    prefer_symptom_subs = [s.lower() for s in (prefer_symptom_subs or [])]

    for it in _BANK:
        if str(it.get("disorder_id")) != did:
            continue
        if it.get("id") in prefer_ids:
            return it
    for it in _BANK:
        if str(it.get("disorder_id")) != did:
            continue
        sym = (it.get("symptom","") or "").lower()
        if any(sub in sym for sub in prefer_symptom_subs):
            return it
    for it in _BANK:
        if str(it.get("disorder_id")) == did:
            return it
    return None

# ============= ساخت Batch =============
def _group_from_item(it: Dict[str, Any]) -> Dict[str, Any]:
    gqs = []
    gw = (it.get("gateway") or {})
    gw_text = gw.get("text", "")
    timeframe = gw.get("timeframe_hint", "")
    if timeframe:
        gw_text = f"{gw_text}\n🕒 بازهٔ مدنظر: {timeframe}"
    gqs.append({"qid": gw.get("id"), "kind": "yesno", "text": gw_text, "required": False})

    for fq in (it.get("followups") or []):
        rt = fq.get("response_type")
        qobj = {"qid": fq.get("id"), "text": fq.get("text",""), "required": False}
        if rt == "yesno":
            qobj["kind"] = "yesno"
        elif rt == "likert_0_3":
            qobj["kind"] = "likert"; qobj["min"]=0; qobj["max"]=3
        else:
            qobj["kind"] = "text"; qobj["placeholder"] = "مثال یا توضیح کوتاه..."
        gqs.append(qobj)

    return {
        "title": it.get("symptom",""),
        "disorder_id": str(it.get("disorder_id","")),
        "questions": gqs
    }

def build_batch_spec_multi(user_text: str,
                           selected_items: List[Dict[str,Any]],
                           per_family: int = BATCH_ITEMS_PER_FAMILY,
                           max_groups: int = BATCH_MAX_GROUPS) -> Tuple[List[Dict[str,Any]], Dict[str,Any]]:
    model = get_model()
    bank_emb = get_bank_emb()
    q_emb = model.encode([user_text], convert_to_tensor=True, normalize_embeddings=True).to(DEVICE)

    picked_idx: List[int] = []
    seen_norm: Set[str] = set()

    for base in selected_items:
        did = str(base.get("disorder_id"))
        same_idx = [i for i,it in enumerate(_BANK) if str(it.get("disorder_id")) == did and it.get("symptom")]
        if bank_emb is not None and same_idx:
            sims = util.cos_sim(q_emb, bank_emb)[0]
            same_sorted = sorted(same_idx, key=lambda i: float(sims[i]), reverse=True)
        else:
            same_sorted = same_idx

        cnt = 0
        for i in same_sorted:
            lab = _norm_label(_BANK[i]["symptom"])
            if lab in seen_norm:
                continue
            seen_norm.add(lab)
            picked_idx.append(i)
            cnt += 1
            if cnt >= per_family:
                break
        if len(picked_idx) >= max_groups:
            break

    items = [_BANK[i] for i in picked_idx]
    groups = [_group_from_item(it) for it in items]
    spec = {"ui":"batch","groups": groups}
    return items, spec

# ============= فیلتر زمینه‌ای =============
def filter_groups_by_context(user_text: str, groups: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    has_substance   = _has_any(user_text, KW_SUBSTANCE)
    has_medical     = _has_any(user_text, KW_MEDICAL)
    mania_like      = is_mania_like(user_text)

    has_peripartum  = _has_any(user_text, KW_PERIPARTUM)
    has_eating      = _has_any(user_text, KW_BINGE_EATING) or _has_any(user_text, KW_COMPENSATORY_BEHAVIORS) or _has_any(user_text, KW_EATING_TRIGGER)
    has_excess_day  = _has_any(user_text, KW_EXCESSIVE_SLEEPINESS)
    has_child_adhd  = has_adhd_signal(user_text)

    # تمایز دیفوریا در برابر پارافیلیک: اگر «پوشیدن لباس جنس دیگر» بدون واژگان برانگیختگی جنسی بیاید،
    # گروه‌های پارافیلیکِ صرف را حذف می‌کنیم (تا جای درست یعنی Gender Dysphoria فعال بماند).
    mention_cross_dress = "لباس جنس دیگر" in (user_text or "")
    mention_arousal     = _has_any(user_text, KW_SEXUAL_AROUSAL_WORDS)

    out: List[Dict[str,Any]] = []
    seen_titles: Set[str] = set()

    for g in groups:
        title = (g.get("title") or "")
        tlow = title.lower()

        if ("ماده" in tlow or "مصرف" in tlow or "دارو" in tlow or "جسمی" in tlow):
            if not (has_substance or has_medical):
                continue

        if mania_like and any(k in tlow for k in ["نارکولپسی","آپنه","پاراسومنیا","ریتم خواب","ریتم","پرخوابی","بی‌خوابی"]):
            continue

        if any(k in tlow for k in ["زایمان","بارداری","پیرامون"]):
            if not has_peripartum:
                continue

        if any(k in tlow for k in ["پرخوری","بی‌اشتهایی","رومینیشن","پیکا","خوردن"]):
            if not has_eating:
                continue

        if any(k in tlow for k in ["نارکولپسی","خواب‌آلودگی","حملات خواب"]):
            if not has_excess_day:
                continue

        if any(k in tlow for k in ["adhd","بیش‌فعالی","نقص توجه","یادگیری","اوتیسم","تیک","tourette"]):
            if not has_child_adhd:
                continue

        # حذف پارافیلیک ترانسوستیک اگر کاربر فقط از پوشش/نقش گفت و اشاره‌ای به برانگیختگی جنسی نکرد
        if ("transvestic" in tlow or "ترانسوستیک" in tlow or "پوشیدن لباس جنس دیگر" in tlow):
            if mention_cross_dress and not mention_arousal:
                continue

        nl = _norm_label(title)
        if nl in seen_titles:
            continue
        seen_titles.add(nl)

        out.append(g)

    return out

# ============= تمایز (Diff) =============
def _label_has_any(did: str, subs: Set[str]) -> bool:
    lab = (_LABELS.get(did, "") or "").lower()
    return any(s in lab for s in subs)

def _rows_contain_labels(rows: List[Tuple[str,float,int]], subs: Set[str]) -> bool:
    return any(_label_has_any(did, subs) for did,_,_ in rows)

def need_mdd_vs_bipolar(text: str, rows) -> bool:
    if is_grief_dominant(text):
        return False
    dep_kw = _has_any(text, KW_DEPRESSIVE)
    cand_dep = _rows_contain_labels(rows, {"افسرد","depress"})
    if not (dep_kw or cand_dep):
        return False
    mania = is_mania_like(text)
    redflag = _has_any(text, KW_SLEEP) and _has_any(text, KW_IRRITABILITY)
    if mania or redflag:
        return True
    has_dep = cand_dep
    has_bip = _rows_contain_labels(rows, {"دو قطبی","دوقطبی","bipolar","مانیا"})
    return has_dep and has_bip

def need_gad_vs_ocd(text: str, rows) -> bool:
    return (_has_any(text, KW_GAD_CORE) and (_has_any(text, KW_OCD) or _has_any(text, KW_OCD_STRONG)))

def need_social_anxiety_vs_avoidant_pd(text: str, rows) -> bool:
    return _has_any(text, {"جمع","اجتماعی","قضاوت","مسخره"}) or _has_any(text, KW_AVOIDANT_PD)

def need_bed_vs_bulimia(text: str, rows) -> bool:
    return _has_any(text, KW_BINGE_EATING) or _has_any(text, KW_COMPENSATORY_BEHAVIORS)

def need_bipolar_vs_adhd(text: str, rows) -> bool:
    return is_mania_like(text) or has_adhd_signal(text)

def need_insomnia_vs_circadian(text: str, rows) -> bool:
    return _has_any(text, KW_SLEEP) and (_has_any(text, KW_SHIFT) or _has_any(text, KW_PHASE))

def need_ocd_vs_ocpd(text: str, rows) -> bool:
    return _has_any(text, KW_OCD) or _has_any(text, KW_OCD_STRONG)

def need_dysthymia_vs_mdd(text: str, rows) -> bool:
    return _has_any(text, KW_DEPRESSIVE)

def need_ptsd_vs_bpd(text: str, rows) -> bool:
    return (_has_any(text, KW_TRAUMA) or _has_any(text, KW_PTSD_SYMPTOMS)) or _has_any(text, KW_BPD)

def need_adhd_vs_depression(text: str, rows) -> bool:
    return has_adhd_signal(text) or (_has_any(text, KW_DEPRESSIVE) and _has_any(text, {"تمرکز","حواس"}))

def need_adhd_vs_anxiety(text: str, rows) -> bool:
    return has_adhd_signal(text) or _has_any(text, KW_GAD_CORE)

def need_atypical_vs_melancholic_depression(text: str, rows) -> bool:
    return _has_any(text, {"پرخوابی","پرخوری","صبح زود"}) or _has_any(text, KW_DEPRESSIVE)

def need_atypical_vs_dysthymia(text: str, rows) -> bool:
    return _has_any(text, KW_DEPRESSIVE)

def need_somatic_vs_mood_anxiety(text: str, rows) -> bool:
    return _has_any(text, KW_HEALTH_ANX) or _has_any(text, {"علائم جسمی","درد"})

def need_mixed_anxiety_depression(text: str, rows) -> bool:
    return _has_any(text, KW_DEPRESSIVE) and _has_any(text, KW_GAD_CORE)

def need_bdd_vs_sad_depression(text: str, rows) -> bool:
    return _has_any(text, KW_BDD) or _has_any(text, {"ظاهر","قیافه","دماغ"})

DIFF_NEED_FUNCS = {
    "mdd_vs_bipolar": need_mdd_vs_bipolar,
    "gad_vs_ocd": need_gad_vs_ocd,
    "social_anxiety_vs_avoidant_pd": need_social_anxiety_vs_avoidant_pd,
    "bed_vs_bulimia": need_bed_vs_bulimia,
    "bipolar_vs_adhd": need_bipolar_vs_adhd,
    "insomnia_vs_circadian": need_insomnia_vs_circadian,
    "ocd_vs_ocpd": need_ocd_vs_ocpd,
    "dysthymia_vs_mdd": need_dysthymia_vs_mdd,
    "ptsd_vs_bpd": need_ptsd_vs_bpd,
    "adhd_vs_depression": need_adhd_vs_depression,
    "adhd_vs_anxiety": need_adhd_vs_anxiety,
    "atypical_vs_melancholic_depression": need_atypical_vs_melancholic_depression,
    "atypical_vs_dysthymia": need_atypical_vs_dysthymia,
    "somatic_vs_mood_anxiety": need_somatic_vs_mood_anxiety,
    "mixed_anxiety_depression": need_mixed_anxiety_depression,
    "bdd_vs_sad_depression": need_bdd_vs_sad_depression,
    "did_vs_bpd_schizo": need_ptsd_vs_bpd,
}

def pick_diff_clusters(user_text: str, rows: List[Tuple[str,float,int]]) -> List[Dict[str,Any]]:
    res = []
    for cl in _DIFF_BANK:
        name = cl.get("cluster","")
        fn = DIFF_NEED_FUNCS.get(name)
        if not fn:
            continue
        try:
            if fn(user_text, rows):
                res.append(cl)
        except Exception:
            continue
    return res

def build_diff_batch_spec(clusters: List[Dict[str,Any]]) -> Dict[str, Any]:
    groups = []
    for cl in clusters:
        qs = []
        for q in (cl.get("questions") or []):
            rt = q.get("response_type","yesno")
            qobj = {"qid": q.get("id"), "text": q.get("text",""), "required": False}
            if rt == "yesno":
                qobj["kind"] = "yesno"
            elif rt == "likert_0_3":
                qobj["kind"] = "likert"; qobj["min"]=0; qobj["max"]=3
            elif rt == "multiple_choice":
                opts = q.get("options") or []
                labels = " / ".join([o.get("label","") for o in opts if isinstance(o, dict)])
                qobj["kind"] = "text"
                qobj["placeholder"] = f"انتخاب: {labels}" if labels else "انتخاب را بنویس..."
            else:
                qobj["kind"] = "text"; qobj["placeholder"] = "مثال یا توضیح کوتاه..."
            qs.append(qobj)

        groups.append({
            "title": cl.get("title",""),
            "disorder_id": "diff",
            "questions": qs
        })
    return {"ui":"batch", "groups": groups}

# ============= Heuristics: افزودن آیتم‌ها =============
def infer_extra_dids_and_items(user_text: str) -> Tuple[List[str], List[str]]:
    """
    خروجی: (extra_dids, direct_item_ids)
    direct_item_ids: آیتم‌هایی که باید صریحاً اضافه شوند (مثل ANX_PANIC، GENDER_dysphoria_adult)
    """
    extras: List[str] = []
    direct_items: List[str] = []
    t = (user_text or "")

    # پانیک: آیتم مستقیم
    if _has_any(t, KW_PANIC):
        direct_items.append("ANX_PANIC")

    # Gender Dysphoria در اولویت بالاتر از ترانسوستیک
    if _has_any(t, KW_GENDER_DYSPHORIA):
        # اگر واژگان برانگیختگی جنسی دیده نشود، به‌صورت مستقیم دیفوریا را اضافه کن
        if not _has_any(t, KW_SEXUAL_AROUSAL_WORDS):
            # این آیتم را باید در بانک داشته باشید
            direct_items.append("GENDER_dysphoria_adult")
        else:
            # هر دو را می‌توان آورد (دیفوریا و پارافیلیک)
            direct_items.append("GENDER_dysphoria_adult")
            extras.append("paraphilic")

    # الگوهای قبلی
    if is_mania_like(t):
        extras.append("bipolar")

    if _has_any(t, KW_DEPRESSIVE) and (_has_any(t, KW_SLEEP) or _has_any(t, KW_IRRITABILITY)):
        if "bipolar" not in extras:
            extras.append("bipolar")

    if _has_any(t, KW_OCD) or _has_any(t, KW_OCD_STRONG):
        extras.append("ocd_related")

    if _has_any(t, KW_SEXUAL_ED) or _has_any(t, KW_SEXUAL_GENERAL):
        extras.append("sexual_function")

    if _has_any(t, KW_SLEEP) and (not is_mania_like(t)):
        extras.append("sleep_wake")

    if _has_any(t, KW_GAD_CORE):
        extras.append("anxiety")

    if has_adhd_signal(t):
        extras.append("neurodev")

    # یکتا
    seen = set()
    extras_u = []
    for d in extras:
        if d not in seen:
            extras_u.append(d); seen.add(d)
    # direct unique
    seen = set()
    direct_u = []
    for d in direct_items:
        if d not in seen:
            direct_u.append(d); seen.add(d)

    return extras_u, direct_u

def _ensure_one_bipolar_gateway_if_dep_like(user_text: str, spec: Dict[str, Any]) -> None:
    if not (_has_any(user_text, KW_DEPRESSIVE) and ("groups" in spec)):
        return
    titles = " ".join([g.get("title","") for g in spec.get("groups",[])]).lower()
    if ("دو قطبی" in titles) or ("بایپولار" in titles) or ("bipolar" in titles) or ("هیپومانیا" in titles) or ("مانیا" in titles):
        return
    it = _find_representative_item_for_did(
        "bipolar",
        prefer_ids=["BP_mania_hypomania_screen"],
        prefer_symptom_subs=["هیپومانیا","مانیا","بالا رفتن","خلق بالا"]
    )
    if it:
        spec["groups"].insert(0, _group_from_item(it))

def _ensure_bipolar_gateway_if_mania_like(user_text: str, spec: Dict[str, Any]) -> None:
    if not (is_mania_like(user_text) and ("groups" in spec)):
        return
    titles = " ".join([g.get("title","") for g in spec.get("groups",[])]).lower()
    if ("دو قطبی" in titles) or ("بایپولار" in titles) or ("bipolar" in titles) or ("هیپومانیا" in titles) or ("مانیا" in titles):
        return
    it = _find_representative_item_for_did(
        "bipolar",
        prefer_ids=["BP_mania_hypomania_screen"],
        prefer_symptom_subs=["هیپومانیا","مانیا","بالا رفتن","خلق بالا"]
    )
    if it:
        spec["groups"].insert(0, _group_from_item(it))

# ============= View: صفحه =============
@ensure_csrf_cookie
def chat_page(request: HttpRequest):
    return render(request, CHAT_TEMPLATE, {})

# ============= View: API =============
@ensure_csrf_cookie
def chat_api(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "POST only"}, status=405)

    try:
        data = json.loads(request.body.decode("utf-8"))
    except Exception:
        data = {}

    action = (data.get("action") or "").strip()
    st = _st_from_session(request.session.get("chat_state", {}))

    def save_ok(payload: Dict[str, Any], *, reset: bool=False):
        if reset:
            st.clear()
        request.session["chat_state"] = _st_to_session(st)
        request.session.modified = True
        return JsonResponse(payload, safe=True)

    # ----------- پیام آزاد -----------
    if action == "" and "message" in data:
        msg = (data.get("message") or "").strip()
        if not msg:
            return save_ok({"ui":"text", "reply":"یه چیزی بنویس لطفاً 😊"})

        if check_emergency(msg):
            return save_ok({"ui":"text", "reply":"به نظر می‌رسه به کمک فوری نیاز داری. لطفاً همین الآن با اورژانس ۱۱۵ تماس بگیر یا با یکی از متخصصین ما صحبت کن. ❤️"})

        # 1) امبدینگ
        rows = rank_disorders_from_text(msg, top_k=5, min_sim=0.45)

        # 2) هیؤریستیک‌ها: DID/آیتم‌های مستقیم مثل پانیک و دیفوریا
        extra_dids, direct_item_ids = infer_extra_dids_and_items(msg)

        # اگر هیچ شباهت کافی نبود، از آیتم‌های مستقیم/دسته‌ها استفاده کن
        if not rows and (extra_dids or direct_item_ids):
            selected_items: List[Dict[str, Any]] = []

            # آیتم‌های مستقیم (مثلاً ANX_PANIC، GENDER_dysphoria_adult)
            for iid in direct_item_ids:
                it = _find_item_by_id(iid)
                if it: selected_items.append(it)

            # دسته‌های پیشنهادی
            for did in extra_dids:
                if did == "ocd_related":
                    rep = _find_representative_item_for_did("ocd_related", prefer_ids=["OCD_core"], prefer_symptom_subs=["وسواس"])
                elif did == "bipolar":
                    rep = _find_representative_item_for_did("bipolar", prefer_ids=["BP_mania_hypomania_screen"], prefer_symptom_subs=["هیپومانیا","مانیا","خلق بالا"])
                elif did == "sexual_function":
                    rep = _find_representative_item_for_did("sexual_function", prefer_ids=["SEX_ED","SEX_function"])
                elif did == "gender_identity":
                    rep = _find_representative_item_for_did("gender_identity", prefer_ids=["GENDER_dysphoria_adult"], prefer_symptom_subs=["دیفوریا","ناهماهنگی جنسیتی"])
                else:
                    rep = _find_representative_item_for_did(did)
                if rep: selected_items.append(rep)

            if selected_items:
                items, spec = build_batch_spec_multi(msg, selected_items, per_family=BATCH_ITEMS_PER_FAMILY, max_groups=BATCH_MAX_GROUPS)
                spec["groups"] = filter_groups_by_context(msg, spec["groups"])

                diff_clusters = pick_diff_clusters(msg, [])
                if diff_clusters:
                    diff_spec = build_diff_batch_spec(diff_clusters)
                    spec["groups"] = diff_spec["groups"] + spec["groups"]
                    st["diff_active"] = [cl.get("cluster") for cl in diff_clusters]

                _ensure_bipolar_gateway_if_mania_like(msg, spec)
                _ensure_one_bipolar_gateway_if_dep_like(msg, spec)

                st["mode"] = "batch"
                st["user_text"] = msg
                st["batch_items_ids"] = [it.get("id") for it in items if it.get("id")]
                return save_ok(spec)

            return save_ok({"ui":"text", "reply":"هنوز مطمئن نیستم. لطفاً کمی بیشتر دربارهٔ علائمت توضیح بده."})

        # اگر rows داریم:
        selected_items = pick_representative_items(rows)

        # آیتم‌های مستقیم را جلوتر تزریق کن
        for iid in direct_item_ids:
            it = _find_item_by_id(iid)
            if it and it not in selected_items:
                selected_items.insert(0, it)

        # دسته‌های اضافی
        existing_dids = {str(it.get("disorder_id")) for it in selected_items}
        for did in extra_dids:
            if did in existing_dids: 
                continue
            if did == "ocd_related":
                rep = _find_representative_item_for_did("ocd_related", prefer_ids=["OCD_core"], prefer_symptom_subs=["وسواس"])
            elif did == "bipolar":
                rep = _find_representative_item_for_did("bipolar", prefer_ids=["BP_mania_hypomania_screen"], prefer_symptom_subs=["هیپومانیا","مانیا","خلق بالا"])
            elif did == "sexual_function":
                rep = _find_representative_item_for_did("sexual_function", prefer_ids=["SEX_ED","SEX_function"])
            elif did == "gender_identity":
                rep = _find_representative_item_for_did("gender_identity", prefer_ids=["GENDER_dysphoria_adult"], prefer_symptom_subs=["دیفوریا","ناهماهنگی جنسیتی"])
            else:
                rep = _find_representative_item_for_did(did)
            if rep:
                selected_items.append(rep)
                existing_dids.add(did)

        items, spec = build_batch_spec_multi(msg, selected_items, per_family=BATCH_ITEMS_PER_FAMILY, max_groups=BATCH_MAX_GROUPS)
        spec["groups"] = filter_groups_by_context(msg, spec["groups"])

        if not spec["groups"]:
            return save_ok({"ui":"text", "reply":"علائمی که گفتی واضح نبود. کمی دقیق‌تر بگو چه چیزهایی اذیتت می‌کنه."})

        diff_clusters = pick_diff_clusters(msg, rows)
        if diff_clusters:
            diff_spec = build_diff_batch_spec(diff_clusters)
            spec["groups"] = diff_spec["groups"] + spec["groups"]
            st["diff_active"] = [cl.get("cluster") for cl in diff_clusters]

        _ensure_bipolar_gateway_if_mania_like(msg, spec)
        _ensure_one_bipolar_gateway_if_dep_like(msg, spec)

        st["mode"] = "batch"
        st["user_text"] = msg
        st["batch_items_ids"] = [it.get("id") for it in items if it.get("id")]

        return save_ok(spec)

    # ----------- دریافت پاسخ فرم -----------
    if action == "batch_submit":
        answers: Dict[str, Any] = data.get("answers") or {}

        total_by_dis: Dict[str, int] = {}
        max_by_dis: Dict[str, int] = {}

        shown_item_ids: Set[str] = set(st.get("batch_items_ids") or [])

        qmeta: Dict[str, Dict[str, Any]] = {}
        for it in _BANK:
            if shown_item_ids and it.get("id") not in shown_item_ids:
                continue
            did = str(it.get("disorder_id"))
            gid = (it.get("gateway") or {}).get("id")
            if gid:
                qmeta[gid] = {"disorder_id": did, "response_type": "yesno"}
            for fq in (it.get("followups") or []):
                qid = fq.get("id")
                if not qid:
                    continue
                qmeta[qid] = {"disorder_id": did, "response_type": fq.get("response_type")}

        def _default_for(rt: Optional[str]) -> Any:
            if rt == "yesno":
                return "no"
            if rt == "likert_0_3":
                return 0
            return ""

        for qid, meta in qmeta.items():
            did = meta["disorder_id"]
            rt  = meta.get("response_type")
            val = answers.get(qid, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                val = _default_for(rt)
            sc = score_answer(meta, val)
            mx = max_score_for(meta)
            total_by_dis[did] = total_by_dis.get(did, 0) + sc
            max_by_dis[did]   = max_by_dis.get(did, 0) + mx

        results = []
        for did, sc in total_by_dis.items():
            mx = max_by_dis.get(did, 0) or 1
            pct = round(100.0 * sc / mx, 1)
            if sc > 0:
                results.append({
                    "disorder_id": did,
                    "label": _LABELS.get(did, f"اختلال {did}"),
                    "score": sc,
                    "max": mx,
                    "percent": pct,
                    "severity": severity_label(pct)
                })

        if not results:
            return save_ok({"ui":"text","reply":"بر اساس پاسخ‌ها نشانهٔ فعالی تأیید نشد. می‌تونی فقط به سؤال‌هایی که دوست داری جواب بدی؛ بقیه به‌صورت «خیر» درنظر گرفته می‌شن."})

        results.sort(key=lambda r: (-r["percent"], -r["score"]))
        lines = ["نتیجهٔ غربالگری (غیردقیق/غیرتشخیصی):"]
        for r in results[:6]:
            lines.append(f"• {r['label']} — امتیاز {r['score']}/{r['max']} (٪{r['percent']})")

        return save_ok({"ui":"text", "reply":"\n".join(lines)}, reset=True)

    return JsonResponse({"ok": False, "error": "bad request"}, status=400)
