import re

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being it its this that
these those i you he she we they them his her our your their not no yes do does did done can could should would may might will
shall have has had about into over under between across what which who whom whose when where why how also than so such very more
most some any each other only own same too just up down out off again further once here there all both few per via using use
used""".split())

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokens(text, keep_stop=False):
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    return toks if keep_stop else [t for t in toks if t not in STOP and len(t) > 1]


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def chunk_text(text, size, overlap):
    """Paragraph-aware word chunking: pack paragraphs up to `size` words, carry `overlap` words."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], []
    for p in paras:
        words = p.split()
        while len(words) > size:  # oversized paragraph: hard split
            if cur:
                chunks.append(" ".join(cur)); cur = []
            chunks.append(" ".join(words[:size]))
            words = words[size - overlap:]
        if len(cur) + len(words) > size and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
        cur.extend(words)
    if cur:
        chunks.append(" ".join(cur))
    return chunks
