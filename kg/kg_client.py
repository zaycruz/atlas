import requests
import urllib.parse as up


class KG:
    def __init__(self, base: str = "http://localhost:4545") -> None:
        self.base = base.rstrip('/')

    def upsert(self, type_, props):
        r = requests.post(f"{self.base}/entity", json={"type": type_, "props": props}, timeout=10)
        r.raise_for_status()
        return r.json() if r.text else {"ok": True}

    def link(self, src, pred, dst):
        payload = {"srcId": src, "predicate": pred, "dstId": dst}
        r = requests.post(f"{self.base}/link", json=payload, timeout=10)
        r.raise_for_status()
        return r.json() if r.text else {"ok": True}

    def find(self, **query):
        r = requests.post(f"{self.base}/find", json=query, timeout=10)
        r.raise_for_status()
        return r.json()

    def lineage(self, id_, direction="up", max_depth=3):
        url = f"{self.base}/lineage/{up.quote(id_)}?direction={direction}&maxDepth={max_depth}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
