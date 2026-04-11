import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_FILE = os.path.join(BASE_DIR, "audit_logs.jsonl")


def log_interaction(user, query, answer, sources):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user,
        "query": query,
        "answer": answer,
        "sources": list(set(sources))
    }

    try:
        with open(AUDIT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        print("Audit log written")   # DEBUG
    except Exception as e:
        print("Audit log failed:", e)