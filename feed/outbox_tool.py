"""
Prepare the outbox for upload to the Discovery Queue.

Uploading happens through Claude's Artifact tools (plots go to the page's
asset store, cards to its database), so this script only does the file
bookkeeping around that:

    python feed/outbox_tool.py pending              # cards waiting, with plot paths
    python feed/outbox_tool.py attach ids.json      # {"<card id>": "<asset id>", ...}
    python feed/outbox_tool.py sent <card id> ...   # move uploaded cards to feed/sent/

`attach` writes each uploaded plot's asset id into its card as `plot_asset`
and drops the `id` field (the card id is the database document id).
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTBOX, SENT = HERE / "outbox", HERE / "sent"


def pending():
    for card in sorted(OUTBOX.glob("*.json")):
        plot = card.with_suffix(".png")
        print(f"{card.stem}\t{plot if plot.exists() else '(no plot)'}")


def attach(mapping_path):
    mapping = json.loads(Path(mapping_path).read_text())
    for card_id, asset_id in mapping.items():
        path = OUTBOX / f"{card_id}.json"
        card = json.loads(path.read_text())
        card["plot_asset"] = asset_id
        card.pop("id", None)
        path.write_text(json.dumps(card, indent=1, ensure_ascii=False))
        print(f"attached {asset_id} to {card_id}")


def sent(card_ids):
    SENT.mkdir(exist_ok=True)
    for card_id in card_ids:
        for path in OUTBOX.glob(f"{card_id}.*"):
            shutil.move(str(path), SENT / path.name)
    print(f"moved {len(card_ids)} card(s) to {SENT}")


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    {"pending": lambda: pending(), "attach": lambda: attach(args[0]),
     "sent": lambda: sent(args)}[command]()
