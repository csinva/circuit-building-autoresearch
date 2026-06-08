"""Generate configs.jsonl for the sweep. Wave 1: tune PERSON/PLACE after names."""
import json

configs = []


def add(name, **env):
    configs.append({"name": name, "env": env})


# Confirm harness reproduces baselines.
add("w1_names_base")                       # USE_NAMES=1 defaults -> expect 0.0797
add("w1_nonames", USE_NAMES=0)             # expect 0.0792

# PERSON_BONUS sweep (names densified PERSON; it had no dedicated bonus before).
for pb in [4, 8, 12, 16, 20, 24, 32, 40, 48, 64]:
    add(f"w1_person{pb}", PERSON_BONUS=pb)

# PLACE_BONUS re-tune (names densified PLACE too; default was 4).
for pl in [0, 8, 12, 16, 24, 32]:
    add(f"w1_place{pl}", PLACE_BONUS=pl)

# Promising PERSON x PLACE combos.
for pb in [8, 16, 24]:
    for pl in [8, 16]:
        add(f"w1_p{pb}_pl{pl}", PERSON_BONUS=pb, PLACE_BONUS=pl)

with open("configs.jsonl", "w") as f:
    for c in configs:
        f.write(json.dumps(c) + "\n")
print(f"wrote {len(configs)} configs")
