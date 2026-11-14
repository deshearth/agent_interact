"""Apply the frozen read-only sensitivity implementation to the refined draft."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mac_agents.common import digest,json_write
from scripts import probe_argument_quality_predictors as base
from scripts import prepare_controlled_argument_probe_uncertain as draft
from scripts import joint_argument_encoding as session


if __name__ == "__main__":
    base.ROOT = session.SESSION/"quality_uncertain_probe"
    base.draft = draft
    base.main()
    json_write(base.ROOT/"adapter.json",{"adapter_sha256":digest(__file__),"implementation_sha256":digest(base.__file__),
        "draft_manifest_sha256":digest(draft.ROOT/"manifest.json"),"complete_sha256":digest(base.ROOT/"complete.json")})
