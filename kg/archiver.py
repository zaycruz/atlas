from datetime import datetime, timezone
from typing import Iterable, Optional

from kg_client import KG

kg = KG()

def archive_reflection(
    memory_id: str,
    project_id: str,
    topic_id: str,
    text_ref: str,
    derived_from_ids: Optional[Iterable[str]] = None,
    timespan_id: Optional[str] = None,
) -> None:
    kg.upsert('Memory', {
        'id': memory_id,
        'layer': 'reflection',
        'textRef': text_ref,
        'createdAt': datetime.now(timezone.utc).isoformat(),
    })
    kg.link(memory_id, 'IN_PROJECT', project_id)
    kg.link(memory_id, 'ABOUT', topic_id)
    if derived_from_ids:
        for source in derived_from_ids:
            kg.link(memory_id, 'DERIVED_FROM', source)
    if timespan_id:
        kg.link(memory_id, 'HAPPENED_AT', timespan_id)
